// Batched shared-dt lockstep RK45 integrator — correctness gate + realized
// throughput/efficiency measurement (0006).
//
// Two metrics, both against the scalar run_to_convergence on the SAME geometries:
//   1. Lockstep efficiency (step-count, ISA-independent): sum(scalar steps) /
//      (K * sum(batch SIMD-iterations)). Comparable to 0002's ceiling (0.63 at
//      K=8, 0.57 at K=16) — now measured for real, including step-rejection.
//   2. Realized wall-time throughput (sims/sec). The headline is the f32
//      production path (K=16, rtol 1e-4 / pe_stop 1e-5 — the ADR 0004 operating
//      point) vs the f64 default scalar driver (rtol 1e-8 / pe_stop 1e-9): the
//      number 0005 projected at ~16x.
//
// Built -march=native (static-dispatch Highway integrator); linked against the
// generic engine for the scalar oracle and the sampler. End-to-end convergence
// runs are timed directly with controlled repetitions (median + cv) rather than
// via Google Benchmark microbenchmark semantics.
//
// Run: ./build/relwithdebinfo/bench/coulomb_batched_integrator [--atoms N]
//        [--batches M] [--reps R] [--csv path]

#include <hwy/highway.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "coulomb/batched_integrator.hpp"
#include "coulomb/driver.hpp"
#include "coulomb/integrator.hpp"
#include "coulomb/molecule.hpp"
#include "coulomb/sampler.hpp"
#include "coulomb/system.hpp"

namespace hn = hwy::HWY_NAMESPACE;
using namespace coulomb;
using Clock = std::chrono::steady_clock;

namespace {

// N atoms cycling H, C, N, O (singly ionized) — the 0002/0004 chemistry.
Molecule make_molecule(std::size_t n) {
  struct Element {
    const char* symbol;
    Real amu;
  };
  static const Element kCycle[] = {
      {"H", Real{1.008}}, {"C", Real{12.011}}, {"N", Real{14.007}}, {"O", Real{15.999}}};
  Molecule m;
  m.atoms.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const Element& e = kCycle[i % 4];
    m.atoms.push_back(atom_from_amu(e.symbol, e.amu, Real{1.0}));
  }
  return m;
}

std::vector<State> sample_pool(const Molecule& mol, std::size_t count, std::uint64_t seed) {
  UniformSphereSampler sampler({.radius = Real{4.0}, .min_separation = Real{0.25}}, seed);
  std::vector<State> pool(count);
  for (std::size_t m = 0; m < count; ++m) sampler.sample(mol, pool[m]);
  return pool;
}

// Median and coefficient of variation of per-rep wall times (seconds).
std::pair<double, double> median_cv(std::vector<double> secs) {
  std::sort(secs.begin(), secs.end());
  const std::size_t r = secs.size();
  const double med = (r % 2) ? secs[r / 2] : 0.5 * (secs[r / 2 - 1] + secs[r / 2]);
  double mean = 0.0;
  for (double s : secs) mean += s;
  mean /= static_cast<double>(r);
  double var = 0.0;
  for (double s : secs) var += (s - mean) * (s - mean);
  const double sd = std::sqrt(var / static_cast<double>(r));
  return {med, mean > 0 ? sd / mean : 0.0};
}

// ---------------------------------------------------------------------------
// Correctness gate: batched per-lane momenta vs scalar run_to_convergence.
// ---------------------------------------------------------------------------

double config_rel(const double* pb, const std::vector<Vec3>& ps, std::size_t n) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = pb[i * 3 + 0] - ps[i].x;
    const double dy = pb[i * 3 + 1] - ps[i].y;
    const double dz = pb[i * 3 + 2] - ps[i].z;
    num += dx * dx + dy * dy + dz * dz;
    den += ps[i].x * ps[i].x + ps[i].y * ps[i].y + ps[i].z * ps[i].z;
  }
  return std::sqrt(num) / std::sqrt(den);
}

bool run_gate(const Molecule& mol, std::size_t n, double rtol, double atol) {
  const std::size_t k = hn::Lanes(hn::ScalableTag<double>());
  const std::vector<State> geoms = sample_pool(mol, k, 0xC0FFEE);
  RunConfig cfg;

  BatchedRK45<double> bi(mol, Real{1.0}, k, rtol, atol);
  for (std::size_t lane = 0; lane < k; ++lane) bi.set_lane_geometry(lane, geoms[lane].positions);
  const auto res = bi.run(cfg);

  CoulombForce force(Real{1.0});
  double max_rel = 0.0;
  std::size_t n_conv = 0;
  for (std::size_t lane = 0; lane < k; ++lane) {
    State s = geoms[lane];
    auto integ = make_integrator(IntegratorKind::RK45, IntegratorOptions{Real(rtol), Real(atol)});
    run_to_convergence(mol, force, *integ, s, cfg);
    std::vector<Vec3> ps(n);
    for (std::size_t i = 0; i < n; ++i) ps[i] = s.velocities[i] * mol.atoms[i].mass;
    max_rel = std::max(max_rel, config_rel(&res.momenta[lane * n * 3], ps, n));
    if (res.converged[lane]) ++n_conv;
  }
  const double gate_tol = std::max(50.0 * rtol, 1e-6);
  std::printf("[gate] batched vs scalar max |dP|/|P| = %.3e (N=%zu K=%zu rtol=%.0e, tol=%.1e)\n",
              max_rel, n, k, rtol, gate_tol);
  return (max_rel < gate_tol) && n_conv == k;
}

// ---------------------------------------------------------------------------
// Measurement.
// ---------------------------------------------------------------------------

struct BatchedMeasure {
  double sims_per_sec{0};
  double cv{0};
  std::size_t batch_steps_total{0};  // summed SIMD-iterations over all batches.
  std::size_t n_batches{0};
};

template <class T>
BatchedMeasure measure_batched(const Molecule& mol, const std::vector<State>& pool, double rtol,
                               double atol, const RunConfig& cfg, int reps) {
  const std::size_t K = hn::Lanes(hn::ScalableTag<T>());
  const std::size_t nb = pool.size() / K;
  BatchedRK45<T> bi(mol, Real{1.0}, K, static_cast<T>(rtol), static_cast<T>(atol));

  auto one_pass = [&](std::size_t* steps_out) {
    double checksum = 0.0;
    std::size_t steps = 0;
    for (std::size_t b = 0; b < nb; ++b) {
      for (std::size_t lane = 0; lane < K; ++lane)
        bi.set_lane_geometry(lane, pool[b * K + lane].positions);
      const auto r = bi.run(cfg);
      steps += r.batch_steps;
      checksum += r.momenta[0];
    }
    if (steps_out) *steps_out = steps;
    return checksum;
  };

  BatchedMeasure m;
  m.n_batches = nb;
  one_pass(&m.batch_steps_total);  // untimed accounting pass.

  std::vector<double> secs;
  volatile double sink = 0.0;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clock::now();
    sink += one_pass(nullptr);
    secs.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
  }
  const auto [med, cv] = median_cv(secs);
  m.sims_per_sec = static_cast<double>(nb * K) / med;
  m.cv = cv;
  return m;
}

struct ScalarMeasure {
  double sims_per_sec{0};
  double cv{0};
  std::size_t steps_total{0};
};

ScalarMeasure measure_scalar(const Molecule& mol, const std::vector<State>& pool, double rtol,
                             double atol, const RunConfig& cfg, int reps) {
  CoulombForce force(Real{1.0});
  auto one_pass = [&](std::size_t* steps_out) {
    double checksum = 0.0;
    std::size_t steps = 0;
    for (const State& g : pool) {
      State s = g;
      auto integ = make_integrator(IntegratorKind::RK45, IntegratorOptions{Real(rtol), Real(atol)});
      const auto rr = run_to_convergence(mol, force, *integ, s, cfg);
      steps += rr.steps;
      checksum += s.velocities[0].x;
    }
    if (steps_out) *steps_out = steps;
    return checksum;
  };

  ScalarMeasure m;
  one_pass(&m.steps_total);

  std::vector<double> secs;
  volatile double sink = 0.0;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = Clock::now();
    sink += one_pass(nullptr);
    secs.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
  }
  const auto [med, cv] = median_cv(secs);
  m.sims_per_sec = static_cast<double>(pool.size()) / med;
  m.cv = cv;
  return m;
}

struct Args {
  std::size_t atoms = 10;
  std::size_t batches = 256;  // pool size = batches * 16.
  int reps = 7;
  std::string csv;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    const bool has = i + 1 < argc;
    auto next = [&] { return std::string(argv[++i]); };
    if (f == "--atoms" && has)
      a.atoms = static_cast<std::size_t>(std::stoul(next()));
    else if (f == "--batches" && has)
      a.batches = static_cast<std::size_t>(std::stoul(next()));
    else if (f == "--reps" && has)
      a.reps = std::stoi(next());
    else if (f == "--csv" && has)
      a.csv = next();
    else {
      std::fprintf(stderr, "unknown or incomplete argument: %s\n", f.c_str());
      std::exit(EXIT_FAILURE);
    }
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  const Molecule mol = make_molecule(args.atoms);

  // Guard: correctness must hold before any timing is meaningful.
  if (!run_gate(mol, args.atoms, 1e-8, 1e-16)) {
    std::printf("GATE FAILED — aborting measurement\n");
    return 1;
  }

  // Two operating points: f64 default (the 0005 baseline) and the f32 production
  // point (ADR 0004: rtol 1e-4 / atol 1e-7 / pe_stop 1e-5).
  RunConfig cfg_default;  // pe_stop 1e-9.
  RunConfig cfg_prod = cfg_default;
  cfg_prod.pe_stop_fraction = Real{1e-5};
  constexpr double kRtolDef = 1e-8, kAtolDef = 1e-16;
  constexpr double kRtolProd = 1e-4, kAtolProd = 1e-7;

  const std::size_t pool_size = args.batches * 16;  // divisible by both K=8 and K=16.
  const std::vector<State> pool = sample_pool(mol, pool_size, 0x5EED);
  const std::size_t k8 = hn::Lanes(hn::ScalableTag<double>());
  const std::size_t k16 = hn::Lanes(hn::ScalableTag<float>());

  std::printf("\nmeasurement: N=%zu  pool=%zu sims  reps=%d  (K f64=%zu, f32=%zu)\n", args.atoms,
              pool_size, args.reps, k8, k16);

  const auto sc_def = measure_scalar(mol, pool, kRtolDef, kAtolDef, cfg_default, args.reps);
  const auto sc_prod = measure_scalar(mol, pool, kRtolProd, kAtolProd, cfg_prod, args.reps);
  const auto bd_def =
      measure_batched<double>(mol, pool, kRtolDef, kAtolDef, cfg_default, args.reps);
  const auto bd_prod =
      measure_batched<double>(mol, pool, kRtolProd, kAtolProd, cfg_prod, args.reps);
  const auto bf_def = measure_batched<float>(mol, pool, kRtolDef, kAtolDef, cfg_default, args.reps);
  const auto bf_prod = measure_batched<float>(mol, pool, kRtolProd, kAtolProd, cfg_prod, args.reps);

  auto eff = [](std::size_t scalar_steps, std::size_t K, const BatchedMeasure& b) {
    return static_cast<double>(scalar_steps) /
           (static_cast<double>(K) * static_cast<double>(b.batch_steps_total));
  };
  const double eff_bd_def = eff(sc_def.steps_total, k8, bd_def);
  const double eff_bd_prod = eff(sc_prod.steps_total, k8, bd_prod);
  const double eff_bf_def = eff(sc_def.steps_total, k16, bf_def);
  const double eff_bf_prod = eff(sc_prod.steps_total, k16, bf_prod);

  // Two baselines: the actual current generator (f64 scalar) and the
  // apples-to-apples batched baseline 0005's per-step ratio implies (f64 batched).
  const double realized_vs_scalar = bf_prod.sims_per_sec / sc_def.sims_per_sec;
  const double realized_vs_batched = bf_prod.sims_per_sec / bd_def.sims_per_sec;

  std::printf("\n%-26s %5s %8s %6s %6s  %s\n", "row", "K", "sims/s", "cv%", "eff", "mean steps");
  auto line = [](const char* lbl, std::size_t K, double sps, double cv, double e, double steps) {
    std::printf("%-26s %5zu %8.3g %6.2f %6.3f  %.1f\n", lbl, K, sps, 100 * cv, e, steps);
  };
  line("scalar f64 default", 1, sc_def.sims_per_sec, sc_def.cv, 0.0,
       static_cast<double>(sc_def.steps_total) / pool_size);
  line("scalar f64 prod", 1, sc_prod.sims_per_sec, sc_prod.cv, 0.0,
       static_cast<double>(sc_prod.steps_total) / pool_size);
  line("batched f64 default", k8, bd_def.sims_per_sec, bd_def.cv, eff_bd_def,
       static_cast<double>(bd_def.batch_steps_total) / bd_def.n_batches);
  line("batched f64 prod", k8, bd_prod.sims_per_sec, bd_prod.cv, eff_bd_prod,
       static_cast<double>(bd_prod.batch_steps_total) / bd_prod.n_batches);
  line("batched f32 default", k16, bf_def.sims_per_sec, bf_def.cv, eff_bf_def,
       static_cast<double>(bf_def.batch_steps_total) / bf_def.n_batches);
  line("batched f32 prod", k16, bf_prod.sims_per_sec, bf_prod.cv, eff_bf_prod,
       static_cast<double>(bf_prod.batch_steps_total) / bf_prod.n_batches);

  std::printf("\nrealized speedup, f32 batched prod (K=16) vs:\n");
  std::printf("  f64 scalar default  = %.2fx  (the actual current generator)\n",
              realized_vs_scalar);
  std::printf("  f64 batched default = %.2fx  (apples-to-apples; 0005 projected ~16x)\n",
              realized_vs_batched);
  std::printf(
      "  0002 efficiency ceiling: 0.63 (K=8), 0.57 (K=16); f32 'default' is sub-eps "
      "thrash (see 0004)\n");

  if (!args.csv.empty()) {
    std::ofstream out(args.csv);
    out << "# N=" << args.atoms << " pool=" << pool_size << " reps=" << args.reps
        << " realized_vs_scalar=" << realized_vs_scalar
        << " realized_vs_batched=" << realized_vs_batched << "\n";
    out << "row,precision,lanes,rtol,pe_stop,sims_per_sec,cv,efficiency,mean_batch_steps\n";
    auto row = [&](const char* lbl, const char* prec, std::size_t K, double rtol, double pe,
                   double sps, double cv, double e, double steps) {
      out << lbl << ',' << prec << ',' << K << ',' << rtol << ',' << pe << ',' << sps << ',' << cv
          << ',' << e << ',' << steps << '\n';
    };
    row("scalar_default", "f64", 1, kRtolDef, 1e-9, sc_def.sims_per_sec, sc_def.cv, 0.0,
        static_cast<double>(sc_def.steps_total) / pool_size);
    row("scalar_prod", "f64", 1, kRtolProd, 1e-5, sc_prod.sims_per_sec, sc_prod.cv, 0.0,
        static_cast<double>(sc_prod.steps_total) / pool_size);
    row("batched_default", "f64", k8, kRtolDef, 1e-9, bd_def.sims_per_sec, bd_def.cv, eff_bd_def,
        static_cast<double>(bd_def.batch_steps_total) / bd_def.n_batches);
    row("batched_prod", "f64", k8, kRtolProd, 1e-5, bd_prod.sims_per_sec, bd_prod.cv, eff_bd_prod,
        static_cast<double>(bd_prod.batch_steps_total) / bd_prod.n_batches);
    row("batched_default", "f32", k16, kRtolDef, 1e-9, bf_def.sims_per_sec, bf_def.cv, eff_bf_def,
        static_cast<double>(bf_def.batch_steps_total) / bf_def.n_batches);
    row("batched_prod", "f32", k16, kRtolProd, 1e-5, bf_prod.sims_per_sec, bf_prod.cv, eff_bf_prod,
        static_cast<double>(bf_prod.batch_steps_total) / bf_prod.n_batches);
    std::printf("\nCSV written to %s\n", args.csv.c_str());
  }
  return 0;
}
