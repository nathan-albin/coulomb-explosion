// Dispersion study for the SIMD-over-lanes decision (not a microbenchmark).
//
// The production workload is millions of independent Coulomb explosions of a
// small molecule sampled at random geometries. Running them with adaptive RK45
// means each sim takes a different number of steps, and the *spread* of that
// step count across geometries decides which batching strategy is worth
// building:
//
//   * Shared-dt lockstep (simple): a batch of K sims shares one clock and steps
//     at the minimum step size any lane demands, finishing only when its slowest
//     lane converges. It pays two penalties — the min-envelope (everyone slows to
//     the lane in its tightest close-encounter) and straggler idle (converged
//     lanes sit masked until the batch ends). No per-lane control flow needed.
//   * Refill / wavefront (complex): per-lane clocks and step sizes, with
//     converged lanes immediately refilled from a queue. No min-envelope and no
//     straggler idle, so the speedup approaches K — at the cost of divergent
//     control flow, masking, and gather/scatter refill.
//
// For M sampled configurations of the 8-atom mixed-mass example this harness
// records each sim's accepted-step count and full step-size-vs-time trace, then
// estimates the shared-dt efficiency by Monte-Carlo over random batches of K.
// The gap to the refill ceiling (~K) is exactly what the wavefront machinery
// would buy. A per-sim CSV is written for external analysis/plotting.
//
// Run (built under the relwithdebinfo preset):
//   ./build/relwithdebinfo/bench/coulomb_explosion [--sims N] [--seed S]
//       [--csv PATH] [--lanes "4,8,16"]

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "coulomb/driver.hpp"
#include "coulomb/integrator.hpp"
#include "coulomb/molecule.hpp"
#include "coulomb/sampler.hpp"
#include "coulomb/system.hpp"

using namespace coulomb;

namespace {

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

// The 8-atom, mixed-mass example: four masses (H, C, N, O), singly ionized.
Molecule make_example_molecule() {
  Molecule m;
  m.atoms = {
      atom_from_amu("H", Real{1.008}, Real{1.0}),  atom_from_amu("C", Real{12.011}, Real{1.0}),
      atom_from_amu("N", Real{14.007}, Real{1.0}), atom_from_amu("O", Real{15.999}, Real{1.0}),
      atom_from_amu("H", Real{1.008}, Real{1.0}),  atom_from_amu("C", Real{12.011}, Real{1.0}),
      atom_from_amu("N", Real{14.007}, Real{1.0}), atom_from_amu("O", Real{15.999}, Real{1.0}),
  };
  return m;
}

Real min_pair_distance(const State& s) {
  Real best = std::numeric_limits<Real>::infinity();
  for (std::size_t i = 0; i < s.size(); ++i) {
    for (std::size_t j = i + 1; j < s.size(); ++j) {
      best = std::min(best, norm(s.positions[i] - s.positions[j]));
    }
  }
  return best;
}

// One simulation's outcome plus the data the batch models need.
struct SimRecord {
  std::size_t steps{0};
  Real t_final{0};
  Real min_init_sep{0};
  Real e_initial{0};
  Real pe_final{0};
  // (t_start, dt) for each accepted step, in order. Spans [t_start, t_start+dt).
  std::vector<std::pair<Real, Real>> trace;
};

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

struct Stats {
  double mean{0}, sd{0}, cv{0};
  double min{0}, p50{0}, p90{0}, p99{0}, max{0};
};

// Linear-interpolation percentile on an already-sorted sample.
double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0.0;
  if (sorted.size() == 1) return sorted.front();
  const double rank = q * static_cast<double>(sorted.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(rank));
  const auto hi = static_cast<std::size_t>(std::ceil(rank));
  const double frac = rank - static_cast<double>(lo);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * frac;
}

Stats summarize(std::vector<double> xs) {
  Stats s;
  if (xs.empty()) return s;
  const double n = static_cast<double>(xs.size());
  const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
  s.mean = sum / n;
  double var = 0.0;
  for (double x : xs) var += (x - s.mean) * (x - s.mean);
  s.sd = std::sqrt(var / n);
  s.cv = s.mean != 0.0 ? s.sd / s.mean : 0.0;
  std::sort(xs.begin(), xs.end());
  s.min = xs.front();
  s.max = xs.back();
  s.p50 = percentile(xs, 0.50);
  s.p90 = percentile(xs, 0.90);
  s.p99 = percentile(xs, 0.99);
  return s;
}

double pearson(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = a.size();
  if (n < 2) return 0.0;
  const double na = static_cast<double>(n);
  const double ma = std::accumulate(a.begin(), a.end(), 0.0) / na;
  const double mb = std::accumulate(b.begin(), b.end(), 0.0) / na;
  double sab = 0.0, saa = 0.0, sbb = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double da = a[i] - ma, db = b[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  const double denom = std::sqrt(saa * sbb);
  return denom > 0.0 ? sab / denom : 0.0;
}

// ---------------------------------------------------------------------------
// Shared-dt lockstep model
// ---------------------------------------------------------------------------

// SIMD-iterations to drive one batch of K sims under the shared-dt rule: every
// active lane advances by the minimum step any active lane demands at the shared
// clock, lanes retire as the clock passes their t_final, and the batch runs
// until the last lane retires. A lane's "demanded dt" at clock T is the step it
// was taking when its own clock was at T — read off its recorded trace.
std::size_t shared_dt_iters(const std::vector<const SimRecord*>& batch) {
  const std::size_t k = batch.size();
  std::vector<std::size_t> idx(k, 0);
  std::vector<char> active(k, 1);
  std::size_t n_active = k;
  Real t = 0;
  std::size_t iters = 0;

  while (n_active > 0) {
    Real dt_shared = std::numeric_limits<Real>::infinity();
    for (std::size_t lane = 0; lane < k; ++lane) {
      if (!active[lane]) continue;
      if (t >= batch[lane]->t_final) {
        active[lane] = 0;
        --n_active;
        continue;
      }
      const auto& tr = batch[lane]->trace;
      // Advance this lane's cursor to the step segment containing the clock.
      while (idx[lane] + 1 < tr.size() && tr[idx[lane] + 1].first <= t) ++idx[lane];
      dt_shared = std::min(dt_shared, tr[idx[lane]].second);
    }
    if (n_active == 0) break;
    t += dt_shared;
    ++iters;
  }
  return iters;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
  std::size_t sims = 4000;
  std::uint64_t seed = 0xC0FFEE;
  std::string csv = "bench_explosion_per_sim.csv";
  std::vector<std::size_t> lanes = {4, 8, 16};
};

std::vector<std::size_t> parse_lanes(const std::string& s) {
  std::vector<std::size_t> out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    std::size_t comma = s.find(',', pos);
    if (comma == std::string::npos) comma = s.size();
    out.push_back(static_cast<std::size_t>(std::stoul(s.substr(pos, comma - pos))));
    pos = comma + 1;
  }
  return out;
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const bool has_val = i + 1 < argc;
    if (flag == "--sims" && has_val) {
      a.sims = static_cast<std::size_t>(std::stoul(argv[++i]));
    } else if (flag == "--seed" && has_val) {
      a.seed = static_cast<std::uint64_t>(std::stoull(argv[++i]));
    } else if (flag == "--csv" && has_val) {
      a.csv = argv[++i];
    } else if (flag == "--lanes" && has_val) {
      a.lanes = parse_lanes(argv[++i]);
    } else {
      std::cerr << "unknown or incomplete argument: " << flag << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  const Molecule molecule = make_example_molecule();

  UniformSphereSampler sampler({.radius = Real{4.0}, .min_separation = Real{0.25}}, args.seed);
  CoulombForce force;
  auto integrator = make_integrator(IntegratorKind::RK45);
  const RunConfig config;  // default pe_stop_fraction + energy redistribution

  // --- Generate the sample -------------------------------------------------
  std::vector<SimRecord> records;
  records.reserve(args.sims);
  for (std::size_t m = 0; m < args.sims; ++m) {
    State state;
    sampler.sample(molecule, state);

    SimRecord rec;
    rec.min_init_sep = min_pair_distance(state);
    rec.trace.reserve(160);
    const StepObserver observer = [&rec](std::size_t, Real t_start, Real dt) {
      rec.trace.emplace_back(t_start, dt);
    };
    const RunResult run = run_to_convergence(molecule, force, *integrator, state, config, observer);

    rec.steps = run.steps;
    rec.t_final = run.t_final;
    rec.e_initial = run.energy_initial;
    rec.pe_final = run.pe_final;
    records.push_back(std::move(rec));
  }

  // --- Per-sim CSV ---------------------------------------------------------
  {
    std::ofstream out(args.csv);
    out << "sim,steps,t_final,min_init_sep,e_initial,pe_final\n";
    for (std::size_t m = 0; m < records.size(); ++m) {
      const SimRecord& r = records[m];
      out << m << ',' << r.steps << ',' << r.t_final << ',' << r.min_init_sep << ',' << r.e_initial
          << ',' << r.pe_final << '\n';
    }
  }

  // --- Step-count distribution --------------------------------------------
  std::vector<double> steps_d, sep_d;
  steps_d.reserve(records.size());
  sep_d.reserve(records.size());
  for (const SimRecord& r : records) {
    steps_d.push_back(static_cast<double>(r.steps));
    sep_d.push_back(r.min_init_sep);
  }
  const Stats st = summarize(steps_d);
  const double sep_corr = pearson(sep_d, steps_d);

  std::cout << "molecule        : 8 atoms (H/C/N/O, singly ionized)\n"
            << "sampler         : " << sampler.name() << " r=4.0 min_sep=0.25\n"
            << "integrator      : " << integrator->name() << "\n"
            << "sims            : " << records.size() << "\n\n";

  std::cout << "accepted-step count per sim\n"
            << "  mean " << st.mean << "  sd " << st.sd << "  cv " << st.cv << "\n"
            << "  min " << st.min << "  p50 " << st.p50 << "  p90 " << st.p90 << "  p99 " << st.p99
            << "  max " << st.max << "\n"
            << "  corr(min_init_sep, steps) = " << sep_corr
            << "  (predictor for difficulty binning)\n\n";

  // --- Shared-dt lockstep efficiency by Monte-Carlo ------------------------
  std::cout << "shared-dt lockstep efficiency (Monte-Carlo over random batches)\n"
            << "  efficiency = speedup / K; refill/wavefront ceiling is ~1.00\n"
            << "  lanes recovered = (1 - eff) * K  <- what refill would buy\n";

  std::mt19937_64 pick(args.seed ^ 0x9E3779B97F4A7C15ULL);
  constexpr std::size_t kBatches = 4000;
  std::uniform_int_distribution<std::size_t> idist(0, records.size() - 1);

  for (std::size_t k : args.lanes) {
    if (k == 0 || k > records.size()) continue;
    std::vector<double> effs;
    effs.reserve(kBatches);
    for (std::size_t b = 0; b < kBatches; ++b) {
      // K distinct sims drawn at random.
      std::vector<const SimRecord*> batch;
      batch.reserve(k);
      std::vector<std::size_t> chosen;
      while (batch.size() < k) {
        const std::size_t picki = idist(pick);
        if (std::find(chosen.begin(), chosen.end(), picki) != chosen.end()) continue;
        chosen.push_back(picki);
        batch.push_back(&records[picki]);
      }
      std::size_t scalar_work = 0;
      for (const SimRecord* r : batch) scalar_work += r->steps;
      const std::size_t iters = shared_dt_iters(batch);
      const double speedup = static_cast<double>(scalar_work) / static_cast<double>(iters);
      effs.push_back(speedup / static_cast<double>(k));
    }
    const Stats es = summarize(effs);
    std::cout << "  K=" << k << "  eff mean " << es.mean << "  sd " << es.sd << "  p50 " << es.p50
              << "  p90 " << es.p90 << "   lanes recovered ~ "
              << (1.0 - es.mean) * static_cast<double>(k) << "\n";
  }

  std::cout << "\nper-sim data written to " << args.csv << "\n";
  return EXIT_SUCCESS;
}
