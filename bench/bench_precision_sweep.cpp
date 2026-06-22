// Precision-robustness sweep: does the engine survive single precision, and how
// far do f32 asymptotic momenta drift from the f64 reference?
//
// The reconstruction model runs fp32 on GPUs and the experimental momentum
// resolution is ~1%, so the data generator may be able to run in float for the
// throughput win (2x SIMD lanes). But a naive switch is risky: the explosion
// starts from near-singular 1/r^2 forces, and the close-encounter tail both
// amplifies round-off and can break the adaptive controller outright. This
// harness measures, per geometry: whether the f32 run *fails* (NaN/Inf, step
// rejected to death, or hitting the step cap) and the *asymptotic momenta* it
// produces. Diffing the f32 output against an f64 run on the *same* geometries
// (python/analysis/plot_precision.py) gives the failure rate, the |dp|/|p|
// distribution vs the 1% floor, and any systematic bias — keyed on min_init_sep,
// the difficulty axis from 0002.
//
// This binary is precision-agnostic (it uses whatever Real the build selects).
// Build it twice — the `relwithdebinfo` preset (f64) and `relwithdebinfo-f32`
// (f32) — and run both on one shared geometry file:
//
//   # f64 reference: sample, dump the geometries, run, emit results
//   ./build/relwithdebinfo/bench/coulomb_precision_sweep --atoms 8 --sims 4000 \
//       --dump-geometries /tmp/geo_n8.txt --csv /tmp/prec_n8_f64.csv
//   # f32: load the *same* geometries, looser fp32-safe tolerances, emit results
//   ./build/relwithdebinfo-f32/bench/coulomb_precision_sweep --atoms 8 \
//       --geometries /tmp/geo_n8.txt --rtol 1e-4 --atol 1e-7 --pe-stop 1e-5 \
//       --csv /tmp/prec_n8_f32.csv

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "coulomb/driver.hpp"
#include "coulomb/integrator.hpp"
#include "coulomb/molecule.hpp"
#include "coulomb/sampler.hpp"
#include "coulomb/system.hpp"

using namespace coulomb;

namespace {

// N-atom molecule built by cycling H, C, N, O (the 0002 mix), each singly
// ionized, so the sweep spans the same chemistry at varying close-encounter
// density. N is the only knob.
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

Real min_pair_distance(const State& s) {
  Real best = std::numeric_limits<Real>::infinity();
  for (std::size_t i = 0; i < s.size(); ++i) {
    for (std::size_t j = i + 1; j < s.size(); ++j) {
      best = std::min(best, norm(s.positions[i] - s.positions[j]));
    }
  }
  return best;
}

// f32 can fail where f64 sails through; the analysis needs to bin these.
enum FailureCode : int {
  kOk = 0,
  kException = 1,    // integrator threw (e.g. step rejected too many times).
  kNotConverged = 2, // hit max_steps without meeting the PE-stop criterion.
  kNonFinite = 3,    // NaN/Inf in the final momenta.
};

// ---------------------------------------------------------------------------
// Geometry I/O — full-precision so the f64 and f32 runs share identical inputs
// (the f32 run reads these doubles and rounds them to float on load, which is
// the correct start of the f32 pipeline).
// ---------------------------------------------------------------------------

void dump_geometries(const std::string& path, std::size_t n,
                     const std::vector<State>& states) {
  std::ofstream out(path);
  out << "atoms " << n << " sims " << states.size() << '\n';
  out.precision(17);
  for (const State& s : states) {
    for (std::size_t i = 0; i < n; ++i) {
      out << s.positions[i].x << ' ' << s.positions[i].y << ' ' << s.positions[i].z;
      out << (i + 1 < n ? ' ' : '\n');
    }
  }
}

std::vector<State> load_geometries(const std::string& path, std::size_t n) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "cannot open geometry file: " << path << '\n';
    std::exit(EXIT_FAILURE);
  }
  std::string tag;
  std::size_t file_n = 0, file_m = 0;
  std::string sims_tag;
  in >> tag >> file_n >> sims_tag >> file_m;
  if (file_n != n) {
    std::cerr << "geometry file atom count " << file_n << " != --atoms " << n << '\n';
    std::exit(EXIT_FAILURE);
  }
  std::vector<State> states;
  states.reserve(file_m);
  for (std::size_t m = 0; m < file_m; ++m) {
    State s;
    s.positions.resize(n);
    s.velocities.assign(n, Vec3{});
    for (std::size_t i = 0; i < n; ++i) {
      double x, y, z;  // read as double, narrow to Real on assignment.
      in >> x >> y >> z;
      s.positions[i] = Vec3{static_cast<Real>(x), static_cast<Real>(y), static_cast<Real>(z)};
    }
    states.push_back(std::move(s));
  }
  return states;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
  std::size_t atoms = 8;
  std::size_t sims = 4000;
  std::uint64_t seed = 0xC0FFEE;
  Real rtol = static_cast<Real>(1e-8);
  Real atol = static_cast<Real>(1e-16);
  Real pe_stop = static_cast<Real>(1e-9);
  std::string geometries;        // load from here if set.
  std::string dump_geometries;   // dump sampled geometries here if set.
  std::string csv = "prec_sweep.csv";
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const bool has_val = i + 1 < argc;
    auto next = [&] { return std::string(argv[++i]); };
    if (flag == "--atoms" && has_val) {
      a.atoms = static_cast<std::size_t>(std::stoul(next()));
    } else if (flag == "--sims" && has_val) {
      a.sims = static_cast<std::size_t>(std::stoul(next()));
    } else if (flag == "--seed" && has_val) {
      a.seed = static_cast<std::uint64_t>(std::stoull(next()));
    } else if (flag == "--rtol" && has_val) {
      a.rtol = static_cast<Real>(std::stod(next()));
    } else if (flag == "--atol" && has_val) {
      a.atol = static_cast<Real>(std::stod(next()));
    } else if (flag == "--pe-stop" && has_val) {
      a.pe_stop = static_cast<Real>(std::stod(next()));
    } else if (flag == "--geometries" && has_val) {
      a.geometries = next();
    } else if (flag == "--dump-geometries" && has_val) {
      a.dump_geometries = next();
    } else if (flag == "--csv" && has_val) {
      a.csv = next();
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
  const Molecule molecule = make_molecule(args.atoms);

  // --- Geometries: load shared, or sample (and optionally dump for the twin) --
  std::vector<State> states;
  if (!args.geometries.empty()) {
    states = load_geometries(args.geometries, args.atoms);
  } else {
    UniformSphereSampler sampler({.radius = Real{4.0}, .min_separation = Real{0.25}}, args.seed);
    states.reserve(args.sims);
    for (std::size_t m = 0; m < args.sims; ++m) {
      State s;
      sampler.sample(molecule, s);
      states.push_back(std::move(s));
    }
    if (!args.dump_geometries.empty()) dump_geometries(args.dump_geometries, args.atoms, states);
  }

  CoulombForce force;
  const IntegratorOptions iopts{args.rtol, args.atol};
  RunConfig config;
  config.pe_stop_fraction = args.pe_stop;

  // Precision tag for the output, so the analysis can label the two runs.
  const char* precision = (sizeof(Real) == sizeof(float)) ? "f32" : "f64";

  std::ofstream out(args.csv);
  out << "# precision=" << precision << " atoms=" << args.atoms << " sims=" << states.size()
      << " rtol=" << static_cast<double>(args.rtol) << " atol=" << static_cast<double>(args.atol)
      << " pe_stop=" << static_cast<double>(args.pe_stop) << '\n';
  out << "idx,min_init_sep,steps,t_final,e_initial,pe_final,converged,failure";
  for (std::size_t i = 0; i < args.atoms; ++i) out << ",p" << i << "x,p" << i << "y,p" << i << "z";
  out << '\n';
  out.precision(17);

  std::size_t n_fail = 0;
  for (std::size_t m = 0; m < states.size(); ++m) {
    State state = states[m];  // fresh integrator owns its own FSAL state below.
    const Real min_sep = min_pair_distance(state);

    auto integrator = make_integrator(IntegratorKind::RK45, iopts);
    int failure = kOk;
    RunResult run;
    try {
      run = run_to_convergence(molecule, force, *integrator, state, config);
      if (!run.converged) failure = kNotConverged;
    } catch (const std::exception&) {
      failure = kException;
    }

    // Asymptotic momenta p_i = m_i * v_i from the (in-place updated) state.
    std::vector<double> mom(3 * args.atoms, 0.0);
    if (failure == kOk) {
      for (std::size_t i = 0; i < args.atoms; ++i) {
        const Vec3 p = state.velocities[i] * molecule.atoms[i].mass;
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
          failure = kNonFinite;
          break;
        }
        mom[3 * i + 0] = static_cast<double>(p.x);
        mom[3 * i + 1] = static_cast<double>(p.y);
        mom[3 * i + 2] = static_cast<double>(p.z);
      }
    }
    if (failure != kOk) ++n_fail;

    out << m << ',' << static_cast<double>(min_sep) << ',' << run.steps << ','
        << static_cast<double>(run.t_final) << ',' << static_cast<double>(run.energy_initial) << ','
        << static_cast<double>(run.pe_final) << ',' << (run.converged ? 1 : 0) << ',' << failure;
    for (double v : mom) out << ',' << v;
    out << '\n';
  }

  std::cout << "precision " << precision << "  atoms " << args.atoms << "  sims " << states.size()
            << "  failures " << n_fail << " ("
            << (100.0 * static_cast<double>(n_fail) / static_cast<double>(states.size())) << "%)\n"
            << "results written to " << args.csv << '\n';
  return EXIT_SUCCESS;
}
