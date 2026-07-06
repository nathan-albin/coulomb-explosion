// CH4 (methane) dataset generator — docs/benchmarks/0008.
//
// Drives the batched f32 lockstep RK45 integrator (BatchedRK45<float>,
// include/coulomb/batched_integrator.hpp) over millions of independent Coulomb
// explosions of a fixed CH4 molecule (C + 4xH, singly ionized — the charge
// convention every ADR/benchmark in this repo uses) sampled from
// UniformSphereSampler, at the ADR 0004 f32 production tolerance point
// (rtol 1e-4 / atol 1e-7 / pe_stop 1e-5).
//
// Output is a raw binary of fixed-width records (not Parquet — see
// examples/to_parquet.py for that conversion step) plus a JSON sidecar
// describing the molecule and schema. Each record:
//   n*3 float32   initial positions (atom order: C, H, H, H, H; x,y,z per atom)
//   n*3 float32   final momenta (post energy-redistribution, m*v)
//   uint32        accepted SIMD-iterations (batch steps) at convergence
//   uint8         converged (1) or hit max_steps (0)
// Fields are written individually (no padded C struct) so the byte layout is
// exactly what it looks like — examples/to_parquet.py's numpy dtype must match.
//
// Run: ./build/relwithdebinfo/examples/coulomb_generate_dataset --out PATH
//        [--sims N] [--seed S] [--radius R] [--min-separation D]
//        [--rtol T] [--atol A] [--pe-stop P] [--progress-every B]
//
// --out is required and has no built-in default — pick your own output
// location (e.g. /path/to/scratch/coulomb_examples/ch4/dataset.bin; a fast
// local or scratch filesystem is recommended for a multi-GB output file, but
// nothing here assumes any particular mount).

#include <hwy/highway.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "coulomb/batched_integrator.hpp"
#include "coulomb/driver.hpp"
#include "coulomb/molecule.hpp"
#include "coulomb/sampler.hpp"
#include "coulomb/system.hpp"
#include "coulomb/units.hpp"

namespace hn = hwy::HWY_NAMESPACE;
using namespace coulomb;
using Clock = std::chrono::steady_clock;

namespace {

// CH4, singly ionized: C:+1, H:+1 x4. Fixed atom order — recorded in the
// sidecar JSON so downstream readers know the column layout.
Molecule make_methane() {
  Molecule m;
  m.atoms = {
      atom_from_amu("C", Real{12.011}, Real{1.0}),
      atom_from_amu("H", Real{1.008}, Real{1.0}),
      atom_from_amu("H", Real{1.008}, Real{1.0}),
      atom_from_amu("H", Real{1.008}, Real{1.0}),
      atom_from_amu("H", Real{1.008}, Real{1.0}),
  };
  return m;
}

struct Args {
  std::size_t sims = 10'000'000;
  std::uint64_t seed = 49188291;  // reused from the old python/reference/config.yaml.
  double radius = 4.0;
  double min_separation = 0.25;
  double rtol = 1e-4;
  double atol = 1e-7;
  double pe_stop = 1e-5;
  std::string out;
  std::size_t progress_every = 2000;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string f = argv[i];
    const bool has = i + 1 < argc;
    auto next = [&] { return std::string(argv[++i]); };
    if (f == "--sims" && has)
      a.sims = static_cast<std::size_t>(std::stoull(next()));
    else if (f == "--seed" && has)
      a.seed = static_cast<std::uint64_t>(std::stoull(next()));
    else if (f == "--radius" && has)
      a.radius = std::stod(next());
    else if (f == "--min-separation" && has)
      a.min_separation = std::stod(next());
    else if (f == "--rtol" && has)
      a.rtol = std::stod(next());
    else if (f == "--atol" && has)
      a.atol = std::stod(next());
    else if (f == "--pe-stop" && has)
      a.pe_stop = std::stod(next());
    else if (f == "--out" && has)
      a.out = next();
    else if (f == "--progress-every" && has)
      a.progress_every = static_cast<std::size_t>(std::stoull(next()));
    else {
      std::fprintf(stderr, "unknown or incomplete argument: %s\n", f.c_str());
      std::exit(EXIT_FAILURE);
    }
  }
  if (a.out.empty()) {
    std::fprintf(stderr,
                 "--out is required (no default — pick your own output location, e.g.\n"
                 "  --out /path/to/scratch/coulomb_examples/ch4/dataset.bin)\n");
    std::exit(EXIT_FAILURE);
  }
  return a;
}

void write_metadata(const Args& args, const Molecule& mol, std::size_t lanes,
                    double wall_seconds, double sims_per_sec, std::size_t n_failed) {
  const std::string path = args.out + ".meta.json";
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open metadata file: " + path);
  out << "{\n"
      << "  \"symbols\": [";
  for (std::size_t i = 0; i < mol.size(); ++i) out << "\"" << mol.atoms[i].symbol << "\""
                                                    << (i + 1 < mol.size() ? ", " : "");
  out << "],\n"
      << "  \"masses_amu\": [";
  for (std::size_t i = 0; i < mol.size(); ++i)
    out << (static_cast<double>(mol.atoms[i].mass) / units::kAmuToElectronMass)
        << (i + 1 < mol.size() ? ", " : "");
  out << "],\n"
      << "  \"charges\": [";
  for (std::size_t i = 0; i < mol.size(); ++i)
    out << static_cast<double>(mol.atoms[i].charge) << (i + 1 < mol.size() ? ", " : "");
  out << "],\n"
      << "  \"n_atoms\": " << mol.size() << ",\n"
      << "  \"n_sims\": " << args.sims << ",\n"
      << "  \"seed\": " << args.seed << ",\n"
      << "  \"radius\": " << args.radius << ",\n"
      << "  \"min_separation\": " << args.min_separation << ",\n"
      << "  \"rtol\": " << args.rtol << ",\n"
      << "  \"atol\": " << args.atol << ",\n"
      << "  \"pe_stop\": " << args.pe_stop << ",\n"
      << "  \"lanes\": " << lanes << ",\n"
      << "  \"n_failed_convergence\": " << n_failed << ",\n"
      << "  \"wall_seconds\": " << wall_seconds << ",\n"
      << "  \"sims_per_sec\": " << sims_per_sec << ",\n"
      << "  \"record_layout\": ["
      << "\"init_pos: n*3 float32 (x,y,z per atom, atom order as symbols)\", "
      << "\"final_momentum: n*3 float32 (m*v per atom, post energy-redistribution)\", "
      << "\"steps: uint32\", "
      << "\"converged: uint8\"]\n"
      << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  const Molecule mol = make_methane();
  const std::size_t n = mol.size();
  const std::size_t lanes = hn::Lanes(hn::ScalableTag<float>());

  std::printf("CH4 dataset generator: n_sims=%zu lanes=%zu rtol=%.0e atol=%.0e pe_stop=%.0e\n",
              args.sims, lanes, args.rtol, args.atol, args.pe_stop);

  UniformSphereSampler sampler(
      {.radius = static_cast<Real>(args.radius),
       .min_separation = static_cast<Real>(args.min_separation)},
      args.seed);

  BatchedRK45<float> bi(mol, Real{1.0}, lanes, static_cast<float>(args.rtol),
                       static_cast<float>(args.atol));
  RunConfig cfg;
  cfg.pe_stop_fraction = static_cast<Real>(args.pe_stop);

  std::ofstream out(args.out, std::ios::binary);
  if (!out) {
    std::fprintf(stderr, "cannot open output file: %s\n", args.out.c_str());
    return EXIT_FAILURE;
  }

  const std::size_t n_batches = (args.sims + lanes - 1) / lanes;
  std::vector<State> lane_geoms(lanes);
  std::vector<float> pos_buf(n * 3), mom_buf(n * 3);
  std::size_t sims_written = 0;
  std::size_t n_failed = 0;

  const auto t_start = Clock::now();
  for (std::size_t b = 0; b < n_batches; ++b) {
    for (std::size_t lane = 0; lane < lanes; ++lane) {
      sampler.sample(mol, lane_geoms[lane]);
      bi.set_lane_geometry(lane, lane_geoms[lane].positions);
    }
    const auto res = bi.run(cfg);

    for (std::size_t lane = 0; lane < lanes && sims_written < args.sims; ++lane, ++sims_written) {
      for (std::size_t i = 0; i < n; ++i) {
        pos_buf[i * 3 + 0] = static_cast<float>(lane_geoms[lane].positions[i].x);
        pos_buf[i * 3 + 1] = static_cast<float>(lane_geoms[lane].positions[i].y);
        pos_buf[i * 3 + 2] = static_cast<float>(lane_geoms[lane].positions[i].z);
        mom_buf[i * 3 + 0] = static_cast<float>(res.momenta[(lane * n + i) * 3 + 0]);
        mom_buf[i * 3 + 1] = static_cast<float>(res.momenta[(lane * n + i) * 3 + 1]);
        mom_buf[i * 3 + 2] = static_cast<float>(res.momenta[(lane * n + i) * 3 + 2]);
      }
      const std::uint32_t steps = static_cast<std::uint32_t>(res.steps[lane]);
      const std::uint8_t converged = static_cast<std::uint8_t>(res.converged[lane] ? 1 : 0);
      if (!res.converged[lane]) ++n_failed;

      out.write(reinterpret_cast<const char*>(pos_buf.data()),
               static_cast<std::streamsize>(pos_buf.size() * sizeof(float)));
      out.write(reinterpret_cast<const char*>(mom_buf.data()),
               static_cast<std::streamsize>(mom_buf.size() * sizeof(float)));
      out.write(reinterpret_cast<const char*>(&steps), sizeof(steps));
      out.write(reinterpret_cast<const char*>(&converged), sizeof(converged));
    }

    if ((b + 1) % args.progress_every == 0 || b + 1 == n_batches) {
      const double elapsed = std::chrono::duration<double>(Clock::now() - t_start).count();
      const double rate = static_cast<double>(sims_written) / elapsed;
      const double remaining = (static_cast<double>(args.sims - sims_written)) / rate;
      std::fprintf(stderr, "\r%zu/%zu sims (%.0f sims/sec, ~%.0fs remaining)   ", sims_written,
                  args.sims, rate, remaining);
      std::fflush(stderr);
    }
  }
  out.close();
  std::fprintf(stderr, "\n");

  const double wall_seconds = std::chrono::duration<double>(Clock::now() - t_start).count();
  const double sims_per_sec = static_cast<double>(sims_written) / wall_seconds;
  std::printf("done: %zu sims in %.2fs (%.0f sims/sec), %zu failed to converge\n", sims_written,
              wall_seconds, sims_per_sec, n_failed);

  write_metadata(args, mol, lanes, wall_seconds, sims_per_sec, n_failed);
  std::printf("wrote %s and %s.meta.json\n", args.out.c_str(), args.out.c_str());

  return EXIT_SUCCESS;
}
