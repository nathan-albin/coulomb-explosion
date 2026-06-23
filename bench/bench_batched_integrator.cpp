// Correctness gate + efficiency teaser for the batched shared-dt lockstep RK45
// integrator (0006 Part A). The batched integrator advances K explosions in
// lockstep on the SIMD force kernel; this driver checks that its per-lane
// asymptotic momenta match the scalar run_to_convergence on the *same* geometries
// (the per-lane oracle), and previews the realized lockstep efficiency against
// 0002's step-count ceiling (0.63 at K=8).
//
// Built -march=native (it includes the static-dispatch Highway integrator) and
// linked against the generic engine for the scalar reference + sampler.
//
// Run: ./build/relwithdebinfo/bench/coulomb_batched_integrator

#include <hwy/highway.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "coulomb/batched_integrator.hpp"
#include "coulomb/driver.hpp"
#include "coulomb/integrator.hpp"
#include "coulomb/molecule.hpp"
#include "coulomb/sampler.hpp"
#include "coulomb/system.hpp"

namespace hn = hwy::HWY_NAMESPACE;
using namespace coulomb;

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

// Config-norm relative momentum error ||p_batched - p_scalar|| / ||p_scalar|| over
// one lane's whole N-fragment momentum vector (the 0004/0005 metric).
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

}  // namespace

int main(int argc, char** argv) {
  const std::size_t n = (argc > 1) ? static_cast<std::size_t>(std::atoi(argv[1])) : 10;
  const double rtol = (argc > 2) ? std::atof(argv[2]) : 1e-8;
  const double atol = (argc > 3) ? std::atof(argv[3]) : 1e-16;

  const Molecule mol = make_molecule(n);
  const std::size_t k = hn::Lanes(hn::ScalableTag<double>());

  // K shared geometries.
  UniformSphereSampler sampler({.radius = Real{4.0}, .min_separation = Real{0.25}}, 0xC0FFEE);
  std::vector<State> geoms(k);
  for (std::size_t lane = 0; lane < k; ++lane) sampler.sample(mol, geoms[lane]);

  RunConfig cfg;  // engine defaults: pe_stop 1e-9, redistribute, max_dt 1e12.

  // Batched run (T=double oracle comparison).
  BatchedRK45<double> bi(mol, Real{1.0}, k, rtol, atol);
  for (std::size_t lane = 0; lane < k; ++lane) bi.set_lane_geometry(lane, geoms[lane].positions);
  const auto res = bi.run(cfg);

  // Scalar reference per lane on identical geometries.
  CoulombForce force(Real{1.0});
  double max_rel = 0.0;
  std::size_t scalar_total_steps = 0;
  std::size_t n_conv = 0;
  for (std::size_t lane = 0; lane < k; ++lane) {
    State s = geoms[lane];
    auto integ = make_integrator(IntegratorKind::RK45, IntegratorOptions{Real(rtol), Real(atol)});
    const RunResult rr = run_to_convergence(mol, force, *integ, s, cfg);
    scalar_total_steps += rr.steps;

    std::vector<Vec3> ps(n);
    for (std::size_t i = 0; i < n; ++i) ps[i] = s.velocities[i] * mol.atoms[i].mass;
    max_rel = std::max(max_rel, config_rel(&res.momenta[lane * n * 3], ps, n));
    if (res.converged[lane]) ++n_conv;
  }

  const double eff = static_cast<double>(scalar_total_steps) /
                     (static_cast<double>(k) * static_cast<double>(res.batch_steps));
  std::printf("batched lockstep integrator  N=%zu  K=%zu  rtol=%.0e atol=%.0e\n", n, k, rtol, atol);
  std::printf("  batched vs scalar  max |dP|/|P| = %.3e   (lanes converged %zu/%zu)\n", max_rel,
              n_conv, k);
  std::printf("  batch SIMD-iters = %zu   scalar total steps = %zu   lockstep efficiency = %.3f\n",
              res.batch_steps, scalar_total_steps, eff);
  std::printf("    (0002 estimated shared-dt efficiency at K=8 was ~0.63)\n");

  // Batched (lockstep, min-step) and scalar (independent steps) are two valid
  // adaptive solutions at the same tolerance, so they agree to a small multiple
  // of rtol, not to machine epsilon. Gate on a tolerance-scaled bound (with a
  // floor) — generous enough for correct tolerance-level disagreement, tight
  // enough to catch a physics/indexing bug, which would miss by orders.
  const double gate_tol = std::max(50.0 * rtol, 1e-6);
  if (!(max_rel < gate_tol) || n_conv != k) {
    std::printf("  GATE FAILED (tol %.1e, all lanes must converge)\n", gate_tol);
    return 1;
  }
  std::printf("  GATE PASSED\n");
  return 0;
}
