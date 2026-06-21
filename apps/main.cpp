// Thin CLI driver around the simulation engine.
//
// This is intentionally minimal: it samples one configuration of a fixed
// example molecule, drives it to its asymptotic state, and prints an energy
// summary, just to exercise the engine end to end. Configuration parsing, a
// batched many-sims driver, and Parquet output are follow-up work.

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "coulomb/driver.hpp"
#include "coulomb/integrator.hpp"
#include "coulomb/molecule.hpp"
#include "coulomb/sampler.hpp"
#include "coulomb/system.hpp"

namespace {

using namespace coulomb;

/// The 8-atom, mixed-mass example used to study the SIMD batch path: four masses
/// (H, C, N, O), singly ionized. Masses are supplied in amu; atom_from_amu
/// converts to the electron-mass simulation units.
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

}  // namespace

int main() {
  const Molecule molecule = make_example_molecule();

  // Draw a starting geometry: atoms uniform in a 4 a.u. sphere, no two closer
  // than 0.25 a.u., all at rest.
  UniformSphereSampler sampler({.radius = Real{4.0}, .min_separation = Real{0.25}},
                               /*seed=*/std::uint64_t{0xC0FFEE});
  State state;
  sampler.sample(molecule, state);

  CoulombForce force;
  auto integrator = make_integrator(IntegratorKind::RK45);

  // Drive the explosion to its asymptotic state: integrate adaptively until the
  // potential energy has decayed away, then redistribute the residual into
  // kinetic energy so the final KE matches the initial total energy exactly.
  RunConfig config;
  const RunResult run = run_to_convergence(molecule, force, *integrator, state, config);

  std::cout << "integrator : " << integrator->name() << '\n'
            << "sampler    : " << sampler.name() << '\n'
            << "atoms      : " << molecule.size() << '\n'
            << "steps      : " << run.steps << '\n'
            << "t_final    : " << run.t_final << '\n'
            << "E_initial  : " << run.energy_initial << '\n'
            << "PE_final   : " << run.pe_final << '\n'
            << "KE (pre)   : " << run.ke_before_redist << '\n'
            << "KE (post)  : " << run.ke_after_redist << '\n'
            << "redist s   : " << run.redistribution_scale << '\n';

  return EXIT_SUCCESS;
}
