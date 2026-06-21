// Thin CLI driver around the simulation engine.
//
// This is intentionally minimal: it runs a single hard-coded molecule for a
// fixed number of steps and prints an energy summary, just to exercise the
// engine end to end. Configuration parsing, samplers, and Parquet output are
// follow-up work once the Python reference lands.

#include <cstdlib>
#include <iostream>

#include "coulomb/driver.hpp"
#include "coulomb/integrator.hpp"
#include "coulomb/molecule.hpp"
#include "coulomb/system.hpp"

namespace {

using namespace coulomb;

/// A trivially small, fully ionized test system: three unit charges placed in a
/// triangle. Stands in for a real sampler until one exists.
struct Demo {
  Molecule molecule;
  State state;
};

Demo make_demo() {
  Demo d;
  // Masses supplied in amu; atom_from_amu converts to electron masses.
  d.molecule.atoms = {
      atom_from_amu("H", Real{1.0}, Real{1.0}),
      atom_from_amu("H", Real{1.0}, Real{1.0}),
      atom_from_amu("H", Real{1.0}, Real{1.0}),
  };
  d.state.positions = {
      {Real{0.0}, Real{0.0}, Real{0.0}},
      {Real{1.0}, Real{0.0}, Real{0.0}},
      {Real{0.5}, Real{0.866}, Real{0.0}},
  };
  d.state.velocities.assign(d.molecule.size(), Vec3{});
  return d;
}

}  // namespace

int main() {
  Demo demo = make_demo();
  CoulombForce force;
  auto integrator = make_integrator(IntegratorKind::RK45);

  // Drive the explosion to its asymptotic state: integrate adaptively until the
  // potential energy has decayed away, then redistribute the residual into
  // kinetic energy so the final KE matches the initial total energy exactly.
  RunConfig config;
  const RunResult run = run_to_convergence(demo.molecule, force, *integrator, demo.state, config);

  std::cout << "integrator : " << integrator->name() << '\n'
            << "atoms      : " << demo.molecule.size() << '\n'
            << "steps      : " << run.steps << '\n'
            << "t_final    : " << run.t_final << '\n'
            << "E_initial  : " << run.energy_initial << '\n'
            << "PE_final   : " << run.pe_final << '\n'
            << "KE (pre)   : " << run.ke_before_redist << '\n'
            << "KE (post)  : " << run.ke_after_redist << '\n'
            << "redist s   : " << run.redistribution_scale << '\n';

  return EXIT_SUCCESS;
}
