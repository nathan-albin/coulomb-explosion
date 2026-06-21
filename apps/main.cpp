// Thin CLI driver around the simulation engine.
//
// This is intentionally minimal: it runs a single hard-coded molecule for a
// fixed number of steps and prints an energy summary, just to exercise the
// engine end to end. Configuration parsing, samplers, and Parquet output are
// follow-up work once the Python reference lands.

#include <cstdlib>
#include <iostream>

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
  d.molecule.atoms = {
      {"H", Real{1.0}, Real{1.0}},
      {"H", Real{1.0}, Real{1.0}},
      {"H", Real{1.0}, Real{1.0}},
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
  auto integrator = make_integrator(IntegratorKind::VelocityVerlet);

  const Real dt = Real{1e-3};
  const int steps = 1000;

  const Real e0 = force.potential_energy(demo.molecule, demo.state) +
                  CoulombForce::kinetic_energy(demo.molecule, demo.state);

  for (int s = 0; s < steps; ++s) {
    integrator->step(demo.molecule, force, demo.state, dt);
  }

  const Real e1 = force.potential_energy(demo.molecule, demo.state) +
                  CoulombForce::kinetic_energy(demo.molecule, demo.state);

  std::cout << "integrator : " << integrator->name() << '\n'
            << "atoms      : " << demo.molecule.size() << '\n'
            << "steps      : " << steps << " (dt=" << dt << ")\n"
            << "E_initial  : " << e0 << '\n'
            << "E_final    : " << e1 << '\n'
            << "drift      : " << (e1 - e0) << '\n';

  return EXIT_SUCCESS;
}
