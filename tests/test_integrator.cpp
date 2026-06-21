#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "coulomb/integrator.hpp"
#include "coulomb/system.hpp"

using namespace coulomb;
using Catch::Matchers::WithinAbs;

TEST_CASE("velocity Verlet approximately conserves total energy", "[integrator]") {
  Molecule mol;
  mol.atoms = {{"X", 1.0, 1.0}, {"X", 1.0, 1.0}};

  State state;
  state.positions = {{0, 0, 0}, {1, 0, 0}};
  state.velocities = {{0, 0, 0}, {0, 0, 0}};

  CoulombForce force(1.0);
  auto integ = make_integrator(IntegratorKind::VelocityVerlet);

  const Real e0 = force.potential_energy(mol, state) +
                  CoulombForce::kinetic_energy(mol, state);

  const Real dt = 1e-4;
  for (int i = 0; i < 2000; ++i) {
    integ->step(mol, force, state, dt);
  }

  const Real e1 = force.potential_energy(mol, state) +
                  CoulombForce::kinetic_energy(mol, state);

  // Symplectic integrators bound the energy error; it should stay small.
  REQUIRE_THAT(e1 - e0, WithinAbs(0.0, 1e-3));
}

TEST_CASE("RK45 is not yet implemented and reports so", "[integrator]") {
  REQUIRE_THROWS_AS(make_integrator(IntegratorKind::RK45), std::invalid_argument);
}
