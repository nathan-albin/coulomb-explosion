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

  const Real e0 = force.potential_energy(mol, state) + CoulombForce::kinetic_energy(mol, state);

  const Real dt = 1e-4;
  for (int i = 0; i < 2000; ++i) {
    integ->step(mol, force, state, dt);
  }

  const Real e1 = force.potential_energy(mol, state) + CoulombForce::kinetic_energy(mol, state);

  // Symplectic integrators bound the energy error; it should stay small.
  REQUIRE_THAT(e1 - e0, WithinAbs(0.0, 1e-3));
}

TEST_CASE("Dormand-Prince RK45 conserves energy on a two-body explosion", "[integrator][rk45]") {
  Molecule mol;
  mol.atoms = {{"X", 1.0, 1.0}, {"X", 2.0, 1.0}};

  State state;
  state.positions = {{0, 0, 0}, {1, 0, 0}};
  state.velocities = {{0, 0, 0}, {0, 0, 0}};

  CoulombForce force(1.0);
  auto integ = make_integrator(IntegratorKind::RK45, IntegratorOptions{1e-10, 1e-16});

  const Real e0 = force.potential_energy(mol, state) + CoulombForce::kinetic_energy(mol, state);

  // Let the adaptive scheme grow its own step from the strong-force regime out
  // into the tail; energy should be conserved to roughly the tolerance.
  Real t = 0;
  while (t < 1e4) {
    t += integ->step(mol, force, state, 1e12);
  }

  const Real e1 = force.potential_energy(mol, state) + CoulombForce::kinetic_energy(mol, state);
  REQUIRE_THAT(e1 - e0, WithinAbs(0.0, 1e-7));
}

TEST_CASE("Dormand-Prince RK45 conserves momentum and center of mass", "[integrator][rk45]") {
  // For a symmetric head-on explosion of two equal masses started at rest, the
  // center of mass must stay fixed and the total momentum must stay zero. These
  // are exact symmetries the tableau and step controller must not break.
  Molecule mol;
  mol.atoms = {{"X", 1.0, 1.0}, {"X", 1.0, 1.0}};

  State state;
  state.positions = {{-0.5, 0, 0}, {0.5, 0, 0}};
  state.velocities = {{0, 0, 0}, {0, 0, 0}};

  CoulombForce force(1.0);
  auto integ = make_integrator(IntegratorKind::RK45);

  Real t = 0;
  while (t < 1e3) {
    t += integ->step(mol, force, state, 1e12);
  }

  // Center of mass (equal masses) stays at the origin; total momentum stays 0.
  const Vec3 com = (state.positions[0] + state.positions[1]) * 0.5;
  const Vec3 p = state.velocities[0] + state.velocities[1];
  REQUIRE_THAT(com.x, WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(p.x, WithinAbs(0.0, 1e-10));
}
