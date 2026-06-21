#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "coulomb/system.hpp"

using namespace coulomb;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Two unit charges a unit distance apart, k = 1.
State two_charges() {
  State s;
  s.positions = {{0, 0, 0}, {1, 0, 0}};
  s.velocities = {{0, 0, 0}, {0, 0, 0}};
  return s;
}

Molecule two_unit_charges() {
  Molecule m;
  m.atoms = {{"X", 1.0, 1.0}, {"X", 1.0, 1.0}};
  return m;
}

}  // namespace

TEST_CASE("potential energy of two unit charges", "[system]") {
  const auto mol = two_unit_charges();
  const auto state = two_charges();
  CoulombForce force(1.0);
  // U = k q1 q2 / r = 1.
  REQUIRE_THAT(force.potential_energy(mol, state), WithinRel(1.0, 1e-12));
}

TEST_CASE("forces are equal and opposite along the separation axis", "[system]") {
  const auto mol = two_unit_charges();
  const auto state = two_charges();
  CoulombForce force(1.0);

  std::vector<Vec3> acc;
  force.accelerations(mol, state, acc);

  REQUIRE(acc.size() == 2);
  // Repulsion pushes them apart along x; magnitude k q1 q2 / r^2 = 1.
  REQUIRE_THAT(acc[0].x, WithinRel(-1.0, 1e-12));
  REQUIRE_THAT(acc[1].x, WithinRel(1.0, 1e-12));
  REQUIRE_THAT(acc[0].y, WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(acc[0].z, WithinAbs(0.0, 1e-12));
}
