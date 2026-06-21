#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>

#include "coulomb/molecule.hpp"
#include "coulomb/sampler.hpp"
#include "coulomb/system.hpp"
#include "coulomb/types.hpp"

using namespace coulomb;

namespace {

// The 8-atom, mixed-mass example used to study the SIMD batch path: four masses
// (H, C, N, O), singly ionized. Geometry is supplied by the sampler.
Molecule mixed_eight() {
  Molecule m;
  m.atoms = {
      atom_from_amu("H", 1.008, 1.0),  atom_from_amu("C", 12.011, 1.0),
      atom_from_amu("N", 14.007, 1.0), atom_from_amu("O", 15.999, 1.0),
      atom_from_amu("H", 1.008, 1.0),  atom_from_amu("C", 12.011, 1.0),
      atom_from_amu("N", 14.007, 1.0), atom_from_amu("O", 15.999, 1.0),
  };
  return m;
}

// Smallest pairwise distance in a configuration; large sentinel for < 2 atoms.
Real min_pair_distance(const State& s) {
  Real best = 1e300;
  for (std::size_t i = 0; i < s.size(); ++i) {
    for (std::size_t j = i + 1; j < s.size(); ++j) {
      best = std::min(best, norm(s.positions[i] - s.positions[j]));
    }
  }
  return best;
}

}  // namespace

TEST_CASE("uniform-sphere sampler respects the radius, separation, and rest conditions",
          "[sampler]") {
  const UniformSphereSampler::Options opts{.radius = 4.0, .min_separation = 0.25};
  UniformSphereSampler sampler(opts, /*seed=*/12345);
  const Molecule mol = mixed_eight();

  // Many draws: every one must land in the ball, clear the separation filter,
  // and leave the atoms at rest. A small float slack absorbs round-off at the
  // boundary.
  for (int draw = 0; draw < 1000; ++draw) {
    State state;
    sampler.sample(mol, state);

    REQUIRE(state.positions.size() == mol.size());
    REQUIRE(state.velocities.size() == mol.size());

    for (std::size_t i = 0; i < state.size(); ++i) {
      REQUIRE(norm(state.positions[i]) <= opts.radius + 1e-9);
      REQUIRE(state.velocities[i].x == 0.0);
      REQUIRE(state.velocities[i].y == 0.0);
      REQUIRE(state.velocities[i].z == 0.0);
    }
    REQUIRE(min_pair_distance(state) >= opts.min_separation - 1e-12);
  }
}

TEST_CASE("a fixed seed reproduces the same stream of configurations", "[sampler]") {
  const UniformSphereSampler::Options opts{.radius = 4.0, .min_separation = 0.25};
  const Molecule mol = mixed_eight();

  UniformSphereSampler a(opts, /*seed=*/777);
  UniformSphereSampler b(opts, /*seed=*/777);

  for (int draw = 0; draw < 16; ++draw) {
    State sa;
    State sb;
    a.sample(mol, sa);
    b.sample(mol, sb);
    for (std::size_t i = 0; i < mol.size(); ++i) {
      REQUIRE(sa.positions[i].x == sb.positions[i].x);
      REQUIRE(sa.positions[i].y == sb.positions[i].y);
      REQUIRE(sa.positions[i].z == sb.positions[i].z);
    }
  }
}

TEST_CASE("successive draws are independent", "[sampler]") {
  UniformSphereSampler sampler({.radius = 4.0, .min_separation = 0.25}, /*seed=*/42);
  const Molecule mol = mixed_eight();

  State first;
  State second;
  sampler.sample(mol, first);
  sampler.sample(mol, second);

  // Different draws should not coincide atom-for-atom (probability ~0).
  bool any_different = false;
  for (std::size_t i = 0; i < mol.size(); ++i) {
    any_different = any_different || norm(first.positions[i] - second.positions[i]) > 1e-12;
  }
  REQUIRE(any_different);
}

TEST_CASE("an infeasible radius/separation combination is reported", "[sampler]") {
  // A separation larger than the ball diameter cannot fit two atoms.
  UniformSphereSampler sampler({.radius = 1.0, .min_separation = 5.0, .max_attempts_per_atom = 200},
                               /*seed=*/1);
  Molecule mol;
  mol.atoms = {{"X", 1.0, 1.0}, {"X", 1.0, 1.0}};

  State state;
  REQUIRE_THROWS_AS(sampler.sample(mol, state), std::runtime_error);
}

TEST_CASE("rejects non-physical options at construction", "[sampler]") {
  REQUIRE_THROWS_AS(UniformSphereSampler({.radius = 0.0}, 1), std::invalid_argument);
  REQUIRE_THROWS_AS(UniformSphereSampler({.min_separation = -1.0}, 1), std::invalid_argument);
}
