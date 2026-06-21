#pragma once

#include <vector>

#include "coulomb/molecule.hpp"
#include "coulomb/types.hpp"

namespace coulomb {

/// Phase-space state of an N-body system: positions and velocities, one entry
/// per atom. Stored as an array-of-structs for now; this is the obvious place
/// to later evaluate a structure-of-arrays layout for SIMD/cache behavior.
struct State {
  std::vector<Vec3> positions;
  std::vector<Vec3> velocities;

  std::size_t size() const { return positions.size(); }
};

/// Coulomb interaction between point charges.
///
/// The current implementation is the naive O(N^2) all-pairs kernel. It is the
/// correctness baseline: simple, easy to verify, and the reference every
/// optimized variant (blocking, SIMD via Highway, ...) is measured against.
class CoulombForce {
 public:
  /// `coulomb_constant` rolls the unit system into one factor (set to 1 for
  /// reduced/atomic units). Charges live on the atoms in `molecule`.
  explicit CoulombForce(Real coulomb_constant = Real{1}) : k_(coulomb_constant) {}

  /// Accumulate forces on each atom into `forces` (resized to N, zeroed first).
  void accelerations(const Molecule& molecule, const State& state, std::vector<Vec3>& forces) const;

  /// Total potential energy of the configuration. Useful as a conserved-ish
  /// quantity for validating integrators.
  Real potential_energy(const Molecule& molecule, const State& state) const;

  /// Total kinetic energy of the configuration.
  static Real kinetic_energy(const Molecule& molecule, const State& state);

 private:
  Real k_;
};

}  // namespace coulomb
