#pragma once

#include <string>
#include <vector>

#include "coulomb/types.hpp"

namespace coulomb {

/// A single atom in the molecular definition: its identity plus the physical
/// quantities the dynamics depend on.
struct Atom {
  std::string symbol;  ///< Element symbol, e.g. "H", "C". For reporting/IO.
  Real mass{0};        ///< Atomic mass (simulation units).
  Real charge{0};      ///< Net charge after ionization (simulation units).
};

/// The molecular definition: which atoms are present and their properties.
/// Equilibrium geometry is supplied separately by a sampler so the same
/// molecule can be initialized from many starting configurations.
struct Molecule {
  std::vector<Atom> atoms;

  std::size_t size() const { return atoms.size(); }
  bool empty() const { return atoms.empty(); }
};

}  // namespace coulomb
