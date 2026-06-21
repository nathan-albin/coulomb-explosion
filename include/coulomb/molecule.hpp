#pragma once

#include <string>
#include <utility>
#include <vector>

#include "coulomb/types.hpp"
#include "coulomb/units.hpp"

namespace coulomb {

/// A single atom in the molecular definition: its identity plus the physical
/// quantities the dynamics depend on.
struct Atom {
  std::string symbol;  ///< Element symbol, e.g. "H", "C". For reporting/IO.
  Real mass{0};        ///< Atomic mass (simulation units).
  Real charge{0};      ///< Net charge after ionization (simulation units).
};

/// Build an atom from human-facing units: mass in amu (daltons), converted to
/// the electron-mass simulation units the dynamics expect. This is the
/// boundary where external mass input enters the engine.
inline Atom atom_from_amu(std::string symbol, Real mass_amu, Real charge) {
  return Atom{std::move(symbol), units::amu_to_electron_mass(mass_amu), charge};
}

/// The molecular definition: which atoms are present and their properties.
/// Equilibrium geometry is supplied separately by a sampler so the same
/// molecule can be initialized from many starting configurations.
struct Molecule {
  std::vector<Atom> atoms;

  std::size_t size() const { return atoms.size(); }
  bool empty() const { return atoms.empty(); }
};

}  // namespace coulomb
