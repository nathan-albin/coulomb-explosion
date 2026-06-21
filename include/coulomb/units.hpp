#pragma once

#include "coulomb/types.hpp"

namespace coulomb::units {

/// Atomic mass unit (dalton) expressed in electron masses, i.e. the ratio
/// m_u / m_e. CODATA 2018 "atomic mass unit-electron mass relationship".
/// The dynamics run in atomic units where the electron mass is 1, so masses
/// supplied in amu are converted through this factor at the input boundary.
inline constexpr Real kAmuToElectronMass = 1822.888486209;

/// Convert a mass in amu (daltons) to electron masses (simulation units).
constexpr Real amu_to_electron_mass(Real mass_amu) {
  return mass_amu * kAmuToElectronMass;
}

}  // namespace coulomb::units
