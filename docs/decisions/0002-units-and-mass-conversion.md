# 0002 — Units and mass conversion

- **Status:** accepted
- **Date:** 2026-06-21

## Context

The dynamics run in atomic units, where the electron mass is 1 and the Coulomb
constant is 1 (see `CoulombForce`, which rolls the unit system into a single
`k` factor). External input, however, is naturally expressed in amu (daltons):
that is how atomic masses appear in periodic tables, chemistry data, and the
project's configuration. Something has to convert amu to electron masses, and
the conversion factor `m_u / m_e` must be applied consistently across both the
C++ engine and the Python reference oracle that validates it.

The factor is a physical constant (CODATA 2018 "atomic mass unit-electron mass
relationship", `1822.888486209`), not a tunable — there is no decision in its
*value*. The decision is in *where* the conversion happens and how the two
implementations are kept in agreement.

## Decision

- **The engine is atomic-units throughout.** `Atom::mass` is always in electron
  masses; the force kernel, integrators, and energy functions never see amu and
  never convert. This keeps the hot paths free of unit logic and avoids any risk
  of double conversion.
- **Conversion happens once, at the input boundary.** `atom_from_amu()`
  (`include/coulomb/molecule.hpp`) is the sole sanctioned entry point for masses
  expressed in amu. Code that ingests external input — config parsing, samplers,
  the CLI demo — constructs atoms through it rather than filling `Atom{}` by
  hand.
- **One constant, defined once per language.** The factor lives in
  `units::kAmuToElectronMass` (`include/coulomb/units.hpp`) for C++ and in
  `convert_masses` (`python/reference/coulomb.py`) for Python. Both carry the
  full-precision CODATA value and a comment pointing at the other.

## Consequences

- The constant is duplicated across the C++/Python boundary and **must be kept
  bit-identical**. If it diverges, the Tier-2 reference fixtures (asymptotic
  momenta, final kinetic energy) drift silently, because those quantities are
  mass-dependent. When the value changes, update both sources and regenerate the
  fixtures with `tests/reference/gen_reference_cases.py`. (Tier-1 force and
  potential-energy checks are mass-invariant and would not catch a mismatch.)
- Hand-constructing `Atom{symbol, mass, charge}` is reserved for masses already
  in electron masses (e.g. fixtures the oracle pre-converted). New input paths
  should prefer `atom_from_amu`.
- A future config/IO layer inherits this rule for free: it converts at the
  boundary and hands atomic-units atoms to the engine.
