#pragma once

#include <memory>
#include <string_view>

#include "coulomb/molecule.hpp"
#include "coulomb/system.hpp"

namespace coulomb {

/// Strategy interface for time integrators.
///
/// Making the solver pluggable is deliberate: the project needs to compare an
/// adaptive RK45 (accuracy reference, ported from the Python version) against a
/// symplectic scheme (energy-conserving, cheaper per step). Alternatives live
/// as real, tested implementations behind this interface rather than as
/// commented-out code paths.
class Integrator {
 public:
  virtual ~Integrator() = default;

  /// Advance `state` in place by one step of size `dt`.
  /// Returns the step size actually taken (may differ from `dt` for adaptive
  /// schemes; fixed-step schemes return `dt`).
  virtual Real step(const Molecule& molecule, const CoulombForce& force,
                    State& state, Real dt) = 0;

  /// Human-readable name for reports and output metadata.
  virtual std::string_view name() const = 0;
};

/// Which integrator to construct. Extend as schemes are added.
enum class IntegratorKind {
  VelocityVerlet,  ///< Symplectic, fixed-step. Implemented.
  RK45,            ///< Adaptive Runge-Kutta-Fehlberg. To be ported from Python.
};

/// Factory: build an integrator by kind. Throws std::invalid_argument for
/// kinds that are not yet implemented.
std::unique_ptr<Integrator> make_integrator(IntegratorKind kind);

}  // namespace coulomb
