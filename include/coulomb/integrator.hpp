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

  /// Advance `state` in place by one step.
  ///
  /// `dt` is an upper bound on the step: fixed-step schemes take exactly `dt`;
  /// adaptive schemes own their internally controlled step size and take the
  /// largest accepted step no greater than `dt` (pass a large value to let an
  /// adaptive scheme grow freely). Returns the step size actually taken.
  virtual Real step(const Molecule& molecule, const CoulombForce& force, State& state, Real dt) = 0;

  /// Human-readable name for reports and output metadata.
  virtual std::string_view name() const = 0;
};

/// Which integrator to construct. Extend as schemes are added.
enum class IntegratorKind {
  VelocityVerlet,  ///< Symplectic, fixed-step. Implemented.
  RK45,            ///< Adaptive Dormand-Prince DP5(4). Matches scipy's `RK45`
                   ///< (the accuracy reference ported from the Python oracle).
};

/// Tunables for adaptive integrators. Ignored by fixed-step schemes. Defaults
/// mirror the Python reference (python/reference/coulomb.py: RKRTOL/RKATOL) so
/// the C++ engine reproduces the oracle's step-size control.
struct IntegratorOptions {
  Real rtol{1e-8};   ///< Relative local-error tolerance per component.
  Real atol{1e-16};  ///< Absolute local-error tolerance per component.
};

/// Factory: build an integrator by kind, with adaptive tunables. Throws
/// std::invalid_argument for unknown kinds.
std::unique_ptr<Integrator> make_integrator(IntegratorKind kind, const IntegratorOptions& options);

/// Convenience overload using default IntegratorOptions.
std::unique_ptr<Integrator> make_integrator(IntegratorKind kind);

}  // namespace coulomb
