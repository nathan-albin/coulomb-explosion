#pragma once

#include <cstddef>

#include "coulomb/integrator.hpp"
#include "coulomb/molecule.hpp"
#include "coulomb/system.hpp"

namespace coulomb {

/// How to drive a Coulomb explosion to its asymptotic (t -> infinity) state.
///
/// A Coulomb explosion never truly "finishes": the atoms separate forever while
/// the residual potential energy decays like 1/t. The Python oracle fakes the
/// limit by integrating to t = 1e10 and asserting the forces have vanished.
/// Owning the loop in C++ lets us instead stop on a physical convergence
/// criterion (cheap to check every step) and then account for the small
/// remaining potential energy exactly -- see `redistribute_energy`.
struct RunConfig {
  /// Terminate once the potential energy falls to this fraction of its initial
  /// value (PE <= pe_stop_fraction * PE_0). This is the main accuracy/cost
  /// knob: smaller means later termination, closer to the true asymptote, more
  /// steps. The energy-redistribution step below is what makes a relatively
  /// loose fraction still yield accurate asymptotic momenta.
  Real pe_stop_fraction{1e-9};

  /// After termination, rescale every velocity by s = sqrt(E_0 / KE) so the
  /// final kinetic energy exactly equals the initial total energy, converting
  /// the residual potential energy into kinetic energy. Uniform scaling is
  /// momentum-preserving (total momentum is unchanged, hence still ~0 for a
  /// system started at rest) and the correction vanishes as the configuration
  /// approaches the true asymptote.
  bool redistribute_energy{true};

  /// Upper bound on a single integrator step, passed through as the step cap.
  /// Bounds how far an adaptive scheme may leap as the forces vanish.
  Real max_dt{1e12};

  /// Safety limit so a pathological case cannot spin forever.
  std::size_t max_steps{1'000'000};
};

/// Outcome of a convergence-driven run. Energies are reported so callers can
/// study the accuracy/termination trade-off (e.g. how far KE drifted before the
/// redistribution correction, and how much energy that correction moved).
struct RunResult {
  std::size_t steps{0};          ///< Accepted integrator steps taken.
  Real t_final{0};               ///< Total simulated time at termination.
  Real energy_initial{0};        ///< E_0 = KE_0 + PE_0.
  Real pe_final{0};              ///< Potential energy at termination.
  Real ke_before_redist{0};      ///< Kinetic energy at termination (pre-correction).
  Real ke_after_redist{0};       ///< Kinetic energy after redistribution (== E_0 if applied).
  Real redistribution_scale{1};  ///< Velocity scale factor s applied (1 if disabled).
  bool converged{false};         ///< False if max_steps was hit first.
};

/// Integrate `state` to the asymptotic explosion state in place, then optionally
/// redistribute residual potential energy into kinetic energy. The integrator's
/// adaptive tolerances are configured at its construction (IntegratorOptions).
RunResult run_to_convergence(const Molecule& molecule, const CoulombForce& force,
                             Integrator& integrator, State& state, const RunConfig& config);

}  // namespace coulomb
