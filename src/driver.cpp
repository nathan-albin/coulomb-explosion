#include "coulomb/driver.hpp"

#include <cmath>

namespace coulomb {

RunResult run_to_convergence(const Molecule& molecule, const CoulombForce& force,
                             Integrator& integrator, State& state, const RunConfig& config,
                             const StepObserver& observer) {
  RunResult result;

  const Real pe0 = force.potential_energy(molecule, state);
  result.energy_initial = pe0 + CoulombForce::kinetic_energy(molecule, state);
  const Real pe_stop = config.pe_stop_fraction * pe0;

  // Advance until the potential energy has decayed below the threshold. The
  // check runs before the first step so an already-converged state is a no-op.
  Real pe = pe0;
  while (pe > pe_stop) {
    if (result.steps >= config.max_steps) {
      result.converged = false;
      break;
    }
    const Real dt = integrator.step(molecule, force, state, config.max_dt);
    if (observer) observer(result.steps, result.t_final, dt);
    result.t_final += dt;
    ++result.steps;
    pe = force.potential_energy(molecule, state);
    result.converged = true;
  }

  result.pe_final = pe;
  result.ke_before_redist = CoulombForce::kinetic_energy(molecule, state);

  // Convert the residual potential energy into kinetic energy by scaling every
  // velocity so the total kinetic energy matches the initial total energy. KE
  // scales as s^2, so s = sqrt(E_0 / KE).
  if (config.redistribute_energy && result.ke_before_redist > Real{0}) {
    const Real s = std::sqrt(result.energy_initial / result.ke_before_redist);
    for (auto& v : state.velocities) v = v * s;
    result.redistribution_scale = s;
  }
  result.ke_after_redist = CoulombForce::kinetic_energy(molecule, state);

  return result;
}

}  // namespace coulomb
