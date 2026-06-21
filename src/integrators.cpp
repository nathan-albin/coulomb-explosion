#include <stdexcept>
#include <vector>

#include "coulomb/integrator.hpp"

namespace coulomb {
namespace {

/// Velocity Verlet: symplectic, time-reversible, fixed-step. Good baseline for
/// the "symplectic solver" goal and cheap (two force evaluations amortized to
/// one per step by reusing the previous acceleration).
class VelocityVerlet final : public Integrator {
 public:
  Real step(const Molecule& molecule, const CoulombForce& force, State& state,
            Real dt) override {
    const std::size_t n = state.size();

    if (!have_accel_) {
      force.accelerations(molecule, state, accel_);
      have_accel_ = true;
    }

    // x(t+dt) = x + v*dt + 0.5*a*dt^2
    for (std::size_t i = 0; i < n; ++i) {
      state.positions[i] += state.velocities[i] * dt + accel_[i] * (Real{0.5} * dt * dt);
    }

    // a(t+dt) from the new positions.
    force.accelerations(molecule, state, next_accel_);

    // v(t+dt) = v + 0.5*(a(t) + a(t+dt))*dt
    for (std::size_t i = 0; i < n; ++i) {
      state.velocities[i] += (accel_[i] + next_accel_[i]) * (Real{0.5} * dt);
    }

    accel_.swap(next_accel_);
    return dt;
  }

  std::string_view name() const override { return "velocity-verlet"; }

 private:
  bool have_accel_{false};
  std::vector<Vec3> accel_;
  std::vector<Vec3> next_accel_;
};

}  // namespace

std::unique_ptr<Integrator> make_integrator(IntegratorKind kind) {
  switch (kind) {
    case IntegratorKind::VelocityVerlet:
      return std::make_unique<VelocityVerlet>();
    case IntegratorKind::RK45:
      // TODO: port the adaptive RK45 scheme from the Python reference.
      throw std::invalid_argument("RK45 integrator not yet implemented");
  }
  throw std::invalid_argument("unknown integrator kind");
}

}  // namespace coulomb
