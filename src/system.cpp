#include "coulomb/system.hpp"

namespace coulomb {

void CoulombForce::accelerations(const Molecule& molecule, const State& state,
                                 std::vector<Vec3>& forces) const {
  const std::size_t n = molecule.size();
  forces.assign(n, Vec3{});

  // Naive O(N^2) all-pairs Coulomb interaction (the correctness baseline).
  // Force on i from j: F = k * q_i * q_j * r_ij / |r_ij|^3, with r_ij = x_i - x_j.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const Vec3 r = state.positions[i] - state.positions[j];
      const Real dist2 = dot(r, r);
      const Real inv_dist = Real{1} / std::sqrt(dist2);
      const Real inv_dist3 = inv_dist / dist2;
      const Real scale = k_ * molecule.atoms[i].charge * molecule.atoms[j].charge * inv_dist3;
      const Vec3 f = r * scale;
      forces[i] += f;
      forces[j] -= f;  // Newton's third law.
    }
  }

  // Convert force to acceleration in place: a_i = F_i / m_i.
  for (std::size_t i = 0; i < n; ++i) {
    forces[i] = forces[i] * (Real{1} / molecule.atoms[i].mass);
  }
}

Real CoulombForce::potential_energy(const Molecule& molecule, const State& state) const {
  const std::size_t n = molecule.size();
  Real u{0};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const Vec3 r = state.positions[i] - state.positions[j];
      u += k_ * molecule.atoms[i].charge * molecule.atoms[j].charge / norm(r);
    }
  }
  return u;
}

Real CoulombForce::kinetic_energy(const Molecule& molecule, const State& state) {
  Real t{0};
  for (std::size_t i = 0; i < molecule.size(); ++i) {
    t += Real{0.5} * molecule.atoms[i].mass * dot(state.velocities[i], state.velocities[i]);
  }
  return t;
}

}  // namespace coulomb
