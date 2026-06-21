#include "coulomb/sampler.hpp"

#include <stdexcept>

namespace coulomb {

UniformSphereSampler::UniformSphereSampler(Options options, std::uint64_t seed)
    : opts_(options), rng_(seed) {
  if (opts_.radius <= Real{0}) {
    throw std::invalid_argument("UniformSphereSampler: radius must be positive");
  }
  if (opts_.min_separation < Real{0}) {
    throw std::invalid_argument("UniformSphereSampler: min_separation must be non-negative");
  }
}

Vec3 UniformSphereSampler::sample_in_ball() {
  std::uniform_real_distribution<Real> coord(-opts_.radius, opts_.radius);
  const Real r2 = opts_.radius * opts_.radius;
  for (;;) {
    const Vec3 p{coord(rng_), coord(rng_), coord(rng_)};
    if (dot(p, p) <= r2) {
      return p;
    }
  }
}

void UniformSphereSampler::sample(const Molecule& molecule, State& state) {
  const std::size_t n = molecule.size();
  state.positions.resize(n);
  state.velocities.assign(n, Vec3{});  // atoms start at rest

  const Real min2 = opts_.min_separation * opts_.min_separation;
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t attempts = 0;
    for (;;) {
      const Vec3 candidate = sample_in_ball();
      bool clears = true;
      for (std::size_t j = 0; j < i; ++j) {
        const Vec3 d = candidate - state.positions[j];
        if (dot(d, d) < min2) {
          clears = false;
          break;
        }
      }
      if (clears) {
        state.positions[i] = candidate;
        break;
      }
      if (++attempts >= opts_.max_attempts_per_atom) {
        throw std::runtime_error(
            "UniformSphereSampler: could not place an atom respecting min_separation; "
            "radius too small or min_separation too large for the atom count");
      }
    }
  }
}

}  // namespace coulomb
