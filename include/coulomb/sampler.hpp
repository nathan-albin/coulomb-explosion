#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string_view>

#include "coulomb/molecule.hpp"
#include "coulomb/system.hpp"
#include "coulomb/types.hpp"

namespace coulomb {

/// Strategy interface for initial-geometry samplers.
///
/// A sampler turns a molecular definition (atom identities, masses, charges)
/// into a starting configuration: positions drawn from some spatial
/// distribution, with every atom at rest -- the explosion begins from rest
/// after instantaneous ionization. The distribution is pluggable because the
/// project needs to study how the sampling choice shapes the asymptotic-momentum
/// dataset, so alternatives live as real implementations behind this interface
/// rather than as commented-out code paths.
class Sampler {
 public:
  virtual ~Sampler() = default;

  /// Draw one configuration for `molecule` into `state`: positions resized to N
  /// and filled, velocities resized to N and zeroed. Advances the sampler's
  /// internal RNG, so successive calls yield independent configurations.
  virtual void sample(const Molecule& molecule, State& state) = 0;

  /// Human-readable name for reports and output metadata.
  virtual std::string_view name() const = 0;
};

/// Positions drawn uniformly from a ball of the given radius, rejecting any pair
/// of atoms closer than `min_separation`.
///
/// The minimum-separation filter is applied by sequential placement: each atom
/// is redrawn until it clears every already-placed atom. At the low packing
/// fractions this problem runs at (a handful of atoms in a comparatively large
/// sphere) the per-atom acceptance rate stays high, so this is far cheaper than
/// rejecting and redrawing the whole configuration. If an atom cannot be placed
/// within `max_attempts_per_atom` tries -- the signature of an infeasible
/// radius/separation combination -- `sample` throws std::runtime_error.
class UniformSphereSampler : public Sampler {
 public:
  struct Options {
    Real radius{4};                            ///< Ball radius (atomic units).
    Real min_separation{0.25};                 ///< Reject pairs closer than this (atomic units).
    std::size_t max_attempts_per_atom{10000};  ///< Placement-retry cap before giving up.
  };

  /// `seed` fixes the RNG stream so a run is reproducible; reseed to replay or
  /// to generate an independent stream.
  UniformSphereSampler(Options options, std::uint64_t seed);

  void sample(const Molecule& molecule, State& state) override;
  std::string_view name() const override { return "uniform-sphere"; }

  void reseed(std::uint64_t seed) { rng_.seed(seed); }

  const Options& options() const { return opts_; }

 private:
  /// One position drawn uniformly from the ball, by rejection in the bounding
  /// cube (acceptance pi/6 ~ 52%).
  Vec3 sample_in_ball();

  Options opts_;
  std::mt19937_64 rng_;
};

}  // namespace coulomb
