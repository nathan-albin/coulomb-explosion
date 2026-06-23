#pragma once

// Batched SIMD-over-lanes Coulomb kernels: one independent simulation of a fixed
// N-atom molecule per SIMD lane. Shared by the force-kernel benchmark
// (bench/bench_force_batch.cpp) and the batched lockstep integrator (0006) so the
// subtle kernel has exactly one definition.
//
// Layout (SoA over lanes): for each coordinate, p[i*k .. i*k+k) holds atom i's
// value across the k sims. Charges/masses are per-atom scalars broadcast across
// lanes — the production workload is many geometries of one fixed molecule.
//   pair_const : k*q_i*q_j per (i,j), length n*n, narrowed to T.
//   inv_mass   : 1/m_i,            length n.
//   px/py/pz, ax/ay/az : length n*k, vector-aligned (hwy::AllocateAligned).
//
// This is Highway code using STATIC dispatch (hwy::HWY_NAMESPACE): every TU that
// includes it MUST be compiled for the host ISA (-march=native), and it must NOT
// be linked into the generic engine library, which stays SSE2-baseline so 0001's
// scalar floor remains the conservative reference (see docs/decisions/0001).

#include <hwy/highway.h>

#include <cstddef>

namespace coulomb {
namespace batched {
namespace hn = hwy::HWY_NAMESPACE;

// a_i = (1/m_i) * sum_j k q_i q_j (x_i - x_j) / |r_ij|^3, for all k lanes at once.
// Mirrors CoulombForce::accelerations (src/system.cpp) lane-for-lane: 1/sqrt, not
// rsqrt, so the numerics match the scalar baseline rather than trading accuracy.
template <class T>
void accelerations(std::size_t n, std::size_t k, const T* px, const T* py, const T* pz,
                   const T* pair_const, const T* inv_mass, T* ax, T* ay, T* az) {
  const hn::ScalableTag<T> d;
  const auto one = hn::Set(d, T(1));

  for (std::size_t i = 0; i < n; ++i) {
    hn::Store(hn::Zero(d), d, &ax[i * k]);
    hn::Store(hn::Zero(d), d, &ay[i * k]);
    hn::Store(hn::Zero(d), d, &az[i * k]);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto pxi = hn::Load(d, &px[i * k]);
    const auto pyi = hn::Load(d, &py[i * k]);
    const auto pzi = hn::Load(d, &pz[i * k]);
    auto axi = hn::Load(d, &ax[i * k]);
    auto ayi = hn::Load(d, &ay[i * k]);
    auto azi = hn::Load(d, &az[i * k]);

    for (std::size_t j = i + 1; j < n; ++j) {
      const auto rx = hn::Sub(pxi, hn::Load(d, &px[j * k]));
      const auto ry = hn::Sub(pyi, hn::Load(d, &py[j * k]));
      const auto rz = hn::Sub(pzi, hn::Load(d, &pz[j * k]));

      const auto dist2 = hn::MulAdd(rx, rx, hn::MulAdd(ry, ry, hn::Mul(rz, rz)));
      const auto inv_dist = hn::Div(one, hn::Sqrt(dist2));
      const auto inv_dist3 = hn::Div(inv_dist, dist2);
      const auto scale = hn::Mul(hn::Set(d, pair_const[i * n + j]), inv_dist3);

      const auto fx = hn::Mul(rx, scale);
      const auto fy = hn::Mul(ry, scale);
      const auto fz = hn::Mul(rz, scale);

      axi = hn::Add(axi, fx);
      ayi = hn::Add(ayi, fy);
      azi = hn::Add(azi, fz);
      hn::Store(hn::Sub(hn::Load(d, &ax[j * k]), fx), d, &ax[j * k]);
      hn::Store(hn::Sub(hn::Load(d, &ay[j * k]), fy), d, &ay[j * k]);
      hn::Store(hn::Sub(hn::Load(d, &az[j * k]), fz), d, &az[j * k]);
    }

    hn::Store(axi, d, &ax[i * k]);
    hn::Store(ayi, d, &ay[i * k]);
    hn::Store(azi, d, &az[i * k]);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto m = hn::Set(d, inv_mass[i]);
    hn::Store(hn::Mul(hn::Load(d, &ax[i * k]), m), d, &ax[i * k]);
    hn::Store(hn::Mul(hn::Load(d, &ay[i * k]), m), d, &ay[i * k]);
    hn::Store(hn::Mul(hn::Load(d, &az[i * k]), m), d, &az[i * k]);
  }
}

// Per-lane potential energy U = sum_{i<j} k q_i q_j / |r_ij|, written to pe[0..k).
// Mirrors CoulombForce::potential_energy (src/system.cpp). pe must hold >= k
// elements; written unaligned so callers need not over-align a small buffer.
template <class T>
void potential_energy(std::size_t n, std::size_t k, const T* px, const T* py, const T* pz,
                      const T* pair_const, T* pe) {
  const hn::ScalableTag<T> d;
  const auto one = hn::Set(d, T(1));
  auto u = hn::Zero(d);

  for (std::size_t i = 0; i < n; ++i) {
    const auto pxi = hn::Load(d, &px[i * k]);
    const auto pyi = hn::Load(d, &py[i * k]);
    const auto pzi = hn::Load(d, &pz[i * k]);
    for (std::size_t j = i + 1; j < n; ++j) {
      const auto rx = hn::Sub(pxi, hn::Load(d, &px[j * k]));
      const auto ry = hn::Sub(pyi, hn::Load(d, &py[j * k]));
      const auto rz = hn::Sub(pzi, hn::Load(d, &pz[j * k]));
      const auto dist2 = hn::MulAdd(rx, rx, hn::MulAdd(ry, ry, hn::Mul(rz, rz)));
      const auto inv_dist = hn::Div(one, hn::Sqrt(dist2));
      u = hn::MulAdd(hn::Set(d, pair_const[i * n + j]), inv_dist, u);
    }
  }
  hn::StoreU(u, d, pe);
}

}  // namespace batched
}  // namespace coulomb
