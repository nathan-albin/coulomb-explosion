// Microbenchmark for the SIMD-over-lanes batched Coulomb force kernel — the
// production lever for the N = 2-10 atom workload (see docs/benchmarks/0001 and
// 0002). Where bench_force.cpp vectorizes nothing (the scalar O(N^2) floor),
// this packs K independent simulations of the *same* molecule — one per SIMD
// lane — and computes all K acceleration fields at once.
//
// This measures the force component's per-lane ceiling only: a pure
// lane-parallel map with no divergence. It is NOT a measurement of realized
// lockstep speedup (that lives in the batched integrator and pays the
// min-envelope / straggler penalties quantified in 0002). What we want here is
// the answer to 0001's check: "batched SIMD should approach ~width x the scalar
// per-pair throughput per lane" — i.e. confirm the lanes actually fill.
//
// Run with: ./build/relwithdebinfo/bench/coulomb_force_batch
//
// Layout: one SIMD lane per simulation. Positions are SoA over lanes — for each
// coordinate, px[i*K .. i*K+K) holds atom i's x across the K sims. Charges and
// masses are per-atom scalars broadcast across lanes: the production workload is
// millions of geometries of one fixed molecule, so the molecule is shared and
// only the geometry varies lane to lane.

#include <benchmark/benchmark.h>
#include <hwy/aligned_allocator.h>
#include <hwy/highway.h>

#include <cstddef>
#include <random>
#include <vector>

#include "coulomb/molecule.hpp"
#include "coulomb/system.hpp"

namespace hn = hwy::HWY_NAMESPACE;
using namespace coulomb;

namespace {

// A batch of K simulations of an N-atom molecule, SoA over lanes.
struct Batch {
  std::size_t n{0};
  std::size_t k{0};  // SIMD lanes = sims in the batch.
  // Per-coordinate position buffers, length n*k, lane-strided.
  hwy::AlignedFreeUniquePtr<double[]> px, py, pz;
  hwy::AlignedFreeUniquePtr<double[]> ax, ay, az;  // acceleration output.
  std::vector<double> pair_const;                  // k*q_i*q_j per (i,j), length n*n.
  std::vector<double> inv_mass;                    // 1/m_i, length n.
};

// Same molecule across the batch (unit charge/mass, matching bench_force.cpp);
// each lane gets an independent random geometry from the same distribution.
Batch make_batch(std::size_t n, std::size_t k, Real coulomb_k = 1.0) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> pos(-1.0, 1.0);

  Batch b;
  b.n = n;
  b.k = k;
  b.px = hwy::AllocateAligned<double>(n * k);
  b.py = hwy::AllocateAligned<double>(n * k);
  b.pz = hwy::AllocateAligned<double>(n * k);
  b.ax = hwy::AllocateAligned<double>(n * k);
  b.ay = hwy::AllocateAligned<double>(n * k);
  b.az = hwy::AllocateAligned<double>(n * k);

  std::vector<double> charge(n, 1.0), mass(n, 1.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t lane = 0; lane < k; ++lane) {
      b.px[i * k + lane] = pos(rng);
      b.py[i * k + lane] = pos(rng);
      b.pz[i * k + lane] = pos(rng);
    }
  }

  b.pair_const.assign(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) b.pair_const[i * n + j] = coulomb_k * charge[i] * charge[j];

  b.inv_mass.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) b.inv_mass[i] = 1.0 / mass[i];
  return b;
}

// Batched kernel: compute K acceleration fields at once, mirroring the scalar
// CoulombForce::accelerations math lane-for-lane (1/sqrt, not rsqrt, so the
// numerics match the scalar baseline rather than trading accuracy for speed).
void accelerations_batched(const Batch& b) {
  const hn::ScalableTag<double> d;
  const std::size_t n = b.n;
  const std::size_t k = b.k;
  const auto one = hn::Set(d, 1.0);

  for (std::size_t i = 0; i < n; ++i) {
    hn::Store(hn::Zero(d), d, &b.ax[i * k]);
    hn::Store(hn::Zero(d), d, &b.ay[i * k]);
    hn::Store(hn::Zero(d), d, &b.az[i * k]);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto pxi = hn::Load(d, &b.px[i * k]);
    const auto pyi = hn::Load(d, &b.py[i * k]);
    const auto pzi = hn::Load(d, &b.pz[i * k]);
    auto axi = hn::Load(d, &b.ax[i * k]);
    auto ayi = hn::Load(d, &b.ay[i * k]);
    auto azi = hn::Load(d, &b.az[i * k]);

    for (std::size_t j = i + 1; j < n; ++j) {
      const auto rx = hn::Sub(pxi, hn::Load(d, &b.px[j * k]));
      const auto ry = hn::Sub(pyi, hn::Load(d, &b.py[j * k]));
      const auto rz = hn::Sub(pzi, hn::Load(d, &b.pz[j * k]));

      const auto dist2 = hn::MulAdd(rx, rx, hn::MulAdd(ry, ry, hn::Mul(rz, rz)));
      const auto inv_dist = hn::Div(one, hn::Sqrt(dist2));
      const auto inv_dist3 = hn::Div(inv_dist, dist2);
      const auto scale = hn::Mul(hn::Set(d, b.pair_const[i * n + j]), inv_dist3);

      const auto fx = hn::Mul(rx, scale);
      const auto fy = hn::Mul(ry, scale);
      const auto fz = hn::Mul(rz, scale);

      axi = hn::Add(axi, fx);
      ayi = hn::Add(ayi, fy);
      azi = hn::Add(azi, fz);
      hn::Store(hn::Sub(hn::Load(d, &b.ax[j * k]), fx), d, &b.ax[j * k]);
      hn::Store(hn::Sub(hn::Load(d, &b.ay[j * k]), fy), d, &b.ay[j * k]);
      hn::Store(hn::Sub(hn::Load(d, &b.az[j * k]), fz), d, &b.az[j * k]);
    }

    hn::Store(axi, d, &b.ax[i * k]);
    hn::Store(ayi, d, &b.ay[i * k]);
    hn::Store(azi, d, &b.az[i * k]);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto m = hn::Set(d, b.inv_mass[i]);
    hn::Store(hn::Mul(hn::Load(d, &b.ax[i * k]), m), d, &b.ax[i * k]);
    hn::Store(hn::Mul(hn::Load(d, &b.ay[i * k]), m), d, &b.ay[i * k]);
    hn::Store(hn::Mul(hn::Load(d, &b.az[i * k]), m), d, &b.az[i * k]);
  }
}

// Port-bound diagnostic (NOT a real kernel — the physics is deliberately
// wrong). Identical data flow, memory traffic, and pair count to
// accelerations_batched, but the per-pair 1/sqrt + two divides are replaced by
// two multiplies. The gap between this and accelerations_batched isolates how
// much of the batched kernel's time is spent on the divide/sqrt execution port
// — the quantity that says whether an rsqrt reformulation is worth pursuing.
void accelerations_batched_multonly(const Batch& b) {
  const hn::ScalableTag<double> d;
  const std::size_t n = b.n;
  const std::size_t k = b.k;

  for (std::size_t i = 0; i < n; ++i) {
    hn::Store(hn::Zero(d), d, &b.ax[i * k]);
    hn::Store(hn::Zero(d), d, &b.ay[i * k]);
    hn::Store(hn::Zero(d), d, &b.az[i * k]);
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto pxi = hn::Load(d, &b.px[i * k]);
    const auto pyi = hn::Load(d, &b.py[i * k]);
    const auto pzi = hn::Load(d, &b.pz[i * k]);
    auto axi = hn::Load(d, &b.ax[i * k]);
    auto ayi = hn::Load(d, &b.ay[i * k]);
    auto azi = hn::Load(d, &b.az[i * k]);

    for (std::size_t j = i + 1; j < n; ++j) {
      const auto rx = hn::Sub(pxi, hn::Load(d, &b.px[j * k]));
      const auto ry = hn::Sub(pyi, hn::Load(d, &b.py[j * k]));
      const auto rz = hn::Sub(pzi, hn::Load(d, &b.pz[j * k]));

      const auto dist2 = hn::MulAdd(rx, rx, hn::MulAdd(ry, ry, hn::Mul(rz, rz)));
      // Stand-in for inv_dist3, multiplies only — no divide/sqrt port traffic.
      const auto inv_dist3 = hn::Mul(hn::Mul(dist2, dist2), dist2);
      const auto scale = hn::Mul(hn::Set(d, b.pair_const[i * n + j]), inv_dist3);

      const auto fx = hn::Mul(rx, scale);
      const auto fy = hn::Mul(ry, scale);
      const auto fz = hn::Mul(rz, scale);

      axi = hn::Add(axi, fx);
      ayi = hn::Add(ayi, fy);
      azi = hn::Add(azi, fz);
      hn::Store(hn::Sub(hn::Load(d, &b.ax[j * k]), fx), d, &b.ax[j * k]);
      hn::Store(hn::Sub(hn::Load(d, &b.ay[j * k]), fy), d, &b.ay[j * k]);
      hn::Store(hn::Sub(hn::Load(d, &b.az[j * k]), fz), d, &b.az[j * k]);
    }

    hn::Store(axi, d, &b.ax[i * k]);
    hn::Store(ayi, d, &b.ay[i * k]);
    hn::Store(azi, d, &b.az[i * k]);
  }
}

// Reconstruct an AoS scalar system for one lane so the scalar baseline does the
// exact same total work (K sims) as one batched call.
std::pair<Molecule, State> lane_system(const Batch& b, std::size_t lane) {
  Molecule mol;
  State state;
  mol.atoms.reserve(b.n);
  state.positions.reserve(b.n);
  for (std::size_t i = 0; i < b.n; ++i) {
    mol.atoms.push_back({"X", 1.0, 1.0});
    state.positions.push_back({b.px[i * b.k + lane], b.py[i * b.k + lane], b.pz[i * b.k + lane]});
  }
  state.velocities.assign(b.n, Vec3{});
  return {std::move(mol), std::move(state)};
}

// Scalar baseline: K independent scalar force evaluations — the same total work
// as one batched call, so wall-time ratio is the realized lane speedup.
void BM_ForceScalarBatch(benchmark::State& bench) {
  const auto n = static_cast<std::size_t>(bench.range(0));
  const hn::ScalableTag<double> d;
  const std::size_t k = hn::Lanes(d);
  Batch b = make_batch(n, k);

  std::vector<Molecule> mols;
  std::vector<State> states;
  for (std::size_t lane = 0; lane < k; ++lane) {
    auto [m, s] = lane_system(b, lane);
    mols.push_back(std::move(m));
    states.push_back(std::move(s));
  }
  CoulombForce force(1.0);
  std::vector<Vec3> acc;

  for (auto _ : bench) {
    for (std::size_t lane = 0; lane < k; ++lane) {
      force.accelerations(mols[lane], states[lane], acc);
      benchmark::DoNotOptimize(acc.data());
    }
    benchmark::ClobberMemory();
  }
  bench.SetItemsProcessed(bench.iterations() * static_cast<int64_t>(k) * static_cast<int64_t>(n) *
                          static_cast<int64_t>(n));
  bench.counters["lanes"] = static_cast<double>(k);
}

void BM_ForceBatched(benchmark::State& bench) {
  const auto n = static_cast<std::size_t>(bench.range(0));
  const hn::ScalableTag<double> d;
  const std::size_t k = hn::Lanes(d);
  Batch b = make_batch(n, k);

  for (auto _ : bench) {
    accelerations_batched(b);
    benchmark::DoNotOptimize(b.ax.get());
    benchmark::ClobberMemory();
  }
  // K*N*N items per iteration, matching BM_ForceScalarBatch so the
  // per-N throughput rows are directly comparable.
  bench.SetItemsProcessed(bench.iterations() * static_cast<int64_t>(k) * static_cast<int64_t>(n) *
                          static_cast<int64_t>(n));
  bench.counters["lanes"] = static_cast<double>(k);
}

// Port-bound diagnostic: see accelerations_batched_multonly. Non-physical;
// reported only as the divide/sqrt-free floor of the batched kernel.
void BM_ForceBatchedNoDivSqrt(benchmark::State& bench) {
  const auto n = static_cast<std::size_t>(bench.range(0));
  const hn::ScalableTag<double> d;
  const std::size_t k = hn::Lanes(d);
  Batch b = make_batch(n, k);

  for (auto _ : bench) {
    accelerations_batched_multonly(b);
    benchmark::DoNotOptimize(b.ax.get());
    benchmark::ClobberMemory();
  }
  bench.SetItemsProcessed(bench.iterations() * static_cast<int64_t>(k) * static_cast<int64_t>(n) *
                          static_cast<int64_t>(n));
  bench.counters["lanes"] = static_cast<double>(k);
}

}  // namespace

// Production molecule-size sweep: N = 2..10 atoms.
BENCHMARK(BM_ForceScalarBatch)->DenseRange(2, 10, 1);
BENCHMARK(BM_ForceBatched)->DenseRange(2, 10, 1);
BENCHMARK(BM_ForceBatchedNoDivSqrt)->DenseRange(2, 10, 1);

BENCHMARK_MAIN();
