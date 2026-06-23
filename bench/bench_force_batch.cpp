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
// The batched kernel is templated on element type T (see docs/benchmarks/0005):
// it is self-contained Highway code, independent of the engine's compile-time
// Real, so f32 (K=16) and f64 (K=8) kernels coexist in one binary. The headline
// 0005 metric is f32-batched / f64-batched items/sec — how much of the doubled
// lane count survives the divide/sqrt port cap measured in 0003. The scalar
// baseline stays f64 (the engine's scalar kernel uses compile-time Real; an f32
// scalar would need the f32 build and is not worth it here).
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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "coulomb/molecule.hpp"
#include "coulomb/system.hpp"

namespace hn = hwy::HWY_NAMESPACE;
using namespace coulomb;

namespace {

// A batch of K simulations of an N-atom molecule, SoA over lanes. Templated on
// element type T so the f32 and f64 batched kernels share one definition.
template <class T>
struct Batch {
  std::size_t n{0};
  std::size_t k{0};  // SIMD lanes = sims in the batch.
  // Per-coordinate position buffers, length n*k, lane-strided.
  hwy::AlignedFreeUniquePtr<T[]> px, py, pz;
  hwy::AlignedFreeUniquePtr<T[]> ax, ay, az;  // acceleration output.
  std::vector<T> pair_const;                  // k*q_i*q_j per (i,j), length n*n.
  std::vector<T> inv_mass;                    // 1/m_i, length n.
};

// Same molecule across the batch (unit charge/mass, matching bench_force.cpp);
// each lane gets an independent random geometry from the same distribution.
// Geometry is drawn in double and narrowed to T on store, so the f32 and f64
// batches built with matching lane counts start from the same geometry up to
// f32 rounding.
template <class T>
Batch<T> make_batch(std::size_t n, std::size_t k, double coulomb_k = 1.0) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> pos(-1.0, 1.0);

  Batch<T> b;
  b.n = n;
  b.k = k;
  b.px = hwy::AllocateAligned<T>(n * k);
  b.py = hwy::AllocateAligned<T>(n * k);
  b.pz = hwy::AllocateAligned<T>(n * k);
  b.ax = hwy::AllocateAligned<T>(n * k);
  b.ay = hwy::AllocateAligned<T>(n * k);
  b.az = hwy::AllocateAligned<T>(n * k);

  std::vector<double> charge(n, 1.0), mass(n, 1.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t lane = 0; lane < k; ++lane) {
      b.px[i * k + lane] = static_cast<T>(pos(rng));
      b.py[i * k + lane] = static_cast<T>(pos(rng));
      b.pz[i * k + lane] = static_cast<T>(pos(rng));
    }
  }

  b.pair_const.assign(n * n, T(0));
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      b.pair_const[i * n + j] = static_cast<T>(coulomb_k * charge[i] * charge[j]);

  b.inv_mass.assign(n, T(0));
  for (std::size_t i = 0; i < n; ++i) b.inv_mass[i] = static_cast<T>(1.0 / mass[i]);
  return b;
}

// Batched kernel: compute K acceleration fields at once, mirroring the scalar
// CoulombForce::accelerations math lane-for-lane (1/sqrt, not rsqrt, so the
// numerics match the scalar baseline rather than trading accuracy for speed).
template <class T>
void accelerations_batched(const Batch<T>& b) {
  const hn::ScalableTag<T> d;
  const std::size_t n = b.n;
  const std::size_t k = b.k;
  const auto one = hn::Set(d, T(1));

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
template <class T>
void accelerations_batched_multonly(const Batch<T>& b) {
  const hn::ScalableTag<T> d;
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
// exact same total work (K sims) as one batched call. The scalar baseline is
// f64, so this consumes the f64 batch.
std::pair<Molecule, State> lane_system(const Batch<double>& b, std::size_t lane) {
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

// One-time correctness gate: f32-batched must match f64-batched to ~f32
// precision (consistent with 0004's accuracy story and 0003's 1.4e-14
// f64-vs-scalar check). Both kernels run on one shared geometry — narrowed to
// f32 for the float batch — so lanes [0, kd) carry identical inputs up to f32
// rounding. Returns the measured max relative difference (reported so the 0005
// writeup can cite it); aborts the binary if it exceeds tolerance.
double verify_f32_matches_f64() {
  const std::size_t n = 10;
  const hn::ScalableTag<double> dd;
  const hn::ScalableTag<float> df;
  const std::size_t kd = hn::Lanes(dd);
  const std::size_t kf = hn::Lanes(df);

  Batch<double> bd = make_batch<double>(n, kd);
  Batch<float> bf = make_batch<float>(n, kf);

  // Overwrite both with one shared geometry so lanes [0, kd) are identical up
  // to f32 rounding. The float batch's extra lanes [kd, kf) repeat lane 0 — the
  // kernel is a pure lane-wise map, so they run harmlessly and are excluded
  // from the compare below.
  std::mt19937 rng(999);
  std::uniform_real_distribution<double> pos(-1.0, 1.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t lane = 0; lane < kd; ++lane) {
      const double x = pos(rng), y = pos(rng), z = pos(rng);
      bd.px[i * kd + lane] = x;
      bd.py[i * kd + lane] = y;
      bd.pz[i * kd + lane] = z;
      bf.px[i * kf + lane] = static_cast<float>(x);
      bf.py[i * kf + lane] = static_cast<float>(y);
      bf.pz[i * kf + lane] = static_cast<float>(z);
    }
    for (std::size_t lane = kd; lane < kf; ++lane) {
      bf.px[i * kf + lane] = bf.px[i * kf];
      bf.py[i * kf + lane] = bf.py[i * kf];
      bf.pz[i * kf + lane] = bf.pz[i * kf];
    }
  }

  accelerations_batched(bd);
  accelerations_batched(bf);

  double max_rel = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t lane = 0; lane < kd; ++lane) {
      const double rx = bd.ax[i * kd + lane];
      const double ry = bd.ay[i * kd + lane];
      const double rz = bd.az[i * kd + lane];
      const double gx = bf.ax[i * kf + lane];
      const double gy = bf.ay[i * kf + lane];
      const double gz = bf.az[i * kf + lane];
      // Relative to the acceleration vector magnitude, so a near-zero component
      // (from cancellation) doesn't manufacture a spurious huge relative error.
      const double mag = std::sqrt(rx * rx + ry * ry + rz * rz);
      const double denom = std::max(mag, 1e-30);
      max_rel = std::max(max_rel, std::abs(gx - rx) / denom);
      max_rel = std::max(max_rel, std::abs(gy - ry) / denom);
      max_rel = std::max(max_rel, std::abs(gz - rz) / denom);
    }
  }

  // f32  epsilon is ~1.2e-7; with ~9 accumulated pair terms and some
  // cancellation, a correct kernel lands well under 1e-4. A templating bug
  // (wrong precision physics) would miss by orders of magnitude.
  constexpr double kTol = 1e-4;
  std::fprintf(stderr,
               "[gate] f32-batched vs f64-batched max relative diff = %.3e "
               "(N=%zu, kf=%zu, kd=%zu, tol=%.1e)\n",
               max_rel, n, kf, kd, kTol);
  if (!(max_rel < kTol)) {
    std::fprintf(stderr, "[gate] FAILED: f32 batched kernel disagrees with f64 beyond tolerance\n");
    std::abort();
  }
  return max_rel;
}

// Scalar baseline: K independent scalar force evaluations — the same total work
// as one batched call, so wall-time ratio is the realized lane speedup. Stays
// f64 (the conservative reference); K is the f64 lane count.
void BM_ForceScalarBatch(benchmark::State& bench) {
  const auto n = static_cast<std::size_t>(bench.range(0));
  const hn::ScalableTag<double> d;
  const std::size_t k = hn::Lanes(d);
  Batch<double> b = make_batch<double>(n, k);

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

template <class T>
void BM_ForceBatched(benchmark::State& bench) {
  const auto n = static_cast<std::size_t>(bench.range(0));
  const hn::ScalableTag<T> d;
  const std::size_t k = hn::Lanes(d);
  Batch<T> b = make_batch<T>(n, k);

  for (auto _ : bench) {
    accelerations_batched(b);
    benchmark::DoNotOptimize(b.ax.get());
    benchmark::ClobberMemory();
  }
  // K*N*N items per iteration, matching BM_ForceScalarBatch so the
  // per-N throughput rows are directly comparable. The lanes counter
  // auto-reports 8 (f64) vs 16 (f32).
  bench.SetItemsProcessed(bench.iterations() * static_cast<int64_t>(k) * static_cast<int64_t>(n) *
                          static_cast<int64_t>(n));
  bench.counters["lanes"] = static_cast<double>(k);
}

// Port-bound diagnostic: see accelerations_batched_multonly. Non-physical;
// reported only as the divide/sqrt-free floor of the batched kernel.
template <class T>
void BM_ForceBatchedNoDivSqrt(benchmark::State& bench) {
  const auto n = static_cast<std::size_t>(bench.range(0));
  const hn::ScalableTag<T> d;
  const std::size_t k = hn::Lanes(d);
  Batch<T> b = make_batch<T>(n, k);

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

// Production molecule-size sweep: N = 2..10 atoms. f64 and f32 batched kernels
// are registered side by side so the 0005 f32/f64 ratio reads off one run.
BENCHMARK(BM_ForceScalarBatch)->DenseRange(2, 10, 1);
BENCHMARK_TEMPLATE(BM_ForceBatched, double)->DenseRange(2, 10, 1);
BENCHMARK_TEMPLATE(BM_ForceBatched, float)->DenseRange(2, 10, 1);
BENCHMARK_TEMPLATE(BM_ForceBatchedNoDivSqrt, double)->DenseRange(2, 10, 1);
BENCHMARK_TEMPLATE(BM_ForceBatchedNoDivSqrt, float)->DenseRange(2, 10, 1);

// Custom main (the expansion of BENCHMARK_MAIN) so the f32-vs-f64 correctness
// gate runs once before any timing.
int main(int argc, char** argv) {
  verify_f32_matches_f64();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
