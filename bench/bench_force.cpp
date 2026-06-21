// Microbenchmark for the Coulomb force kernel — the expected hot loop.
//
// Run with: ./build/bench/coulomb_bench
// This establishes the scalar baseline that blocked / SIMD variants are
// measured against. Sweep over N to expose the O(N^2) scaling and any cache
// cliffs.

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "coulomb/molecule.hpp"
#include "coulomb/system.hpp"

using namespace coulomb;

namespace {

std::pair<Molecule, State> random_system(std::size_t n) {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<Real> pos(-1.0, 1.0);

  Molecule mol;
  State state;
  mol.atoms.reserve(n);
  state.positions.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    mol.atoms.push_back({"X", 1.0, 1.0});
    state.positions.push_back({pos(rng), pos(rng), pos(rng)});
  }
  state.velocities.assign(n, Vec3{});
  return {std::move(mol), std::move(state)};
}

void BM_CoulombForce(benchmark::State& bench) {
  const auto n = static_cast<std::size_t>(bench.range(0));
  auto [mol, state] = random_system(n);
  CoulombForce force(1.0);
  std::vector<Vec3> acc;

  for (auto _ : bench) {
    force.accelerations(mol, state, acc);
    benchmark::DoNotOptimize(acc.data());
    benchmark::ClobberMemory();
  }
  // Report pairwise interactions/sec to normalize across N.
  bench.SetItemsProcessed(bench.iterations() *
                          static_cast<int64_t>(n) * static_cast<int64_t>(n));
}

}  // namespace

BENCHMARK(BM_CoulombForce)->RangeMultiplier(2)->Range(8, 1024);

BENCHMARK_MAIN();
