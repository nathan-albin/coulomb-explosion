# 0001 — Force kernel scalar baseline

- **Date / SHA / machine:** 2026-06-21 · `5e90c56` · 11th Gen Intel Core
  i7-11800H (Tiger Lake, 8C/16T), L1d 48 KiB/core, L2 1.25 MiB/core, L3 24 MiB
- **Hypothesis:** Establish the reference throughput of the naive O(N²) all-pairs
  Coulomb kernel (`CoulombForce::accelerations`) so every later blocked/SIMD
  variant has a fixed point to be compared against. No optimization is under
  test here — this is the floor.

> **Scope (added 2026-06-21):** This characterizes the scalar kernel's per-pair
> throughput swept to large N. The production workload is **N = 2–10 atoms ×
> millions of independent sims**, where the optimization lever is
> **SIMD-over-lanes (one simulation per lane)**, not SIMD-over-j. The large-N
> rows here describe the kernel's asymptotic behavior, not the production hot
> path; the directly relevant row is **N = 8**. The production baseline is 0002.
> What carries forward from this report: the Google Benchmark harness and
> methodology, the N = 8 data point, and per-pair scalar throughput as a sanity
> floor — batched SIMD should approach ~width× this figure per lane, which is how
> you confirm the lanes are actually filled.

## Method

- Benchmark: `bench/bench_force.cpp` (`BM_CoulombForce`), Google Benchmark,
  sweeping N = 8…1024 by ×2 over a random unit-charge system (fixed seed).
- Build: `cmake --preset relwithdebinfo` → RelWithDebInfo, `-O2 -g -DNDEBUG`,
  GCC 13.3.0.
  **No `-march`**, so this is the generic x86-64 ISA (SSE2) — no AVX2/AVX-512,
  no `-ffast-math`. That is deliberate: it is the conservative scalar floor the
  ISA-targeted variants must beat.
- Command (idle machine, load 0.36, CPU scaling disabled):

  ```bash
  ./build/relwithdebinfo/bench/coulomb_bench \
    --benchmark_repetitions=15 --benchmark_report_aggregates_only=true \
    --benchmark_out=docs/benchmarks/0001-force-kernel-baseline.json \
    --benchmark_out_format=json
  ```

- Raw evidence committed alongside this report:
  [`0001-force-kernel-baseline.json`](0001-force-kernel-baseline.json). Use it as
  the `base.json` for `compare.py` when measuring a candidate.

## Result

Throughput is reported N²-normalized (the counter uses N² items/iter, i.e. it
includes self- and double-counted pairs; the unique interaction count is
N(N−1)/2, so per-*unique*-interaction throughput is ≈2× the figures below). Mean
of 15 repetitions; `cv` is the coefficient of variation.

| N    | CPU time (mean) | cv    | Throughput (N²-norm.) |
|------|-----------------|-------|-----------------------|
| 8    | 118 ns          | 2.10% | 541 M/s               |
| 16   | 470 ns          | 0.44% | 545 M/s               |
| 32   | 1.91 µs         | 0.81% | 537 M/s               |
| 64   | 8.00 µs         | 2.83% | 512 M/s               |
| 128  | 31.2 µs         | 1.07% | 525 M/s               |
| 256  | 129 µs          | 3.06% | 510 M/s               |
| 512  | 546 µs          | 1.45% | 480 M/s               |
| 1024 | 2.19 ms         | 1.41% | 479 M/s               |

## Conclusion

- **Baseline ≈ 480–545 M N²-normalized pair-ops/s**, essentially flat in N with a
  gentle ~12% decline from small N to N ≥ 512. Speedups for future variants
  should be quoted against the same-N row here.
- **No cache cliff in this range.** Even at N = 1024 the working set (positions
  + accelerations ≈ 2 × 1024 × 24 B ≈ 48 KiB) only just reaches the L1d edge and
  stays well within L2, so the kernel is compute-bound, not memory-bound. The
  cliff the benchmark comment anticipates will appear at larger N (or once the
  inner loop is vectorized and pressure shifts).
- **Headroom is large and untapped.** The kernel is scalar `double` on a CPU
  with AVX2 (and AVX-512). The dominant cost is the per-pair `sqrt` +
  reciprocal; a SIMD variant over j with a vectorized rsqrt is the obvious first
  lever *at large N*. At N = 2–10 — the production size — the inner-j vectors are
  too short to pay off, and the real lever is batching independent sims across
  lanes (SIMD-over-lanes; see 0002).

## Follow-ups

- Add an `-march=native` (or explicit AVX2) build variant and re-measure as
  `0002-*` to separate "free" auto-vectorization from the algorithm.
- **Production lever:** implement and benchmark a SIMD-over-lanes batch (one
  simulation per lane, Highway) — the right axis for the N = 2–10 workload.
  A SIMD-over-j force kernel only helps at large N, which this workload never
  reaches; treat it as a secondary/large-N experiment, not the production path.
- Extend the N sweep past 1024 to locate the actual L2→L3→DRAM cliffs.
  **Not on the critical path** for this project — a generic HPC exercise at
  sizes the production workload never runs.
- Separate experiment (different shape — cost vs. accuracy, not throughput):
  sweep `pe_stop_fraction` in the explosion driver, emitting steps + asymptotic
  momentum error vs. the oracle. Warrants its own `bench_explosion` + CSV rather
  than `compare.py`.
