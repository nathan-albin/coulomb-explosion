# 0003 — Batched (SIMD-over-lanes) force kernel: per-lane ceiling

- **Date / SHA / machine:** 2026-06-22 · `ccc96ba` · 11th Gen Intel Core
  i7-11800H (Tiger Lake, 8C/16T, AVX-512), L1d 48 KiB/core, L2 1.25 MiB/core,
  L3 24 MiB
- **Hypothesis:** The production lever for the N = 2–10 atom workload is
  SIMD-over-lanes — one independent simulation per SIMD lane (see [0001](0001-force-kernel-baseline.md)
  and [0002](0002-explosion-dispersion.md)). This experiment builds the batched
  Coulomb force kernel in Highway and measures its **per-lane ceiling**: how much
  of the K-wide lane count an honest f64 kernel actually captures, and what caps
  it. This is the force component only — *not* a realized lockstep speedup, which
  also pays the min-envelope / straggler penalties quantified in 0002.

## Scope

This isolates the force kernel: K independent geometries of one fixed molecule,
one per lane, accelerations computed for all K at once. It deliberately omits the
RK45 bookkeeping (stage combinations, error norm, step control) and the adaptive
divergence across lanes. Those belong to the batched integrator (0002's first
follow-up) and dominate the per-*step* cost at small N; here we measure only the
per-*pair* force ceiling that the integrator's force evaluations will inherit.

What carries forward from 0001: the in-binary scalar baseline reproduces 0001's
N = 8 figure (537 vs 541 M pair-items/s), confirming this harness measures the
same kernel on the same machine — so the batched/scalar ratio is a clean speedup.

## Method

- Harness: `bench/bench_force_batch.cpp` (Google Benchmark), three benchmarks
  over a molecule-size sweep N = 2…10:
  - **`BM_ForceScalarBatch`** — K independent scalar `CoulombForce::accelerations`
    calls (the same total work as one batched call; wall-time ratio is the
    realized lane speedup).
  - **`BM_ForceBatched`** — the Highway SIMD-over-lanes kernel, one sim per lane,
    mirroring the scalar math lane-for-lane (`1/sqrt` + two divides, *not* an
    rsqrt approximation — numerics match the scalar baseline to 1.4e-14, FMA
    reassociation noise only).
  - **`BM_ForceBatchedNoDivSqrt`** — a **port-bound diagnostic, not a real
    kernel** (the physics is deliberately wrong): identical data flow and memory
    traffic, but the per-pair `1/sqrt` + two divides are replaced by two
    multiplies. The gap to `BM_ForceBatched` isolates the divide/sqrt-port share.
- Layout: SoA over lanes. For each coordinate, `p[i*K .. i*K+K)` holds atom `i`
  across the K sims. Charges/masses are per-atom scalars broadcast across lanes —
  the production workload is many geometries of one fixed molecule.
- Build: `coulomb_force_batch` is compiled `-march=native` so Highway targets the
  host's widest ISA: **AVX-512 → K = 8 doubles/vector** on this node. The rest of
  the project stays generic (SSE2), so 0001's scalar floor remains conservative.
- Commands (idle laptop, RelWithDebInfo, GCC 13.3.0):

  ```bash
  cmake --preset relwithdebinfo && cmake --build --preset relwithdebinfo \
      --target coulomb_force_batch
  ./build/relwithdebinfo/bench/coulomb_force_batch \
      --benchmark_repetitions=15 --benchmark_report_aggregates_only=true \
      --benchmark_out=docs/benchmarks/0003-batched-force-kernel.json \
      --benchmark_out_format=json
  ```

- Raw evidence committed alongside:
  [`0003-batched-force-kernel.json`](0003-batched-force-kernel.json).

## Result

Mean CPU time of 15 repetitions (cv ≤ 2.5% for all rows; the batched kernel is
the steadiest, cv ≈ 0.6–0.8%). "Speedup" is scalar ÷ batched. "div/sqrt share" is
`(batched − nodivsqrt) / batched`. "floor" is scalar ÷ nodivsqrt — the speedup an
idealized divide/sqrt-free kernel would reach (an optimistic upper bound, since a
real rsqrt+refinement is not free).

| N  | scalar (ns) | batched (ns) | nodivsqrt (ns) | speedup | div/sqrt share | floor |
|----|-------------|--------------|----------------|---------|----------------|-------|
| 2  | 80.9        | 15.1         | 6.41           | 5.36×   | 58%            | 12.6× |
| 3  | 143         | 40.6         | 14.4           | 3.52×   | 65%            | 9.9×  |
| 4  | 236         | 80.6         | 25.5           | 2.93×   | 68%            | 9.3×  |
| 5  | 373         | 134          | 40.5           | 2.78×   | 70%            | 9.2×  |
| 6  | 542         | 200          | 59.2           | 2.71×   | 70%            | 9.2×  |
| 7  | 726         | 282          | 81.2           | 2.57×   | 71%            | 8.9×  |
| 8  | 952         | 371          | 108            | 2.57×   | 71%            | 8.8×  |
| 9  | 1209        | 477          | 138            | 2.53×   | 71%            | 8.8×  |
| 10 | 1492        | 608          | 173            | 2.45×   | 72%            | 8.6×  |

## Conclusion

- **The lanes fill, but the honest f64 kernel captures only ~2.5× of the 8× lane
  width** at the production-relevant N = 8–10. Correctness is verified and K = 8
  is confirmed, so this is not a packing failure — each lane simply runs at ~⅓ of
  the scalar per-pair rate because the work it is bound on does not widen.
- **The cap is the divide/sqrt execution port, not the algorithm.** It is a flat
  ~71% of the batched kernel's time at N ≥ 7, and removing it (the multiply-only
  diagnostic) runs ~3.5× faster. AVX-512's divide/sqrt unit delivers only ~3× the
  scalar `vsqrtpd`/`vdivpd` throughput, not 8×; multiply/FMA ports give the full
  width (the divide/sqrt-free floor reaches ~8.6× scalar). 0001 predicted exactly
  this ("the dominant cost is the per-pair sqrt + reciprocal").
- **Ignore the small-N speedup.** The 5.4× at N = 2 is scalar per-call overhead
  (`forces.assign` memset + call cost) amortized by the single batched call, not
  kernel speedup; it decays to the true ~2.5× as O(N²) work takes over. The
  asymptotic figure (N ≥ 8) is the one to quote.
- **Headroom is large and the lever is named.** ~71% of the kernel is recoverable
  in principle. An rsqrt seed (`vrsqrt14pd`, 14-bit) refined with one or two
  Newton–Raphson steps computes `dist2^(-3/2)` with multiplies on the FMA ports,
  bypassing the divide/sqrt port. The 8.6× floor is the optimistic ceiling; a
  realistic landing — seed + refinement still cost a few FMAs — is plausibly in
  the 5–7× range. That is the next experiment.

## Follow-ups

- **rsqrt + Newton–Raphson variant.** Implement `dist2^(-3/2)` via `vrsqrt14pd`
  + NR refinement, measure realized speedup against this report's 2.5× and the
  8.6× floor, and quantify the f64 accuracy cost (relative error per pair, and
  whether it perturbs the integrator's accepted step count vs the oracle). This
  is where the bulk of the remaining throughput lives.
- **Batched integrator (0002's first follow-up).** Feed this kernel into a
  shared-dt lockstep step and measure realized speedup including the RK
  bookkeeping share and the min-envelope / straggler penalties — the number this
  report's per-lane ceiling gets multiplied into.
- **Re-measure on the HPC target node.** The speedup-vs-N curve and the
  divide/sqrt share are the portable artifacts; a different ISA width (or a
  better-pipelined divide/sqrt unit) shifts both.
- **f32 variant.** Halving element size doubles K (16 lanes) and changes the
  divide/sqrt economics; worth measuring if the accuracy budget allows it.
