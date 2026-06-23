# 0004 — Precision policy: f64 the reference, f32 for production data generation

- **Status:** accepted
- **Date:** 2026-06-23

## Context

The engine carries precision as a single alias `Real` (`include/coulomb/types.hpp`),
selected at build time: `COULOMB_SINGLE_PRECISION` (the `*-f32` CMake presets)
makes the whole engine `float`, and the default is `double`. That switch was
introduced (commit `e039b51`) as an *experiment harness*, not a policy — a way to
diff f32 trajectories against f64. Two reports have since turned the experiment
into a decision that needs recording, because it cuts across the engine, the
validation suite, and the data-generation pipeline, and must stay consistent and
reversible:

- **[Benchmark 0004](../benchmarks/0004-precision-sweep.md) — accuracy.** Across
  16,000 explosions (N ∈ {3,5,8,10}, 4000 geometries each), f32 had **zero
  failures** and its worst asymptotic-momentum discrepancy from f64 was ~7e-4 —
  about **13× inside the ~1% experimental-momentum floor** — with zero-mean bias
  (~1e-10, nothing learnable). f32 *integration* is safe at the production scale.
- **[Benchmark 0005](../benchmarks/0005-f32-throughput.md) — throughput.** The
  f32 batched force kernel runs **~3.2× f64** at N=10 (beyond the 2× from doubled
  SIMD lanes — the surplus is f32's cheaper divide/sqrt), and f32's accuracy
  headroom permits looser RK45 tolerance (fewer steps per sim). Combined,
  **~16× projected** sims/sec over the f64 default.

The decision is not f32's *value* — that is measured. It is *where* each precision
is authoritative, *which tolerances* the f32 path runs at, and *what bar* it must
clear, kept reversible by construction.

## Decision

- **f64 remains the default and the reference.** `Real` defaults to `double`. f64
  is the correctness oracle, the precision for the test/reference fixtures
  (`tests/`), and the production fallback. We are **not** flipping the global
  default to float.
- **f32 is the sanctioned precision for production data generation**, via the
  existing compile-time switch (`COULOMB_SINGLE_PRECISION` / `relwithdebinfo-f32`).
  This is justified by 0004's **accuracy** clearance — throughput is the
  motivation, not the permission.
- **The f32 operating point is `rtol 1e-4`, `atol 1e-7`, `pe_stop 1e-5`** (0005
  Part B): fp32-safe (at/above fp32 ε ≈ 1.2e-7) and ~13× inside the floor. A
  looser `rtol 1e-3` (~20% fewer steps) is available *only* if the production
  geometry distribution is confirmed no tighter than the sampler's
  `min_separation = 0.25` a.u. — its floor margin is thin (~1.1×).
- **The acceptance bar is the ~1% experimental momentum resolution, not f64
  fidelity.** f32 is "good enough" while its f64 discrepancy stays comfortably
  under 1% with no systematic bias.

## Consequences

- **f64 stays the source of truth.** Validation, reference-fixture generation, and
  any result needing full fidelity run the default build; CI correctness is on
  f64. f32 is a *generation-time* choice layered on top, not a replacement.
- **Reversible by construction.** One compile-time switch, f64 default, no code
  committed to `float` — the force kernel is templated and `Real`-independent
  (0005 Part A), so f32 does not fork the hot path. If the policy is wrong, revert
  the build preset; nothing else changes.
- **The throughput win is projected, not yet realized.** The ~16× combines a
  measured per-step force-kernel speedup with a step-count reduction; the true
  end-to-end figure awaits the **batched integrator** (0003 follow-up #2), which
  pays the straggler / min-envelope penalties from
  [0002](../benchmarks/0002-explosion-dispersion.md). The decision rests on
  accuracy (proven) plus a throughput projection (strong prior), not on realized
  end-to-end throughput.
- **The definitive downstream check is still open.** Training the reconstruction
  model on f32- vs f64-generated data (0004 follow-up) is the final proof; the
  margins here are the prior that it passes. The reversibility above is the
  insurance if it does not.
- **f32 tolerances must stay at/above fp32 ε.** The f64 defaults (`rtol 1e-8`,
  `atol 1e-16`, `pe_stop 1e-9`) are sub-ε and meaningless in float; the operating
  point above is the sanctioned setting, recorded here and in 0005.
- **Conditioned on the current geometry distribution.** The accuracy margin holds
  at `min_separation = 0.25` a.u. and N ≤ 10. A tighter sampling floor, closer
  bonded pairs, or larger N raises the close-encounter tail — re-validate against
  0004 before relying on f32 there, and the thin `rtol 1e-3` margin would erode
  first.
