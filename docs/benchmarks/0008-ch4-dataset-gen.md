# 0008 — CH4, 10M-simulation dataset: realized throughput and oracle verification

- **Date / SHA / machine:** 2026-07-05 · `ebcce28` · AMD EPYC 7713 (Zen 3,
  single core allocated), the same host as [0007](0007-avx2-tuning.md); GCC
  13.3.0.

- **Hypothesis:** 0005–0007 measured the batched f32 lockstep integrator at
  N=10 on synthetic H-C-N-O chemistry (109k sims/sec on this host per 0007).
  The actual production target is smaller, real molecules — does the engine
  sustain a "10M sims in a couple of minutes" claim for one, and does its
  accuracy hold against the *original* ground truth: the fp64
  `scipy.solve_ivp`-to-t=1e10 oracle in `python/reference/coulomb.py` (no
  early stopping, no energy-redistribution shortcut), not just the scalar C++
  oracle 0005/0006 check against?

## Scope and framing

This is also the first exercise of `examples/` (see `examples/README.md`), a
new directory for end-to-end runnable demonstrations distinct from `tests/`
and `bench/`. Molecule: **CH4** (methane), singly ionized (C:+1, H:+1×4) —
the charge convention every existing ADR/benchmark uses, so this run stays
inside ADR [0004](../decisions/0004-precision-policy.md)'s validated accuracy
envelope. Sampler and tolerances match 0005-0007 exactly (`UniformSphereSampler`
radius 4.0, min_separation 0.25; f32 production point rtol 1e-4 / atol 1e-7 /
pe_stop 1e-5) so the N=5 vs. N=10 comparison below is apples-to-apples.

## Method

```sh
cmake --preset relwithdebinfo -DCOULOMB_BUILD_EXAMPLES=ON
cmake --build --preset relwithdebinfo --target coulomb_generate_dataset
taskset -c 66 ./build/relwithdebinfo/examples/coulomb_generate_dataset \
    --out /fastscratch/albin/coulomb_examples/ch4/dataset.bin --sims 10000000
python to_parquet.py --bin .../dataset.bin --out .../dataset.parquet
python verify_subset.py --bin .../dataset.bin --n 1000
```

- **Generation**: one `BatchedRK45<float>` instance (lanes = 8, this host's
  AVX2 f32 width, same as 0007), looped over 1,250,000 batches; per-batch
  geometries drawn from `UniformSphereSampler`, initial positions captured by
  the driver, final momenta taken from `BatchedRK45::Result` post
  energy-redistribution.
- **Parquet conversion**: `examples/to_parquet.py` memory-maps the raw binary
  with a matching `numpy` structured dtype and writes columns directly via
  `pyarrow` (no pandas dependency).
- **Oracle verification**: `examples/verify_subset.py` re-integrates the
  first 1000 geometries with `python/reference/coulomb.py`'s
  `simulate_explosion` (full fp64, t=1e10, no shortcuts — the same oracle
  `tests/reference/gen_reference_cases.py` validates the C++ engine against)
  and reports `|Δp|/|p|` against the dataset's f32 result, the same
  config-norm metric ADR 0004 and `bench_batched_integrator.cpp`'s
  correctness gate use.
- Evidence: `dataset.bin.meta.json` sidecar (molecule, sampler, tolerances,
  realized throughput) is written alongside every run; not checked into the
  repo (the dataset itself lives on `/fastscratch`, outside version control).

## Result

**Throughput:**

| | N | lanes (K) | sims/sec | mean steps | failures |
|---|---|---|----------|------------|----------|
| CH4 (this report)         | 5  | 8 | **255,465** | 27.6 | 0 / 10,000,000 |
| H-C-N-O, f32 prod (0007)  | 10 | 8 | 109,223     | 31.5 | 0 (gate-checked) |

10,000,000 simulations completed in **39.14 s** — comfortably inside "a
couple of minutes," in fact under one. Parquet conversion took 20.2 s for the
resulting 10M-row, 32-column file (1.79 GB, vs. 1.25 GB for the raw binary).

**Accuracy**, 1000 sims vs. the true fp64 no-early-stop oracle:

| metric | value |
|--------|-------|
| mean `\|Δp\|/\|p\|`   | 5.4e-06 |
| median                | 3.8e-06 |
| p99                   | 2.1e-05 |
| max                   | 1.8e-04 |
| exceeded 1% floor     | 0 / 1000 |

## Conclusion

- **The throughput claim holds, with margin to spare.** 255k sims/sec at
  N=5 — 39 seconds for the full 10M-sim run, not "a couple of minutes."
- **The N=5 vs. N=10 speedup (2.3×) is smaller than the pairwise-cost ratio
  (45/10 = 4.5×) would predict.** The force kernel is O(N²), but per-step
  integrator overhead (7-stage RK45, error norm, sampling) is not — it does
  not shrink as fast as the pair count does, so it makes up a larger share of
  the per-step cost at small N. Consistent with 0005/0006's own observation
  that non-force overhead cuts into the raw force-kernel ratio.
- **Accuracy is comfortably better than ADR 0004's N=10 numbers**, as
  expected with fewer atoms/pairs: max discrepancy 1.8e-4 here (~56× inside
  the 1% floor) vs. 0004's 7.4e-4 at N=10 (~13× inside). Zero of 1000 checked
  sims exceeded the floor, and this is checked against the *stricter*,
  original fp64 oracle (no redistribution shortcut), not just the C++ scalar
  oracle — closing the gap 0005/0006 left open (they validated against the
  scalar C++ RK45, not the pre-existing Python ground truth).
- **The dataset is real and usable**: 10M rows, 32 flat float32/uint columns
  (`x0..z4`, `px0..pz4`, `steps`, `converged`), Parquet schema metadata
  carrying the molecule/sampler/tolerance provenance, on `/fastscratch/albin`.

## Caveats

- **One molecule, one host.** N=5 CH4 only; the N=5-vs-N=10 overhead argument
  above is a plausible mechanism, not independently isolated (would need a
  per-step-cost breakdown like 0003's divide/sqrt analysis to confirm).
- **Oracle verification subset is 1000 of 10,000,000** (0.01%) — the fp64
  oracle is far too slow (full adaptive integration to t=1e10 per sim) to
  check exhaustively; 1000 is consistent with ADR 0004's own N=4000-geometry
  sweep methodology in spirit, just smaller given the oracle's cost here is
  paid interactively rather than as a batch job.
- **`examples/` has no default output path by design** — `--out` is required
  so the code stays reusable across users/machines; this run's
  `/fastscratch/albin` path is specific to this invocation (`/fastscratch` is
  a Beocat-cluster mount, not a portable convention — a different host would
  use a different path entirely), not a default anyone else's invocation
  would inherit.
- **Single-core cgroup allocation**, same as 0007 — not a dedicated idle
  machine; absolute sims/sec here should be read with the same caution as
  0007's numbers.

## Follow-ups

- Repeat across a small N sweep (e.g. N=4,5,6 with a couple of real
  molecules each) to properly separate the O(N²) force-kernel term from the
  fixed per-step integrator overhead this report attributes the sub-linear
  speedup to.
- Once Parquet dataset output is promoted from `examples/` to a first-class
  engine feature (per the top-level README's "still planned" list), revisit
  whether the binary-then-convert step is still the right design or whether
  direct C++ Parquet writing (Arrow C++) is worth the added build dependency.
