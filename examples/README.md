# Examples

Runnable, end-to-end demonstrations of the engine at production scale —
distinct from `tests/` (correctness) and `bench/` (microbenchmarks/dispersion
studies). Not built by default: `COULOMB_BUILD_EXAMPLES` is `OFF` unless you
opt in.

## CH4 dataset generation (docs/benchmarks/0008)

Generates millions of independent CH4 (methane) Coulomb-explosion
simulations with the batched f32 lockstep integrator, at the ADR
[0004](../docs/decisions/0004-precision-policy.md) production tolerance
point, writes them to a Parquet dataset, and verifies a subset against the
true fp64 oracle (no early stopping).

### 1. Build

```sh
cmake --preset relwithdebinfo -DCOULOMB_BUILD_EXAMPLES=ON
cmake --build --preset relwithdebinfo --target coulomb_generate_dataset
```

### 2. Generate

`--out` is required — there is no built-in default output path. Pick a
location with room for a multi-GB file (this example does **not** default to
any particular user's scratch directory):

```sh
./build/relwithdebinfo/examples/coulomb_generate_dataset \
    --out /path/to/your/scratch/coulomb_examples/ch4/dataset.bin \
    --sims 10000000
```

Prints a progress line to stderr and a final summary (wall time, sims/sec,
convergence failures) to stdout. Writes `dataset.bin` plus a
`dataset.bin.meta.json` sidecar (molecule, sampler, and tolerance settings —
`to_parquet.py` and `verify_subset.py` both read it, so keep it next to the
binary).

For a quick smoke test before committing to the full run: `--sims 10000`
finishes in well under a second and exercises the same code path.

### 3. Convert to Parquet

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python to_parquet.py --bin /path/to/dataset.bin --out /path/to/dataset.parquet
```

Columns are flat and index-based (CH4 repeats "H", so symbol-based names would
collide): `x0,y0,z0,...,x4,y4,z4` (initial positions, atom order C,H,H,H,H —
see the sidecar's `symbols`), `px0,py0,pz0,...,px4,py4,pz4` (final momenta),
plus `steps` and `converged`. Molecule/sampler/tolerance settings are attached
as Parquet schema metadata.

### 4. Verify a subset against the true oracle

```sh
python verify_subset.py --bin /path/to/dataset.bin --n 1000
```

Re-integrates the first `--n` geometries with
`python/reference/coulomb.py`'s `simulate_explosion` — full fp64
`scipy.solve_ivp` to t=1e10, no early stopping, no energy-redistribution
shortcut, the same oracle `tests/reference/gen_reference_cases.py` uses for
the C++ engine's own correctness tests — and reports the `|dp|/|p|`
discrepancy against the fast dataset (mean/median/p99/max), same metric ADR
0004 uses, against the 1% experimental-momentum floor. This is slow per-sim
(a full adaptive integration to t=1e10) — keep `--n` in the hundreds to low
thousands, not the full dataset.
