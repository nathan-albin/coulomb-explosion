# Python reference & analysis

This directory holds:

1. **The reference implementation** — the existing Python Coulomb-explosion
   simulator (RK45 + Parquet output). It is the ground truth for correctness:
   the C++ engine's trajectories and energies are validated against it to
   tolerance, and its RK45 scheme is the one being ported to C++.

2. **Analysis & reporting** — notebooks/scripts that read the generated Parquet
   output and produce the plots and tables used in `../docs/` (accuracy vs.
   solver, throughput vs. configuration, etc.).

## Suggested layout (once the code lands)

```
python/
  reference/        the existing simulator
  analysis/         plotting + report generation
  pyproject.toml    or requirements.txt
```

Drop the existing implementation in here and I can wire up a correctness test
that diffs C++ vs. Python trajectories.
