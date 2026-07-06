#!/usr/bin/env python3
"""Verify a subset of a generated dataset against the true fp64 oracle.

The fast dataset (examples/generate_dataset.cpp) uses the f32 batched
lockstep integrator at the ADR 0004 production tolerance (rtol 1e-4) with the
energy-redistribution shortcut for early stopping. This script checks a small
subset against `python/reference/coulomb.py`'s `simulate_explosion` — full
fp64 `scipy.solve_ivp` integrated to t=1e10, no early stopping, no
redistribution — the original ground truth the C++ engine itself is validated
against (see tests/reference/gen_reference_cases.py for the same import
pattern).

For each of the first `--n` records: re-run the true oracle from the same
initial geometry (v_init = 0, matching the engine's own convention) and
compare its asymptotic momenta against the dataset's fast f32 result, using
the same |dp|/|p| config-norm ADR 0004 and bench_batched_integrator.cpp use.

Usage:
  python verify_subset.py --bin dataset.bin --n 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python" / "reference"))
import coulomb as co  # noqa: E402


def record_dtype(n_atoms: int) -> np.dtype:
    return np.dtype(
        [
            ("init_pos", "<f4", (n_atoms * 3,)),
            ("final_p", "<f4", (n_atoms * 3,)),
            ("steps", "<u4"),
            ("converged", "u1"),
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", required=True)
    p.add_argument("--n", type=int, default=1000, help="number of records to verify (from the start)")
    p.add_argument("--floor", type=float, default=0.01, help="experimental-momentum floor (ADR 0004: 1%%)")
    args = p.parse_args()

    with open(args.bin + ".meta.json") as f:
        meta = json.load(f)

    n_atoms = meta["n_atoms"]
    masses = co.convert_masses(np.asarray(meta["masses_amu"], dtype=float))
    charges = np.asarray(meta["charges"], dtype=float)

    dtype = record_dtype(n_atoms)
    arr = np.memmap(args.bin, dtype=dtype, mode="r")
    n = min(args.n, len(arr))

    rel_errors = np.empty(n)
    t0 = time.time()
    for i in range(n):
        x_init = arr["init_pos"][i].reshape(n_atoms, 3).astype(float)
        v_init = np.zeros_like(x_init)
        p_fast = arr["final_p"][i].reshape(n_atoms, 3).astype(float)

        _, v_final = co.simulate_explosion(x_init, v_init, masses, charges)
        p_oracle = masses[:, None] * v_final

        num = np.linalg.norm((p_fast - p_oracle).ravel())
        den = np.linalg.norm(p_oracle.ravel())
        rel_errors[i] = num / den

        if (i + 1) % max(1, n // 20) == 0 or i + 1 == n:
            elapsed = time.time() - t0
            print(f"\r{i + 1}/{n} verified ({elapsed:.1f}s elapsed)", end="", file=sys.stderr)
    print(file=sys.stderr)

    n_over = int(np.sum(rel_errors > args.floor))
    print(f"\nVerified {n} sims against the fp64 no-early-stop oracle (python/reference/coulomb.py)")
    print(f"  |dp|/|p|  mean={rel_errors.mean():.3e}  median={np.median(rel_errors):.3e}  "
          f"p99={np.percentile(rel_errors, 99):.3e}  max={rel_errors.max():.3e}")
    print(f"  {n_over}/{n} exceeded the {args.floor:.0%} experimental-momentum floor")


if __name__ == "__main__":
    main()
