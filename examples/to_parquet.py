#!/usr/bin/env python3
"""Convert coulomb_generate_dataset's raw binary output to Parquet.

Reads the sidecar `<bin>.meta.json` for the schema (atom count, molecule,
sampler/tolerance settings) and the fixed-width binary records written by
`examples/generate_dataset.cpp`:

    n*3 float32   initial positions (atom order per meta["symbols"])
    n*3 float32   final momenta (m*v, post energy-redistribution)
    uint32        accepted SIMD-iterations (batch steps) at convergence
    uint8         converged (1) or hit max_steps (0)

Output columns are flat and index-based (not symbol-based, since CH4 repeats
"H") to avoid name collisions: x0,y0,z0,...,x{n-1},y{n-1},z{n-1} for initial
positions, px0,py0,pz0,...,px{n-1},py{n-1},pz{n-1} for final momenta, plus
steps and converged. The molecule/sampler/tolerance settings from the sidecar
JSON are attached as Parquet schema metadata.

Usage:
  python to_parquet.py --bin dataset.bin --out dataset.parquet
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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
    p.add_argument("--bin", required=True, help="path to the raw binary from coulomb_generate_dataset")
    p.add_argument("--out", required=True, help="output .parquet path")
    p.add_argument("--row-group-size", type=int, default=250_000)
    args = p.parse_args()

    with open(args.bin + ".meta.json") as f:
        meta = json.load(f)

    n = meta["n_atoms"]
    n_sims = meta["n_sims"]
    dtype = record_dtype(n)
    arr = np.memmap(args.bin, dtype=dtype, mode="r")
    if len(arr) != n_sims:
        print(f"warning: metadata says {n_sims} records, file has {len(arr)}", file=sys.stderr)

    columns: dict[str, np.ndarray] = {}
    for i in range(n):
        for c, axis in enumerate("xyz"):
            columns[f"{axis}{i}"] = np.ascontiguousarray(arr["init_pos"][:, i * 3 + c])
    for i in range(n):
        for c, axis in enumerate(("px", "py", "pz")):
            columns[f"{axis}{i}"] = np.ascontiguousarray(arr["final_p"][:, i * 3 + c])
    columns["steps"] = np.ascontiguousarray(arr["steps"])
    columns["converged"] = np.ascontiguousarray(arr["converged"]).astype(bool)

    table = pa.table(columns)
    table = table.replace_schema_metadata(
        {
            "symbols": json.dumps(meta["symbols"]),
            "masses_amu": json.dumps(meta["masses_amu"]),
            "charges": json.dumps(meta["charges"]),
            "seed": str(meta["seed"]),
            "radius": str(meta["radius"]),
            "min_separation": str(meta["min_separation"]),
            "rtol": str(meta["rtol"]),
            "atol": str(meta["atol"]),
            "pe_stop": str(meta["pe_stop"]),
        }
    )
    pq.write_table(table, args.out, compression="snappy", row_group_size=args.row_group_size)
    print(f"wrote {args.out}: {table.num_rows} rows, {len(table.column_names)} columns")


if __name__ == "__main__":
    main()
