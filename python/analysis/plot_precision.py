#!/usr/bin/env python3
"""Diff f32 vs f64 precision-sweep runs and summarize the robustness picture.

For each molecule size it reads the two CSVs emitted by
``bench/coulomb_precision_sweep`` (one f64 build, one f32 build, run on the
*same* geometry file) and reports, against the ~1% experimental momentum floor:

  * the f32 failure rate (NaN/Inf, step rejected to death, or step-cap hit),
  * the asymptotic-momentum discrepancy distribution |dP|/|P| (config-norm and
    per-atom worst), p50 / p99 / max,
  * the systematic bias (mean signed component error), and
  * how the discrepancy concentrates against min_init_sep (the close-encounter
    difficulty axis from 0002).

Usage:
  python plot_precision.py --sizes 3,5,8,10 \
      --f64 '/tmp/prec_n{n}_f64.csv' --f32 '/tmp/prec_n{n}_f32.csv' \
      --summary-csv /tmp/prec_summary.csv \
      --out docs/benchmarks/0004-precision-sweep.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FLOOR = 1e-2  # ~1% experimental momentum resolution: the bar f32 must clear.


def momentum_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("p") and c[1:2].isdigit()]


def analyze_size(n: int, f64_path: str, f32_path: str) -> dict:
    a = pd.read_csv(f64_path, comment="#").set_index("idx")
    b = pd.read_csv(f32_path, comment="#").set_index("idx")
    joined = a.join(b, lsuffix="_64", rsuffix="_32", how="inner")

    n_total = len(joined)
    fail_mask = joined["failure_32"] != 0
    n_fail = int(fail_mask.sum())

    ok = joined[(joined["failure_64"] == 0) & (joined["failure_32"] == 0)]
    cols = momentum_cols(a)
    p64 = ok[[c + "_64" for c in cols]].to_numpy().reshape(len(ok), -1, 3)
    p32 = ok[[c + "_32" for c in cols]].to_numpy().reshape(len(ok), -1, 3)

    dp = p32 - p64
    # Config-norm relative error: ||dP|| / ||P|| over the whole momentum vector.
    # Decision-relevant because reconstruction uses all fragments jointly and
    # this does not blow up on near-zero (heavy, slow) atoms.
    config_rel = np.linalg.norm(dp.reshape(len(ok), -1), axis=1) / np.linalg.norm(
        p64.reshape(len(ok), -1), axis=1
    )
    # Per-atom worst relative error (guarded against tiny |p|).
    atom_p64 = np.linalg.norm(p64, axis=2)
    atom_dp = np.linalg.norm(dp, axis=2)
    guard = atom_p64 > (1e-6 * atom_p64.max())
    atom_rel = np.where(guard, atom_dp / np.where(guard, atom_p64, 1.0), 0.0)
    peratom_max = atom_rel.max(axis=1)
    # Systematic bias: mean signed component error, normalized by the typical
    # momentum magnitude. Nonzero => f32 shifts labels (learnable artifact);
    # zero-mean noise is far more benign.
    bias = float(dp.reshape(-1, 3).mean() / np.linalg.norm(p64, axis=2).mean())

    sep = ok["min_init_sep_64"].to_numpy()
    over = config_rel > FLOOR
    # Is the over-floor population confined to the close-encounter tail?
    sep_over = sep[over]

    def pct(x, q):
        return float(np.percentile(x, q)) if len(x) else float("nan")

    return {
        "n": n,
        "total": n_total,
        "fail_pct": 100.0 * n_fail / n_total if n_total else float("nan"),
        "config_p50": pct(config_rel, 50),
        "config_p99": pct(config_rel, 99),
        "config_max": pct(config_rel, 100),
        "peratom_p99": pct(peratom_max, 99),
        "bias": bias,
        "over_floor_pct": 100.0 * int(over.sum()) / len(ok) if len(ok) else float("nan"),
        "over_sep_max": float(sep_over.max()) if len(sep_over) else float("nan"),
        "_config_rel": config_rel,
        "_sep": sep,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", required=True, help="comma-separated atom counts, e.g. 3,5,8,10")
    p.add_argument("--f64", required=True, help="f64 CSV path template with {n}")
    p.add_argument("--f32", required=True, help="f32 CSV path template with {n}")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--out", default="docs/benchmarks/0004-precision-sweep.png")
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    results = [analyze_size(n, args.f64.format(n=n), args.f32.format(n=n)) for n in sizes]

    # --- stdout table --------------------------------------------------------
    hdr = ["N", "fail%", "|dP|/|P| p50", "p99", "max", "bias", ">1% frac", ">1% sep<="]
    print(f"floor = {FLOOR:.0e}  (config-norm |dP|/|P| vs the 1% experimental floor)")
    print("  ".join(f"{h:>12}" for h in hdr))
    for r in results:
        print(
            "  ".join(
                f"{v:>12}"
                for v in (
                    r["n"],
                    f"{r['fail_pct']:.2f}",
                    f"{r['config_p50']:.2e}",
                    f"{r['config_p99']:.2e}",
                    f"{r['config_max']:.2e}",
                    f"{r['bias']:+.2e}",
                    f"{r['over_floor_pct']:.2f}",
                    f"{r['over_sep_max']:.2f}",
                )
            )
        )

    if args.summary_csv:
        pd.DataFrame(
            [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        ).to_csv(args.summary_csv, index=False)

    # --- figure --------------------------------------------------------------
    fig, (ax_n, ax_sep) = plt.subplots(1, 2, figsize=(11, 4.2))

    ns = [r["n"] for r in results]
    ax_n.plot(ns, [r["config_p50"] for r in results], "o-", label="median")
    ax_n.plot(ns, [r["config_p99"] for r in results], "s-", label="p99")
    ax_n.plot(ns, [r["config_max"] for r in results], "^-", label="max")
    ax_n.axhline(FLOOR, color="r", ls="--", label="1% floor")
    ax_n.set_yscale("log")
    ax_n.set_xlabel("molecule size N (atoms)")
    ax_n.set_ylabel("f32 vs f64  |dP|/|P|")
    ax_n.set_title("Momentum discrepancy vs molecule size")
    ax_n.legend()

    for r in results:
        ax_sep.scatter(r["_sep"], r["_config_rel"], s=4, alpha=0.3, label=f"N={r['n']}")
    ax_sep.axhline(FLOOR, color="r", ls="--", label="1% floor")
    ax_sep.set_yscale("log")
    ax_sep.set_xlabel("min initial separation (a.u.)")
    ax_sep.set_ylabel("f32 vs f64  |dP|/|P|")
    ax_sep.set_title("Discrepancy vs close-encounter tightness")
    ax_sep.legend(markerscale=3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nfigure written to {args.out}")


if __name__ == "__main__":
    main()
