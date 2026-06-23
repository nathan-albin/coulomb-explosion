#!/usr/bin/env python3
"""Part B of 0005: f32 steps-per-sim vs RK45 tolerance, against the 1% floor.

0004 established that f32 is accurate at one fp32-safe tolerance point. This
sweep asks the throughput question: how few steps can we get away with? Looser
``rtol`` ⇒ larger adaptive steps ⇒ fewer steps per sim (the multiplicand that,
times Part A's per-step lane speedup, projects realized throughput). The catch
is the 1% experimental-momentum floor — so for each ``rtol`` we report both the
median step count (the win) and the f32-vs-f64 momentum discrepancy (the cost),
and pick the loosest tolerance that still clears the floor.

Inputs are the CSVs emitted by ``bench/coulomb_precision_sweep``: one f64
reference (tight tolerances, the momentum truth) and one f32 run per ``rtol``,
all on the *same* geometry file. The momentum metric is the config-norm
``|dP|/|P|`` from plot_precision.py, reused verbatim so Part B and 0004 agree.

Usage:
  python plot_steps_vs_rtol.py \
      --f64 /tmp/prec_n10_f64.csv \
      --f32 /tmp/prec_n10_f32_rtol1e-3.csv /tmp/prec_n10_f32_rtol1e-4.csv \
            /tmp/prec_n10_f32_rtol1e-5.csv \
      --summary-csv docs/benchmarks/0005-f32-throughput.csv \
      --out docs/benchmarks/0005-f32-throughput.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_precision import FLOOR, momentum_cols


def read_header(path: str) -> dict:
    """Parse the harness's ``# key=val ...`` comment line into a dict."""
    with open(path) as fh:
        line = fh.readline().lstrip("#").strip()
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def analyze(f64_path: str, f32_path: str) -> dict:
    meta = read_header(f32_path)
    a = pd.read_csv(f64_path, comment="#").set_index("idx")
    b = pd.read_csv(f32_path, comment="#").set_index("idx")
    joined = a.join(b, lsuffix="_64", rsuffix="_32", how="inner")

    n_total = len(joined)
    n_fail = int((joined["failure_32"] != 0).sum())

    ok = joined[(joined["failure_64"] == 0) & (joined["failure_32"] == 0)]
    cols = momentum_cols(a)
    p64 = ok[[c + "_64" for c in cols]].to_numpy().reshape(len(ok), -1, 3)
    p32 = ok[[c + "_32" for c in cols]].to_numpy().reshape(len(ok), -1, 3)

    # Config-norm relative error ||dP|| / ||P|| — identical to plot_precision.py.
    dp = (p32 - p64).reshape(len(ok), -1)
    config_rel = np.linalg.norm(dp, axis=1) / np.linalg.norm(p64.reshape(len(ok), -1), axis=1)

    # Step count is the throughput multiplicand; use the f32 run's own steps.
    steps = ok["steps_32"].to_numpy()

    def pct(x, q):
        return float(np.percentile(x, q)) if len(x) else float("nan")

    return {
        "rtol": float(meta.get("rtol", "nan")),
        "atol": float(meta.get("atol", "nan")),
        "pe_stop": float(meta.get("pe_stop", "nan")),
        "sims": n_total,
        "fail_pct": 100.0 * n_fail / n_total if n_total else float("nan"),
        "steps_median": float(np.median(steps)),
        "steps_p99": pct(steps, 99),
        "dP_p50": pct(config_rel, 50),
        "dP_p99": pct(config_rel, 99),
        "dP_max": pct(config_rel, 100),
        "over_floor_pct": (
            100.0 * int((config_rel > FLOOR).sum()) / len(ok) if len(ok) else float("nan")
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--f64", required=True, help="f64 reference CSV (momentum truth)")
    p.add_argument("--f32", required=True, nargs="+", help="f32 CSVs, one per rtol")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--out", default="docs/benchmarks/0005-f32-throughput.png")
    args = p.parse_args()

    rows = sorted((analyze(args.f64, f) for f in args.f32), key=lambda r: r["rtol"], reverse=True)

    # --- stdout table --------------------------------------------------------
    hdr = ["rtol", "atol", "pe_stop", "fail%", "med steps", "|dP|/|P| p50", "p99", "max", ">1%"]
    print(f"floor = {FLOOR:.0e}   N=10, f32 vs f64 on identical geometries")
    print("  ".join(f"{h:>12}" for h in hdr))
    for r in rows:
        print(
            "  ".join(
                f"{v:>12}"
                for v in (
                    f"{r['rtol']:.0e}",
                    f"{r['atol']:.0e}",
                    f"{r['pe_stop']:.0e}",
                    f"{r['fail_pct']:.2f}",
                    f"{r['steps_median']:.0f}",
                    f"{r['dP_p50']:.2e}",
                    f"{r['dP_p99']:.2e}",
                    f"{r['dP_max']:.2e}",
                    f"{r['over_floor_pct']:.2f}",
                )
            )
        )

    if args.summary_csv:
        pd.DataFrame(rows).to_csv(args.summary_csv, index=False)

    # --- figure: steps (the win) and max |dP|/|P| (the cost) vs rtol ---------
    rtols = [r["rtol"] for r in rows]
    fig, ax_steps = plt.subplots(figsize=(7, 4.4))

    ax_steps.plot(rtols, [r["steps_median"] for r in rows], "o-", color="C0", label="median steps")
    ax_steps.set_xscale("log")
    # Natural log order puts the larger (looser) rtol on the right — the
    # throughput-win direction — so no axis inversion is needed.
    ax_steps.set_xlabel("RK45 rtol (looser →)")
    ax_steps.set_ylabel("median steps / sim", color="C0")
    ax_steps.tick_params(axis="y", labelcolor="C0")

    ax_err = ax_steps.twinx()
    ax_err.plot(rtols, [r["dP_max"] for r in rows], "s--", color="C3", label="max |dP|/|P|")
    ax_err.plot(rtols, [r["dP_p50"] for r in rows], "^:", color="C1", label="median |dP|/|P|")
    ax_err.axhline(FLOOR, color="r", ls="--", lw=1, label="1% floor")
    ax_err.set_yscale("log")
    ax_err.set_ylabel("f32 vs f64  |dP|/|P|", color="C3")
    ax_err.tick_params(axis="y", labelcolor="C3")

    lines = ax_steps.get_lines() + ax_err.get_lines()
    ax_steps.legend(lines, [ln.get_label() for ln in lines], loc="center left", fontsize=8)
    ax_steps.set_title("f32 steps-per-sim vs tolerance, against the 1% floor (N=10)")

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nfigure written to {args.out}")


if __name__ == "__main__":
    main()
