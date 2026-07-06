#!/usr/bin/env python3
"""0006 Part B figure: realized batched-integrator throughput and lockstep efficiency.

Reads the CSV emitted by ``bench/coulomb_batched_integrator`` and draws two panels:

  * Left — realized speedup of the f32 production path (K=16, rtol 1e-4) against
    two baselines: the actual current generator (f64 scalar default) and the
    apples-to-apples f64 *batched* default that 0005's per-step projection
    implies, with 0005's ~16x projection as a reference line.
  * Right — realized lockstep efficiency vs 0002's step-count ceiling, by lane
    count. Lane counts (K) are read from the CSV, not hardcoded, since they
    depend on the host ISA (e.g. K=4/8 f64/f32 on AVX2, K=8/16 on AVX-512) —
    this script is re-run as-is on each new architecture. (The f32 production
    number is at the fp32-safe tolerance; f32 at the sub-eps f64 default
    thrashes — see 0004 — so there is no tight-tol f32 point to compare against
    0002's tight-tol estimate.)

Usage:
  python plot_batched_integrator.py --csv docs/benchmarks/0006-batched-integrator.csv \
      --out docs/benchmarks/0006-batched-integrator.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

PROJECTION_0005 = 16.0  # 0005's projected f32-batched / f64-batched sims/sec.
# 0002's shared-dt efficiency estimate by lane count K (ISA-independent — it is
# a step-count-dispersion model, not an arithmetic one).
CEILING_0002 = {4: 0.71, 8: 0.63, 16: 0.57}


def get(df: pd.DataFrame, row: str, prec: str) -> pd.Series:
    m = df[(df["row"] == row) & (df["precision"] == prec)]
    if len(m) != 1:
        raise SystemExit(f"expected exactly one {row}/{prec} row, got {len(m)}")
    return m.iloc[0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="docs/benchmarks/0006-batched-integrator.png")
    args = p.parse_args()

    df = pd.read_csv(args.csv, comment="#")
    f32_prod = get(df, "batched_prod", "f32")
    sc_def = get(df, "scalar_default", "f64")
    bd_def = get(df, "batched_default", "f64")

    vs_scalar = f32_prod["sims_per_sec"] / sc_def["sims_per_sec"]
    vs_batched = f32_prod["sims_per_sec"] / bd_def["sims_per_sec"]

    fig, (ax_sp, ax_eff) = plt.subplots(1, 2, figsize=(11, 4.4))

    # --- realized speedup ----------------------------------------------------
    labels = ["vs f64 scalar\n(current generator)", "vs f64 batched\n(0005 baseline)"]
    vals = [vs_scalar, vs_batched]
    bars = ax_sp.bar(labels, vals, color=["C0", "C2"], width=0.6)
    ax_sp.axhline(PROJECTION_0005, color="r", ls="--", lw=1,
                  label=f"0005 projection ({PROJECTION_0005:.0f}x)")
    for b, v in zip(bars, vals):
        ax_sp.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}x", ha="center",
                   fontweight="bold")
    k_f64, k_f32 = int(bd_def["lanes"]), int(f32_prod["lanes"])
    ax_sp.set_ylabel("realized speedup (sims/sec)")
    ax_sp.set_title(f"f32 batched lockstep integrator, realized speedup (N=10, K={k_f32})")
    ax_sp.set_ylim(0, max(vals) * 1.18)
    ax_sp.legend(loc="upper center")

    # --- lockstep efficiency vs 0002 ceiling ---------------------------------
    # Realized: f64 default at its native lane count (matched tol to 0002); f32
    # prod at its native lane count (the only fp32-safe point).
    eff_f64 = float(bd_def["efficiency"])
    eff_f32 = float(f32_prod["efficiency"])
    ks = [k_f64, k_f32]
    est = [CEILING_0002[k_f64], CEILING_0002[k_f32]]
    realized = [eff_f64, eff_f32]
    x = range(len(ks))
    ax_eff.bar([i - 0.2 for i in x], est, width=0.4, color="0.6", label="0002 estimate")
    ax_eff.bar([i + 0.2 for i in x], realized, width=0.4, color="C1", label="realized")
    for i, v in zip(x, est):
        ax_eff.text(i - 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in zip(x, realized):
        ax_eff.text(i + 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax_eff.set_xticks(list(x))
    ax_eff.set_xticklabels([f"K={k}" for k in ks])
    ax_eff.set_ylabel("lockstep efficiency (effective lanes / K)")
    ax_eff.set_ylim(0, 1.0)
    ax_eff.set_title("Realized lockstep efficiency vs 0002 ceiling")
    ax_eff.legend(loc="upper right")
    ax_eff.text(1.0, 0.06, f"K={k_f32} at fp32-safe tol\n(f64 default tol thrashes)", ha="center",
                fontsize=7, style="italic", color="0.3")

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"realized vs scalar = {vs_scalar:.2f}x, vs batched = {vs_batched:.2f}x")
    print(f"efficiency K={k_f64} = {eff_f64:.3f}, K={k_f32} = {eff_f32:.3f}")
    print(f"figure written to {args.out}")


if __name__ == "__main__":
    main()
