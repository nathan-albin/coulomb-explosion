#!/usr/bin/env python3
"""Plot the explosion step-count dispersion study.

Reads the two CSVs emitted by ``bench/coulomb_explosion`` and renders a
two-panel figure for the 0002 benchmark report:

  * left  -- distribution of accepted RK45 steps per simulation,
  * right -- shared-dt lockstep efficiency vs. lane count K, against the
             refill/wavefront ceiling (1.0).

Usage:
  python plot_explosion.py \
      --per-sim bench_explosion_per_sim.csv \
      --eff bench_explosion_efficiency.csv \
      --out docs/benchmarks/0002-explosion-dispersion.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-sim", default="bench_explosion_per_sim.csv")
    p.add_argument("--eff", default="bench_explosion_efficiency.csv")
    p.add_argument("--out", default="docs/benchmarks/0002-explosion-dispersion.png")
    args = p.parse_args()

    per_sim = pd.read_csv(args.per_sim)
    eff = pd.read_csv(args.eff).sort_values("lanes")

    fig, (ax_hist, ax_eff) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Left: step-count distribution ------------------------------------
    steps = per_sim["steps"]
    ax_hist.hist(steps, bins=range(int(steps.min()), int(steps.max()) + 2),
                 color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax_hist.axvline(steps.mean(), color="#C44E52", linestyle="--", linewidth=1.5,
                    label=f"mean {steps.mean():.0f}  (cv {steps.std() / steps.mean():.2f})")
    ax_hist.set_xlabel("accepted RK45 steps per simulation")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"Step-count distribution (N = {len(per_sim):,} sims)")
    ax_hist.legend(frameon=False)

    # --- Right: efficiency vs. lane count ---------------------------------
    ax_eff.axhline(1.0, color="#55A868", linestyle="--", linewidth=1.5,
                   label="refill / wavefront ceiling")
    ax_eff.errorbar(eff["lanes"], eff["eff_mean"], yerr=eff["eff_sd"],
                    marker="o", capsize=3, color="#4C72B0", label="shared-dt lockstep")
    for _, row in eff.iterrows():
        ax_eff.annotate(f"{row['eff_mean']:.0%}",
                        (row["lanes"], row["eff_mean"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax_eff.set_xscale("log", base=2)
    ax_eff.set_xticks(eff["lanes"])
    ax_eff.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_eff.set_ylim(0.0, 1.05)
    ax_eff.set_xlabel("SIMD lanes K (sims per batch)")
    ax_eff.set_ylabel("efficiency  (speedup / K)")
    ax_eff.set_title("Shared-dt lockstep efficiency")
    ax_eff.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
