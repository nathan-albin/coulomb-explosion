#!/usr/bin/env python3
"""Plot the physics of a generated Coulomb-explosion dataset.

Reads the raw binary + `<bin>.meta.json` sidecar written by
`examples/generate_dataset.cpp` (same fixed-width record layout as
`examples/to_parquet.py`) and renders a two-panel summary figure:

  * left  -- distribution of kinetic-energy release (KER): the total final
             kinetic energy per simulation, the headline observable of a real
             Coulomb-explosion measurement.
  * right -- distribution of final momentum magnitude, split by atomic
             species, showing how the explosion energy partitions across the
             light and heavy fragments.

The figure's subtitle carries the throughput the generator recorded in the
sidecar, so the picture doubles as the performance headline.

Usage:
  python plot_demo.py --bin dataset.bin --out ch4_demo.png [--machine "..."]
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

# amu -> electron masses; matches units::kAmuToElectronMass in the C++ engine
# and convert_masses() in python/reference/coulomb.py.
AMU_TO_ELECTRON_MASS = 1822.888486209
# atomic units of energy (Hartree) -> electron-volts (CODATA).
HARTREE_TO_EV = 27.211386245988

# Chart chrome and a CVD-safe categorical pair, from the dataviz palette
# (validated: worst-adjacent CVD ΔE 96.7). Slot order is fixed, not cycled.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SPECIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]


def record_dtype(n_atoms: int) -> np.dtype:
    return np.dtype(
        [
            ("init_pos", "<f4", (n_atoms * 3,)),
            ("final_p", "<f4", (n_atoms * 3,)),
            ("steps", "<u4"),
            ("converged", "u1"),
        ]
    )


def species_groups(symbols: list[str]) -> list[tuple[str, list[int]]]:
    """Group atom indices by symbol, preserving first-appearance order."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(symbols):
        groups.setdefault(s, []).append(i)
    return list(groups.items())


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bin", required=True, help="raw binary from coulomb_generate_dataset")
    p.add_argument("--out", required=True, help="output .png path")
    p.add_argument("--machine", default="", help="optional CPU label for the subtitle")
    args = p.parse_args()

    with open(args.bin + ".meta.json") as f:
        meta = json.load(f)

    symbols = meta["symbols"]
    n = meta["n_atoms"]
    masses = np.asarray(meta["masses_amu"], dtype=float) * AMU_TO_ELECTRON_MASS

    arr = np.memmap(args.bin, dtype=record_dtype(n), mode="r")
    p_final = arr["final_p"].reshape(-1, n, 3).astype(float)  # (sims, atoms, xyz)

    # Per-atom kinetic energy KE = |p|^2 / 2m (Hartree, since the engine works
    # in atomic units with the Coulomb constant = 1), then per-simulation KER.
    ke_atom = (p_final**2).sum(axis=2) / (2.0 * masses)
    ker_ev = ke_atom.sum(axis=1) * HARTREE_TO_EV
    p_mag = np.linalg.norm(p_final, axis=2)  # (sims, atoms), atomic units

    fig, (ax_ker, ax_p) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    formula = "".join(f"{s}$_{c}$" if c > 1 else s for s, c in _formula(symbols))

    # --- Left: KER distribution -------------------------------------------
    # Clip the long thin high-energy tail so the bulk fills the panel.
    ker_hi = np.percentile(ker_ev, 99.9)
    ker_bins = np.linspace(ker_ev.min(), ker_hi, 80)
    ax_ker.hist(ker_ev, bins=ker_bins, color=SPECIES_COLORS[0], edgecolor="white", linewidth=0.2)
    ax_ker.set_xlim(ker_ev.min(), ker_hi)
    mean_ker = ker_ev.mean()
    ax_ker.axvline(mean_ker, color=INK, linestyle="--", linewidth=1.3)
    ax_ker.annotate(
        f"mean {mean_ker:.0f} eV",
        xy=(mean_ker, 0.94),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK,
        fontsize=10,
        ha="left",
        va="top",
    )
    ax_ker.set_xlabel("kinetic-energy release  (eV)")
    ax_ker.set_ylabel("simulations")
    ax_ker.set_title("Total energy released per explosion", color=INK, fontsize=12)

    # --- Right: momentum magnitude by species -----------------------------
    groups = species_groups(symbols)
    lo = np.percentile(p_mag, 0.5)
    hi = np.percentile(p_mag, 99.5)
    bins = np.linspace(lo, hi, 70)
    ke_share = ke_atom.sum(axis=0)  # summed over sims, per atom
    for k, (sym, idx) in enumerate(groups):
        color = SPECIES_COLORS[k % len(SPECIES_COLORS)]
        share = ke_share[idx].sum() / ke_share.sum()
        label = f"{sym} (×{len(idx)}) – {share:.0%} of energy"
        ax_p.hist(
            p_mag[:, idx].ravel(),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=label,
        )
    ax_p.set_xlabel("final momentum magnitude |p|  (atomic units)")
    ax_p.set_ylabel("probability density")
    ax_p.set_title("Where the momentum goes", color=INK, fontsize=12)
    ax_p.legend(frameon=False, fontsize=10, loc="upper right")

    for ax in (ax_ker, ax_p):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    # Headline: what was run, how fast.
    n_sims = meta["n_sims"]
    rate_m = meta.get("sims_per_sec", 0.0) / 1e6
    wall = meta.get("wall_seconds", 0.0)
    machine = f" on {args.machine}" if args.machine else ""
    fig.suptitle(
        f"{n_sims:,} {formula} Coulomb explosions",
        fontsize=15,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.text(
        0.5,
        0.905,
        f"batched f32 SIMD engine · generated in {wall:.1f} s "
        f"({rate_m:.2f} M simulations/s){machine}",
        ha="center",
        color=MUTED,
        fontsize=10.5,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(args.out, dpi=140, facecolor="white")
    print(f"wrote {args.out}  ({n_sims:,} sims, mean KER {mean_ker:.1f} eV)")


def _formula(symbols: list[str]) -> list[tuple[str, int]]:
    """Condense an atom-symbol list into (symbol, count) formula terms."""
    out: list[tuple[str, int]] = []
    for s in symbols:
        if out and out[-1][0] == s:
            out[-1] = (s, out[-1][1] + 1)
        else:
            out.append((s, 1))
    return out


if __name__ == "__main__":
    main()
