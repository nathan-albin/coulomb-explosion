#!/usr/bin/env python3
"""Physics sanity checks on a generated Coulomb-explosion dataset.

These are *internal-consistency* checks: physical laws the output must obey
regardless of any experiment, plus one absolute-scale anchor. They complement
`examples/verify_subset.py` (which checks the fast f32 result against the fp64
`scipy.solve_ivp` oracle) — this script needs no oracle and runs in a second.

Checks:
  1. Energy conservation   -- final KER must equal the initial Coulomb PE
     (the run starts at rest and redistributes residual PE into KE, so
     E_final = E_0 = PE_0 exactly). This also means the KER distribution is
     nothing more than the sampler's initial-PE distribution.
  2. Momentum conservation -- total momentum must stay ~0 (started at rest).
  3. Energy partition      -- per-species share of the kinetic energy, compared
     to the closed-form prediction from masses alone: with comparable momentum
     magnitudes, KE = p^2/2m gives species s a share proportional to n_s / m_s.
  4. Absolute scale        -- mean KER in eV; for CH4, compared to the KER a
     real (equilibrium-geometry) methane would give, as an order-of-magnitude
     anchor. NOTE the sampler draws a *synthetic* geometry distribution, so the
     absolute number is a property of the sampler, not of real methane.

Usage:
  python demo/sanity_check.py --bin build/demo-scratch/ch4.bin [--n 200000]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

AMU_TO_ELECTRON_MASS = 1822.888486209
HARTREE_TO_EV = 27.211386245988
BOHR_IN_ANGSTROM = 0.529177210903

# Reference equilibrium geometries for the absolute-scale anchor, keyed by
# formula string. Only what we need; extend as needed.
#   value: callable(charges) -> total Coulomb PE in Hartree at real geometry.
def _ch4_reference_ker_hartree() -> float:
    r_ch = 1.087 / BOHR_IN_ANGSTROM              # experimental C-H bond, Bohr
    r_hh = r_ch * np.sqrt(8.0 / 3.0)             # tetrahedral H-H distance
    return 4 * (1.0 / r_ch) + 6 * (1.0 / r_hh)   # 4 C-H pairs + 6 H-H pairs, unit charges


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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bin", required=True, help="raw binary from coulomb_generate_dataset")
    p.add_argument("--n", type=int, default=200_000,
                   help="records to check (from the start); the pairwise PE check is O(n*atoms^2)")
    args = p.parse_args()

    with open(args.bin + ".meta.json") as f:
        meta = json.load(f)

    symbols = meta["symbols"]
    n_atoms = meta["n_atoms"]
    masses_amu = np.asarray(meta["masses_amu"], dtype=float)
    masses = masses_amu * AMU_TO_ELECTRON_MASS
    charges = np.asarray(meta["charges"], dtype=float)

    arr = np.memmap(args.bin, dtype=record_dtype(n_atoms), mode="r")
    n = min(args.n, len(arr))
    x = arr["init_pos"][:n].reshape(-1, n_atoms, 3).astype(float)
    p_final = arr["final_p"][:n].reshape(-1, n_atoms, 3).astype(float)
    converged = arr["converged"][:n].astype(bool)

    ke_atom = (p_final**2).sum(axis=2) / (2.0 * masses)  # (n, atoms), Hartree
    ker = ke_atom.sum(axis=1)

    print(f"Dataset: {meta['n_sims']:,} sims of {''.join(symbols)}  "
          f"(checking first {n:,})")
    print(f"  converged: {converged.mean():.4%}\n")

    # (1) Energy conservation: KER vs initial Coulomb PE.
    diff = x[:, :, None, :] - x[:, None, :, :]
    r = np.linalg.norm(diff, axis=3)
    iu = np.triu_indices(n_atoms, 1)
    qq = np.outer(charges, charges)[iu]
    pe = (qq / r[:, iu[0], iu[1]]).sum(axis=1)  # initial PE, Hartree
    rel = np.abs(ker - pe) / pe
    ok1 = rel.max() < 1e-4
    print("(1) Energy conservation  (KER == initial Coulomb PE)")
    print(f"    mean KER = {ker.mean():.4f} Ha  |  mean PE = {pe.mean():.4f} Ha")
    print(f"    |KER-PE|/PE:  mean {rel.mean():.2e}   max {rel.max():.2e}   "
          f"[{'PASS' if ok1 else 'FAIL'}]\n")

    # (2) Momentum conservation: total p ~ 0.
    tot = np.linalg.norm(p_final.sum(axis=1), axis=1)
    typ = np.linalg.norm(p_final, axis=2).mean()
    ratio = (tot / typ).mean()
    ok2 = ratio < 1e-4
    print("(2) Momentum conservation  (|sum p| / typical |p|)")
    print(f"    mean {ratio:.2e}   [{'PASS' if ok2 else 'FAIL'}]\n")

    # (3) Energy partition per species vs the masses-only prediction.
    #     equal-|p| => KE_s ∝ n_s / m_s  =>  share_s = (n_s/m_s) / Σ_k (n_k/m_k).
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(symbols):
        groups.setdefault(s, []).append(i)
    weights = {s: len(idx) / masses_amu[idx[0]] for s, idx in groups.items()}
    wsum = sum(weights.values())
    print("(3) Energy partition per species  (measured vs. equal-|p| prediction)")
    ok3 = True
    for s, idx in groups.items():
        measured = ke_atom[:, idx].sum() / ke_atom.sum()
        predicted = weights[s] / wsum
        ok3 &= abs(measured - predicted) < 0.01
        print(f"    {s} (x{len(idx)}):  measured {measured:6.2%}   predicted {predicted:6.2%}")
    print(f"    [{'PASS' if ok3 else 'FAIL'}]\n")

    # (4) Absolute-scale anchor.
    print("(4) Absolute scale  (order-of-magnitude anchor, not an experiment match)")
    print(f"    sampler mean KER: {ker.mean() * HARTREE_TO_EV:.1f} eV")
    formula = "".join(symbols)
    if formula == "CHHHH":
        ref = _ch4_reference_ker_hartree() * HARTREE_TO_EV
        print(f"    real CH4 equilibrium geometry would give ~{ref:.0f} eV "
              f"(same order; the sampler's looser cloud sits a bit lower)")
    print("    NOTE: the absolute KER reflects the *synthetic* uniform-sphere sampler,")
    print("          not a physical/thermal geometry ensemble — see demo/README.md.\n")

    all_ok = ok1 and ok2 and ok3
    print("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
