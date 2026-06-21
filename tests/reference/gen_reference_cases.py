#!/usr/bin/env python3
"""Generate reference test cases for the C++ engine from the Python oracle.

This is *not* part of the research pipeline. It imports the physics core from
``python/reference/coulomb.py`` (the trusted oracle) and writes a small set of
deterministic cases to ``reference_cases.json``, which is checked into the repo
and consumed by the C++ tests. CI never runs this script; it only reads the
committed JSON. Regenerate it (and commit the result) whenever the oracle
physics changes.

Usage:
    python tests/reference/gen_reference_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Import the oracle physics directly from the reference implementation.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python" / "reference"))
import coulomb as co  # noqa: E402  (path set up above)

OUTPUT = Path(__file__).resolve().parent / "reference_cases.json"


def potential_energy(x: np.ndarray, charges: np.ndarray) -> float:
    """Total Coulomb potential energy, k = 1 (atomic units)."""
    n = len(charges)
    u = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            r = np.linalg.norm(x[i] - x[j])
            u += charges[i] * charges[j] / r
    return float(u)


def kinetic_energy_from_momenta(p: np.ndarray, masses: np.ndarray) -> float:
    """KE = sum |p_i|^2 / (2 m_i)."""
    return float(0.5 * np.sum(np.sum(p**2, axis=1) / masses))


def make_case(name, symbols, masses_amu, charges, positions):
    """Build one reference case in the C++ data contract.

    The C++ engine tracks position and *momentum* (not velocity). The oracle
    works in velocity, so we convert at this boundary: p = m * v. Initial
    momenta are zero for now (the sampler starts atoms at rest)."""
    masses = co.convert_masses(np.asarray(masses_amu, dtype=float))
    charges = np.asarray(charges, dtype=float)
    x_init = np.asarray(positions, dtype=float)
    v_init = np.zeros_like(x_init)

    forces_init = co.forces(x_init, charges)
    x_final, v_final = co.simulate_explosion(x_init, v_init, masses, charges)

    p_init = masses[:, None] * v_init          # zero for now
    p_final = masses[:, None] * v_final        # the measured asymptotic quantity

    return {
        "name": name,
        "symbols": list(symbols),
        "masses": masses.tolist(),          # electron-mass units (post-convert)
        "charges": charges.tolist(),
        # Initial state (the ML target): positions + momenta.
        "positions_init": x_init.tolist(),
        "momenta_init": p_init.tolist(),
        # Tier 1 — force-kernel validation (testable without an integrator):
        "forces_init": forces_init.tolist(),
        # Energy invariants (p_init = 0, so PE_init should equal KE_final):
        "potential_energy_init": potential_energy(x_init, charges),
        "kinetic_energy_final": kinetic_energy_from_momenta(p_final, masses),
        # Tier 2 — full asymptotic explosion (needs the C++ RK45 port).
        # momenta_final is the ML input feature; positions_final is kept for
        # validating the integrator's trajectory, not part of the dataset.
        "positions_final": x_final.tolist(),
        "momenta_final": p_final.tolist(),
    }


def main() -> None:
    cases = []

    # Two protons on the x-axis, unit separation. Smallest non-trivial check.
    cases.append(make_case(
        "two_protons",
        symbols=["H", "H"],
        masses_amu=[1.0, 1.0],
        charges=[1.0, 1.0],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ))

    # Three protons in an equilateral triangle (side 2 au).
    h = np.sqrt(3.0)
    cases.append(make_case(
        "triangle_H3",
        symbols=["H", "H", "H"],
        masses_amu=[1.0, 1.0, 1.0],
        charges=[1.0, 1.0, 1.0],
        positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, h, 0.0]],
    ))

    # The pentatomic system from config.yaml, with a fixed, well-separated
    # geometry (all pairwise distances > min_radius = 0.5 au). Distinct masses
    # and a genuinely 3D layout exercise the full kernel.
    cases.append(make_case(
        "pentatomic_BrClFCH",
        symbols=["Br", "Cl", "F", "C", "H"],
        masses_amu=[79.0, 35.0, 19.0, 12.0, 1.0],
        charges=[1.0, 1.0, 1.0, 1.0, 1.0],
        positions=[
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 0.0, 2.5],
            [-1.5, -1.5, 1.0],
        ],
    ))

    doc = {
        "description": (
            "Reference cases generated from python/reference/coulomb.py. "
            "Atomic units (Coulomb constant = 1). Masses are in electron-mass "
            "units (amu * 1822.888486209). Do not edit by hand; regenerate with "
            "tests/reference/gen_reference_cases.py."
        ),
        "coulomb_constant": 1.0,
        "rk_rtol": co.RKRTOL,
        "rk_atol": co.RKATOL,
        "cases": cases,
    }

    OUTPUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"Wrote {len(cases)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
