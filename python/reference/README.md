# Python reference oracle

This is the trusted physics implementation that the C++ engine is validated
against. It originated as quick research code for generating Coulomb-explosion
datasets; everything except the physics core has been removed, since the C++
engine owns sampling, output, and the data pipeline now.

## What's here

- `coulomb.py` — the oracle. Pairwise Coulomb forces and an adaptive RK45
  (`scipy.solve_ivp`) integration to the asymptotic state, in **atomic units**
  (Coulomb constant = 1; masses converted from amu to electron masses via
  `convert_masses`, ×1822.89).
- `config.yaml` — the molecular definition and sampler parameters (the
  pentatomic Br/Cl/F/C/H system) that the C++ side mirrors. Kept as a spec; no
  code here reads it.

## Setup

```sh
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Generating reference cases for the C++ tests

The C++ tests don't call Python. They read a checked-in JSON fixture produced
from this oracle by the adapter in `../../tests/reference/`:

```sh
python ../../tests/reference/gen_reference_cases.py
```

This converts the oracle's velocity output to **momentum** (`p = m·v`) — the
quantity the C++ engine tracks — and writes
`../../tests/reference/reference_cases.json`. Regenerate and commit that file
whenever the oracle physics changes.
