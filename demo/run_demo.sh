#!/usr/bin/env bash
#
# One-command demo: build the batched SIMD engine, generate a million CH4
# Coulomb explosions, and plot the result.
#
#   demo/run_demo.sh [N_SIMS]
#
# N_SIMS defaults to 1,000,000 (~120 MB of scratch, a few seconds on a laptop).
# Everything it produces lands in build/demo-scratch/ (raw dataset, gitignored)
# and demo/ch4_demo.png (the figure). Nothing leaves the repo.
set -euo pipefail

N_SIMS="${1:-1000000}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRATCH="build/demo-scratch"
BIN="$SCRATCH/ch4.bin"
OUT_PNG="demo/ch4_demo.png"
GEN="build/relwithdebinfo/examples/coulomb_generate_dataset"

echo ">>> [1/4] Building the dataset generator (relwithdebinfo, examples on)"
cmake --preset relwithdebinfo -DCOULOMB_BUILD_EXAMPLES=ON >/dev/null
cmake --build --preset relwithdebinfo --target coulomb_generate_dataset >/dev/null

echo ">>> [2/4] Generating $N_SIMS CH4 explosions -> $BIN"
mkdir -p "$SCRATCH"
"$GEN" --out "$BIN" --sims "$N_SIMS"

echo ">>> [3/4] Setting up a Python venv for plotting (demo/.venv)"
if [ ! -d demo/.venv ]; then
  python3 -m venv demo/.venv
fi
# shellcheck disable=SC1091
. demo/.venv/bin/activate
pip install --quiet --disable-pip-version-check -r demo/requirements.txt

echo ">>> [4/4] Plotting -> $OUT_PNG"
# A short CPU label for the figure subtitle (best-effort; Linux /proc/cpuinfo).
CPU="$(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -1 \
        | sed 's/(R)//g; s/(TM)//g; s/  */ /g; s/ @.*//')"
python demo/plot_demo.py --bin "$BIN" --out "$OUT_PNG" --machine "$CPU"

echo
echo "Done. Open $OUT_PNG"
echo "Optional: validate the fast f32 dataset against the fp64 oracle with"
echo "  python examples/verify_subset.py --bin $BIN --n 500"
