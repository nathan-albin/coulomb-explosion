#!/usr/bin/env bash
# Runs once after the Codespace/devcontainer is created. Warms the CMake
# FetchContent cache (Catch2, nlohmann_json) by doing a real configure +
# build + test pass, so the toolchain is verified and the first build a user
# runs by hand is fast.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo ">>> Configuring (release preset)"
cmake --preset release

echo ">>> Building"
cmake --build --preset release

echo ">>> Running tests"
ctest --preset release

cat <<'EOF'

Devcontainer ready.

  demo/run_demo.sh              one-command SIMD demo + figure
  cmake --build --preset release && ctest --preset release
  cmake --preset relwithdebinfo && cmake --build --preset relwithdebinfo   # benchmarks

EOF
