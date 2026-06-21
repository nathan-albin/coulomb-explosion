# 0001 — Build system and repository layout

- **Status:** accepted
- **Date:** 2026-06-21

## Context

The project is a physics data generator with performance as a first-class goal:
vectorized, cache-aware C++. It must build cleanly, run on at least two hardware
platforms (a laptop CPU and an HPC node).

## Decision

- **Build:** CMake (>= 3.20) with `CMakePresets.json`. Single, unambiguous build
  for anyone cloning the repo.
- **Layout:** the simulation lives in a library (`coulomb::engine`); the CLI,
  tests, and benchmarks are thin clients of it. This keeps the hot code
  independently testable and benchmarkable.

  ```
  src/        engine implementation        include/    public headers
  apps/       CLI driver                    tests/      Catch2 correctness tests
  bench/      Google Benchmark kernels      docs/       this directory
  python/     reference impl + analysis
  ```
- **Toolchain matrix:** GCC and Clang, Debug + Release, in CI. Debug builds run
  with ASan/UBSan.
- **Measurement first:** a naive O(N^2) scalar Coulomb kernel is the correctness
  and performance baseline. Every optimization (blocking, SoA, Highway SIMD) is
  reported as a ratio against it, on named hardware.

## Consequences

- Adding an alternative algorithm means adding a real implementation behind a
  strategy interface (e.g. `Integrator`), not a compile-time `#ifdef` or
  commented-out code.
- Rejected approaches are documented here with numbers and, if their code is
  worth preserving, referenced by git tag rather than left in the tree.

## Template for future decisions

```
# NNNN — Title
- Status: proposed | accepted | superseded by NNNN
- Date: YYYY-MM-DD
## Context
## Decision
## Consequences
```
