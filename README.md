# Coulomb

A C++ simulator for Coulomb explosions of small molecules (2–10 atoms), built to
generate large training sets for neural networks that invert asymptotic momentum
measurements back into initial molecular configurations.

Performance is a first-class concern: the engine is built to be vectorized and
cache-aware, with correctness and a scalar baseline established first, then
measured, documented optimization.

> Status: **early scaffold.** The engine has a working symplectic integrator and
> a naive O(N²) Coulomb kernel (the baseline). Samplers, an RK45 port, Parquet
> output, and SIMD are planned. See [docs/](docs/).

## Building

Requires CMake ≥ 3.20, a C++20 compiler, and Ninja. Tests fetch Catch2 at
configure time (needs network on first build).

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
./build/release/apps/coulomb        # run the demo driver
```

Other presets: `debug` (warnings + ASan/UBSan), `bench` (builds the Google
Benchmark microbenchmarks).

```sh
cmake --preset bench && cmake --build --preset bench
./build/bench/bench/coulomb_bench
```

## Layout

| Path        | Contents                                              |
|-------------|-------------------------------------------------------|
| `include/`  | Public engine headers (`coulomb/…`)                   |
| `src/`      | Engine implementation (`coulomb::engine` library)     |
| `apps/`     | Thin CLI driver                                       |
| `tests/`    | Catch2 correctness tests                              |
| `bench/`    | Google Benchmark kernels                              |
| `python/`   | Reference implementation + analysis/plotting          |
| `docs/`     | Design decisions and benchmark reports                |

## Design

Algorithm choices (integrators, samplers) are pluggable strategies so
alternatives are real, tested implementations rather than commented-out code.
Decisions and performance findings are written up in [docs/](docs/). The Python
reference in [python/](python/) is the correctness ground truth.

## License

[MIT](LICENSE).
