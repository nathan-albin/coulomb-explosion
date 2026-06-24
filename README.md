# Coulomb Explosion

A C++ simulator for Coulomb explosions of small molecules (2–10 atoms), built to
generate large training sets for neural networks that invert asymptotic momentum
measurements back into initial molecular configurations.

## The physics model

The simulator begins from a hypothetical state in which the atoms within a molecule are at rest and have been instantaneously ionized, with the $`i\text{th}`$ atom receiving a charge of $`q_i`$. The simulator assumes a purely Coulombic potential energy surface, resulting in the following ODE system of point charges:

```math
m_i \frac{d^2 \vec{r}_i}{dt^2} = \sum_{j\neq i} \frac{q_i q_j}{|\vec{r}_i - \vec{r}_j|^3}(\vec{r}_i - \vec{r}_j).
```

Here, $`m_i`$ is the mass of the $`i\text{th}`$ atom and $`\vec{r}_i`$ is its time-dependent position vector. The equations are written in atomic mass units so that the Coulomb constant is equal to 1.

During the simulation, the ions will force each other apart, eventually settling into an asymptotic regime in which the time-dependent momentum vectors, $`\vec{p}_i`$ are very close to their limit $`\lim\limits_{t\to\infty}\vec{p}_i(t)`$.

## Overall goal

The overall goal is to make the simulation code high performance in order to perform millions of simulations efficiently. The engine is built to be vectorized and
cache-aware, with correctness and a scalar baseline established first, then measured, documented optimization. A secondary goal is to gain experience guiding Claude Code. I'm using Claude to write the code, reports, and other documentation.

> Status: **working vectorized engine; data-output layer still to come.** In
> place: the scalar O(N²) Coulomb baseline, symplectic velocity-Verlet and
> adaptive RK45 (DP5(4)) integrators, a uniform-sphere sampler, and
> convergence-driven explosion runs. The SIMD-over-lanes path is built and
> measured — a batched force kernel and a batched lockstep integrator realize
> ~27× over the f64 scalar baseline at single precision (see
> [docs/benchmarks/](docs/benchmarks/) and [docs/decisions/](docs/decisions/)).
> Still planned: Parquet dataset output and further integrator optimizations
> (difficulty binning, refill/wavefront, rsqrt). See [docs/](docs/).

## Building

Requires CMake ≥ 3.20, a C++20 compiler, and Ninja. Tests fetch Catch2 at
configure time (needs network on first build).

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
./build/release/apps/coulomb        # run the demo driver
```

Other presets: `debug` (warnings + ASan/UBSan), `relwithdebinfo` (builds the
Google Benchmark microbenchmarks).

```sh
cmake --preset relwithdebinfo && cmake --build --preset relwithdebinfo
./build/relwithdebinfo/bench/coulomb_bench
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

Algorithm choices (integrators, samplers) are pluggable strategies.
Decisions and performance findings are written up in [docs/](docs/). The Python
reference in [python/](python/) is the correctness ground truth.

## License

[MIT](LICENSE).
