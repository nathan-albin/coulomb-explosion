# Coulomb Explosion

A C++ simulator for Coulomb explosions of small molecules (2–10 atoms), built to
generate large training sets for neural networks that invert asymptotic momentum
measurements back into initial molecular configurations.

## The physics model

The simulator begins from a hypothetical state in which the atoms within a molecule are at rest and have been instantaneously ionized, with the $i\text{th}$ atom receiving a charge of $q_i$. The simulator assumes a purely Coulombic potential energy surface, resulting in the following ODE system of point charges:
$$
m_i  \frac{d^2 \mathbf{r}_i}{dt^2}= \sum_{j\neq i}  \frac{q_i q_j}{|\mathbf{r}_i- \mathbf{r}_j |^3}(\mathbf{r}_i -\mathbf{r}_j).
$$
Here, $m_i$ is the mass of the $i\text{th}$ atom and $\mathbf{r}_i$ is its time-dependent position vector. The equations are written in atomic mass units so that the Coulomb constant is equal to 1.

During the simulation, the ions will force each other apart, eventually settling into an asymptotic regime in which the time-dependent momentum vectors, $\mathbf{p}_i$ are very close to their limit $\lim\limits_{t\to\infty}\mathbf{p}_i(t)$.

## Overall goal

The overall goal is to make the simulation code high performance in order to perform millions of simulations efficiently. The engine is built to be vectorized and
cache-aware, with correctness and a scalar baseline established first, then measured, documented optimization.

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

Algorithm choices (integrators, samplers) are pluggable strategies.
Decisions and performance findings are written up in [docs/](docs/). The Python
reference in [python/](python/) is the correctness ground truth.

## License

[MIT](LICENSE).
