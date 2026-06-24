# How the batched SIMD force kernel works

An annotated walkthrough of [`include/coulomb/batched_force.hpp`](../../include/coulomb/batched_force.hpp).

This header is the SIMD-accelerated heart of the project. It computes Coulomb
accelerations and potential energies, but unlike the scalar baseline in
[`src/system.cpp`](../../src/system.cpp) it does **k independent simulations at
once** — one per SIMD lane. The clever part is *what* gets vectorized: not the
atoms within one molecule, but copies of the *same* molecule in different
geometries.

---

## 1. The big idea: SIMD over simulations, not over atoms

A CPU SIMD register holds several scalars side by side (e.g. an AVX2 register
holds 8 `float`s or 4 `double`s). The hardware applies one instruction to all of
them simultaneously — that's the speedup.

The naive way to vectorize an N-body kernel is to pack *atoms* into a register.
That gets awkward fast: the inner loop has cross-lane reductions (summing a
force vector), the atom count rarely divides the lane count, and Newton's-third-
law force scatter fights the layout.

This kernel takes the other route. The production workload is **many geometries
of one fixed molecule** (e.g. sampling thousands of initial conditions for a
Coulomb explosion). So we run `k` of those geometries together: **lane `l`
holds the `l`-th simulation.** Every lane executes the identical control flow —
same `n`, same loop bounds, same pair `(i, j)` — just with different position
numbers. That is the *perfect* shape for SIMD: zero divergence, zero cross-lane
communication, the lane count `k` is something we choose, and the inner math is
pure element-wise arithmetic.

This is sometimes called a **"batch" or "lockstep"** layout, and it's why the
file lives behind `namespace batched`.

---

## 2. The data layout: structure-of-arrays over lanes

This is the single most important thing to understand. From the header comment:

> Layout (SoA over lanes): for each coordinate, `p[i*k .. i*k+k)` holds atom
> `i`'s value across the `k` sims.

So for the x-coordinate array `px`, with `n` atoms and `k` sims:

```
        atom 0              atom 1              atom 2
px = [ s0 s1 ... s(k-1) | s0 s1 ... s(k-1) | s0 s1 ... s(k-1) | ... ]
       \_____ k _______/   \_____ k _______/
        one SIMD vector     one SIMD vector
```

- `px[i*k + l]` = x-coordinate of atom `i` in simulation `l`.
- The `k` consecutive values for a single atom form **one contiguous SIMD
  vector**. That's why a plain `hn::Load(d, &px[i*k])` pulls in "atom `i`'s x
  across all sims" with no shuffling.
- Same scheme for `py`, `pz` (positions) and `ax`, `ay`, `az` (outputs). All are
  length `n*k` and **vector-aligned** (allocated via `hwy::AllocateAligned`),
  which lets the code use aligned `Load`/`Store`.

The per-atom *scalars* are broadcast, not batched, because the molecule is the
same across all sims:

- `pair_const` — length `n*n`, where `pair_const[i*n + j] = k * q_i * q_j`
  (the Coulomb constant times the two charges). Precomputed and narrowed to `T`.
  Indexed `[i*n + j]`, a row-major `n×n` table; only the `i<j` upper triangle is
  read.
- `inv_mass` — length `n`, `inv_mass[i] = 1/m_i`.

Because charges and masses are identical in every sim, these enter the math via
`hn::Set` (broadcast one scalar into every lane) rather than `hn::Load`.

> **`k` must equal the hardware lane count** for the target type `T`, i.e.
> `k == hn::Lanes(hn::ScalableTag<T>())`. Every `Load`/`Store` in the kernel
> touches exactly one vector's worth of elements (`k`), so the layout only lines
> up when `k` is that width. The call sites enforce this:
> - the benchmark sizes the batch with `const std::size_t k = hn::Lanes(d);`
>   ([`bench/bench_force_batch.cpp`](../../bench/bench_force_batch.cpp));
> - the batched integrator checks it in its constructor and **throws** otherwise:
>   `if (k_ != hn::Lanes(hn::ScalableTag<T>())) throw std::invalid_argument(...)`
>   ([`include/coulomb/batched_integrator.hpp`](../../include/coulomb/batched_integrator.hpp)).
>
> So `k` is not a free "how many sims do you want" knob — it's fixed by the ISA
> and element type (e.g. AVX2 → 4 for `double`, 8 for `float`). To run more sims
> than one vector holds, you process multiple batches.

---

## 3. Highway and static dispatch

```cpp
#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;
```

[Highway](https://github.com/google/highway) is Google's portable SIMD library.
You write against generic ops (`hn::Load`, `hn::Mul`, `hn::Sqrt`, …) and it maps
them to whatever the target ISA provides (SSE, AVX2, AVX-512, NEON, SVE, …).

Two Highway concepts appear here:

- **`hn::ScalableTag<T> d;`** — a zero-size tag describing "a full-width vector
  of `T` on this target." It's passed to nearly every op so Highway knows the
  element type and width. `T` is `float` or `double`; the lane count follows from
  the target register width. The variable `d` carries no data — it's compile-time
  type information.
- **`HWY_NAMESPACE` = static dispatch.** Highway can do runtime dispatch
  (detect the CPU and pick a code path), but this header uses the namespace
  directly, meaning the code is compiled **once for the host ISA**.

The header comment spells out the consequence, and it's a real constraint, not a
footnote:

> every TU that includes it MUST be compiled for the host ISA
> (`-march=native`), and it must NOT be linked into the generic engine library,
> which stays SSE2-baseline so 0001's scalar floor remains the conservative
> reference.

In other words: this fast code is deliberately **quarantined** to the benchmark
([`bench/bench_force_batch.cpp`](../../bench/bench_force_batch.cpp)) and the batched
integrator (report 0006). The general-purpose engine library stays at a portable
SSE2 baseline so the scalar reference numbers in
[decision 0001](../decisions/0001-build-and-layout.md) remain an honest, conservative
floor. Mixing `-march=native` objects into that library would silently change the
baseline.

---

## 4. `accelerations()` line by line

Goal (per lane):

$$a_i = \frac{1}{m_i} \sum_{j \ne i} k\, q_i q_j \, \frac{(x_i - x_j)}{|r_{ij}|^3}$$

It mirrors `CoulombForce::accelerations` in
[`src/system.cpp`](../../src/system.cpp#L5) **lane-for-lane** — same operation order,
and crucially `1/sqrt` (`hn::Div(one, hn::Sqrt(...))`) rather than the faster
approximate `rsqrt`, so the batched results match the scalar baseline bit-for-bit
in spirit rather than trading accuracy for speed.

### 4a. Zero the accumulators

```cpp
for (std::size_t i = 0; i < n; ++i) {
  hn::Store(hn::Zero(d), d, &ax[i * k]);
  hn::Store(hn::Zero(d), d, &ay[i * k]);
  hn::Store(hn::Zero(d), d, &az[i * k]);
}
```

`hn::Zero(d)` is a vector of zeros; this clears the whole `n*k` output for all
three components, all lanes at once. Needed because the main loop *accumulates*
into these arrays (both the `i` side and the scattered `j` side).

### 4b. The pair loop with Newton's third law

```cpp
for (std::size_t i = 0; i < n; ++i) {
  const auto pxi = hn::Load(d, &px[i * k]);   // atom i's x across all sims
  const auto pyi = hn::Load(d, &py[i * k]);
  const auto pzi = hn::Load(d, &pz[i * k]);
  auto axi = hn::Load(d, &ax[i * k]);          // running accumulator for i
  auto ayi = hn::Load(d, &ay[i * k]);
  auto azi = hn::Load(d, &az[i * k]);
```

`pxi/pyi/pzi` are atom `i`'s coordinates held in registers for the whole inner
loop. `axi/ayi/azi` are atom `i`'s force accumulators kept **in registers**
across all `j` — that's the key optimization in this loop: atom `i`'s
contributions are summed register-side and written back only once, after the
inner loop. Only atom `j`'s side is read-modify-written to memory each iteration.

```cpp
  for (std::size_t j = i + 1; j < n; ++j) {
    const auto rx = hn::Sub(pxi, hn::Load(d, &px[j * k]));   // r = x_i - x_j
    const auto ry = hn::Sub(pyi, hn::Load(d, &py[j * k]));
    const auto rz = hn::Sub(pzi, hn::Load(d, &pz[j * k]));
```

Only the upper triangle `j = i+1 .. n-1` is visited; each unordered pair is
handled once and the equal/opposite force is applied to both atoms (Newton's
third law). `rx/ry/rz` is the separation vector, computed for all `k` sims
simultaneously.

```cpp
    const auto dist2 = hn::MulAdd(rx, rx, hn::MulAdd(ry, ry, hn::Mul(rz, rz)));
```

`dist2 = rx² + ry² + rz² = |r|²`. `hn::MulAdd(a, b, c)` is a fused
multiply-add `a*b + c` — one instruction, and on FMA hardware it's both faster
and (incidentally) more accurate than separate multiply + add. Read inside-out:
`rz*rz`, then `+ ry*ry`, then `+ rx*rx`.

```cpp
    const auto inv_dist  = hn::Div(one, hn::Sqrt(dist2));   // 1/|r|
    const auto inv_dist3 = hn::Div(inv_dist, dist2);        // 1/|r|^3
    const auto scale = hn::Mul(hn::Set(d, pair_const[i * n + j]), inv_dist3);
```

- `inv_dist = 1/√(dist2)` = `1/|r|`. Note the explicit `Div(one, Sqrt(...))` —
  this is the deliberate "match the baseline" choice, not approximate `rsqrt`.
- `inv_dist3 = inv_dist / dist2 = 1/|r|³`. (Dividing `1/|r|` by `|r|²` gives
  `1/|r|³` while reusing `dist2` — avoids a second sqrt.)
- `scale = (k q_i q_j) · (1/|r|³)`. The charge product `pair_const[i*n+j]` is a
  single scalar, **broadcast** into every lane via `hn::Set` — the charges are
  the same in every sim, only the geometry differs.

```cpp
    const auto fx = hn::Mul(rx, scale);   // force vector on i from j
    const auto fy = hn::Mul(ry, scale);
    const auto fz = hn::Mul(rz, scale);

    axi = hn::Add(axi, fx);               // i gets +f  (in registers)
    ayi = hn::Add(ayi, fy);
    azi = hn::Add(azi, fz);
    hn::Store(hn::Sub(hn::Load(d, &ax[j * k]), fx), d, &ax[j * k]);  // j gets -f
    hn::Store(hn::Sub(hn::Load(d, &ay[j * k]), fy), d, &ay[j * k]);
    hn::Store(hn::Sub(hn::Load(d, &az[j * k]), fz), d, &az[j * k]);
  }
```

`f = r · scale` is the force on atom `i` due to `j`, for all sims. By Newton's
third law atom `j` feels `-f`. Atom `i`'s share is accumulated in the registers
`axi/ayi/azi`; atom `j`'s share is a read-modify-write straight to memory
(`load j`, subtract, `store j`). The `j` side can't stay in a register because a
later `i` iteration will revisit atom `j` as a partner.

```cpp
  hn::Store(axi, d, &ax[i * k]);   // flush atom i's accumulator once
  hn::Store(ayi, d, &ay[i * k]);
  hn::Store(azi, d, &az[i * k]);
}
```

After the inner loop, atom `i`'s total is written back a single time.

### 4c. Force → acceleration

```cpp
for (std::size_t i = 0; i < n; ++i) {
  const auto m = hn::Set(d, inv_mass[i]);
  hn::Store(hn::Mul(hn::Load(d, &ax[i * k]), m), d, &ax[i * k]);
  hn::Store(hn::Mul(hn::Load(d, &ay[i * k]), m), d, &ay[i * k]);
  hn::Store(hn::Mul(hn::Load(d, &az[i * k]), m), d, &az[i * k]);
}
```

Up to here `ax/ay/az` hold *forces*. This final pass multiplies each atom's
force by `1/m_i` (broadcast scalar) to get acceleration, in place — exactly the
`forces[i] = forces[i] * (1/mass)` step of the scalar version, vectorized over
sims.

---

## 5. `potential_energy()`

Goal (per lane):

$$U = \sum_{i<j} \frac{k\, q_i q_j}{|r_{ij}|}$$

```cpp
auto u = hn::Zero(d);                 // one running sum per lane
for (std::size_t i = 0; i < n; ++i) {
  ... load pxi/pyi/pzi ...
  for (std::size_t j = i + 1; j < n; ++j) {
    ... rx/ry/rz, dist2 as before ...
    const auto inv_dist = hn::Div(one, hn::Sqrt(dist2));         // 1/|r|
    u = hn::MulAdd(hn::Set(d, pair_const[i * n + j]), inv_dist, u);  // u += kqq/|r|
  }
}
hn::StoreU(u, d, pe);
```

Simpler than `accelerations`: there's no force scatter, just one accumulator
vector `u` that collects the energy of every `i<j` pair across all lanes. The
per-pair term is `pair_const[i*n+j] * (1/|r|)`, folded in with a fused
multiply-add. `u` lives entirely in a register through both loops.

Note the final **`StoreU`** (unaligned store) instead of `Store`. The comment
explains why:

> `pe` must hold ≥ `k` elements; written unaligned so callers need not
> over-align a small buffer.

The energy output is a tiny `k`-element buffer, so it's not worth forcing callers
to vector-align it — an unaligned store is cheap on modern hardware. Contrast with
the big `n*k` position/accel arrays, which *are* aligned so the hot loops use
fast aligned access.

---

## 6. Why it's fast — the summary

| Decision | Effect |
|---|---|
| SIMD over sims, not atoms | identical control flow in every lane → zero divergence, no cross-lane reduction |
| SoA-over-lanes layout | each atom's `k` values are one contiguous, aligned vector → plain `Load`/`Store`, no shuffles |
| Charges/masses broadcast (`Set`) | the constant molecule is shared, only geometry varies per lane |
| `i`-accumulator in registers | atom `i`'s force summed register-side, written once per `i` |
| Newton's third law (upper triangle) | each pair computed once, ~halving the work |
| FMA (`MulAdd`) | fused multiply-adds for `dist2` and the energy/force sums |
| Exact `1/sqrt` (not `rsqrt`) | matches the scalar baseline's numerics instead of trading accuracy |
| Static dispatch, quarantined TU | `-march=native` only where it's wanted; engine library stays SSE2-baseline so the 0001 reference stays honest |

## 7. Cross-references

- Scalar reference these mirror: [`src/system.cpp`](../../src/system.cpp#L5)
  (`CoulombForce::accelerations`, `potential_energy`).
- Benchmark consumer: [`bench/bench_force_batch.cpp`](../../bench/bench_force_batch.cpp),
  results in [`docs/benchmarks/0003-batched-force-kernel.md`](../benchmarks/0003-batched-force-kernel.md).
- Throughput / precision sweeps: benchmarks
  [0004](../benchmarks/0004-precision-sweep.md), [0005](../benchmarks/0005-f32-throughput.md),
  batched integrator [0006](../benchmarks/0006-batched-integrator.md).
- Build/baseline rationale: [`docs/decisions/0001-build-and-layout.md`](../decisions/0001-build-and-layout.md);
  precision policy: [`docs/decisions/0004-precision-policy.md`](../decisions/0004-precision-policy.md).
