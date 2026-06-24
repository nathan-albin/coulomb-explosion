# How the batched lockstep integrator works

An annotated walkthrough of [`include/coulomb/batched_integrator.hpp`](../../include/coulomb/batched_integrator.hpp).

This header builds an **adaptive RK45 ODE solver that advances `k` independent
Coulomb explosions at once**, one per SIMD lane, on top of the batched force
kernel described in [batched-force-kernel.md](batched-force-kernel.md). Read that
one first — this document assumes you already understand the *SoA-over-lanes*
layout (`p[i*k + l]` = atom `i` in sim `l`) and why `k` is pinned to the hardware
vector width.

It is the realized implementation behind benchmark report
[0006](../benchmarks/0006-batched-integrator.md), and it mirrors the scalar
`DormandPrince45` in [`src/integrators.cpp`](../../src/integrators.cpp)
lane-for-lane.

---

## 1. The big idea: one solver, k simulations, one shared step size

The scalar code runs *one* Coulomb explosion: integrate the ODE
`y = (positions, velocities)`, `dy/dt = (velocities, accelerations)`, with an
adaptive step controller that shrinks/grows `dt` to hold the local error under
tolerance, and stop when the molecule has flown apart (potential energy has
decayed to a tiny fraction of its start value).

This class runs `k` of those **in lockstep**. Two design choices define it, and
both are deliberate simplifications called out in the header and in
[0006](../benchmarks/0006-batched-integrator.md):

- **Shared `dt`** — the whole batch advances with *one* step size each iteration,
  chosen by the **tightest (worst-error) still-active lane**. A lane that could
  legally take a big step is dragged down to the batch's step. This is the
  **min-envelope penalty**: you pay the cost of the hardest simulation for all of
  them.
- **No refill** — when a lane finishes (meets its stop threshold), it is
  *latched* (its final velocities are frozen) and *masked out of the error norm*
  so it stops constraining `dt`, but **its lane slot is not refilled** with a new
  simulation. The batch keeps running until the *last* lane converges. The idle
  lanes in the meantime are the **straggler idle**.

Together these make the integrator the *simple, honest lower bound* of the
lockstep family: the cheapest version to build, with no per-lane clocks and no
gather/scatter for refill. [0002](../benchmarks/0002-explosion-dispersion.md)
estimated how much throughput this leaves on the table; 0006 measures it.

Everything else — the Butcher tableau, FSAL, the RMS error norm, the step
controller, the initial-step heuristic — is **identical to the scalar
`DormandPrince45`**, just evaluated across lanes and reduced to one shared
accept/reject decision.

---

## 2. Data members: the working set, all SoA-over-lanes

```cpp
Vec3Soa<T> pos_, vel_, stage_, probe_, final_vel_;
Vec3Soa<T> kpos_[7], kvel_[7];
```

`Vec3Soa<T>` ([line 46](../../include/coulomb/batched_integrator.hpp#L46)) is just
three aligned arrays `x, y, z`, each length `n*k`, with a `swap` that swaps the
pointers (O(1), no copy). It's the unit the kernels and stages operate on.

- `pos_`, `vel_` — current state `y` of all `k` sims.
- `stage_` — scratch positions for an RK stage (the trial point a derivative is
  evaluated at).
- `probe_` — scratch used only by the initial-step heuristic.
- `final_vel_` — where a lane's velocities are frozen when it latches.
- `kpos_[s]`, `kvel_[s]` for `s = 0..6` — the **seven RK stage derivatives**.
  Because `y = (pos, vel)`, each stage derivative also has two halves:
  `kpos_[s]` holds the position-derivative (= a velocity) and `kvel_[s]` holds
  the velocity-derivative (= an acceleration) at stage `s`.

The scalars per atom, shared across all lanes (same molecule), are precomputed in
the constructor:

```cpp
pair_const_[i*n_ + j] = coulomb_k * q_i * q_j;   // narrowed to T, fed to the kernel
inv_mass_[i]          = 1 / m_i;                 // narrowed to T
mass_d_[i]            = m_i;                      // kept in double for finalize
```

And several tiny `k`-length lane buffers (`scratch_`, `d1_`, `d2_`, `h0_`,
`pe_lane_`, …) used to pull a per-lane vector out of a register into plain memory
so the scalar reduction loops can read individual lanes.

> Note the precision split: positions/velocities/stages are `T` (`float` or
> `double`), but the **finalize accounting** (`mass_d_`, `pe0_`, `e0_`, the
> kinetic-energy sum) is always `double`. Momenta and energy bookkeeping are
> done in double regardless of the integration precision — see
> [0004](../decisions/0004-precision-policy.md).

The constructor also enforces the lane-count contract from the force kernel:

```cpp
if (k_ != hn::Lanes(hn::ScalableTag<T>()))
  throw std::invalid_argument("BatchedRK45: lanes must equal the SIMD width for T");
```

`k` is the ISA vector width, not a free knob (AVX2: 4 for `double`, 8 for
`float`).

---

## 3. The ODE and the DP5(4) tableau

The system is first-order in `y = (x, v)`:

$$\frac{d}{dt}\begin{pmatrix} x \\ v \end{pmatrix}
   = \begin{pmatrix} v \\ a(x) \end{pmatrix},\qquad
   a(x) = \text{batched Coulomb acceleration}.$$

So every RK stage needs two things: the current velocity (the `x`-derivative) and
`a(stage positions)` (the `v`-derivative). That's why each stage stores a
`kpos_`/`kvel_` pair.

The Butcher tableau ([lines 196–206](../../include/coulomb/batched_integrator.hpp#L196))
is Dormand–Prince 5(4) — a 7-stage pair giving a 5th-order solution and an
embedded 4th-order solution; their difference (`kE_`) is the local error estimate.
The coefficients are stored as `double` and narrowed to `T` at each use, and they
are *byte-identical* to the scalar integrator's. Two features matter for the code
shape:

- **FSAL** (First Same As Last): the stage-7 `a`-row equals the 5th-order weights
  `b`, so the last stage's derivative *is* the new state's derivative — it becomes
  the next step's stage-1 derivative for free. That's the `kvel_[0].swap(kvel_[6])`
  on accept.
- The error weights `kE_` are applied directly to the stage derivatives to form
  the embedded error without separately forming the 4th-order solution.

---

## 4. `run()` — the outer convergence loop

[Lines 123–191](../../include/coulomb/batched_integrator.hpp#L123). The lifecycle:

### 4a. Per-lane stop thresholds

```cpp
batched::potential_energy<T>(... pe_lane_.data());   // PE_0 for every lane at once
for each lane:
  pe0_[lane]     = PE_0;
  pe_stop_[lane] = pe_stop_fraction * PE_0;          // stop when PE drops this low
  e0_[lane]      = pe0_[lane];                        // total energy; KE_0 = 0 (started at rest)
  if (pe0_ <= pe_stop_) latch immediately            // degenerate pe_stop_fraction >= 1
```

Each lane gets its *own* stop threshold from its *own* initial potential energy.
`pe_stop_fraction` and `max_dt`/`max_steps` come from `RunConfig`
([`include/coulomb/driver.hpp`](../../include/coulomb/driver.hpp), the shared
convergence config) — the same knobs the scalar driver uses.

### 4b. The lockstep loop

```cpp
while (n_done < k_ && iter < cfg.max_steps) {
  step(done.data(), max_dt);                  // one accepted shared-dt step, all lanes
  ++iter;
  batched::potential_energy<T>(... pe_lane_); // re-measure PE for every lane
  for each lane:
    if (!done[lane] && pe_lane_[lane] <= pe_stop_[lane]):
      done[lane] = 1; converged; record steps; latch_lane(lane); ++n_done;
}
```

The batch takes one accepted step, then checks which lanes have now flown apart.
A finished lane is marked `done`, its convergence step index recorded, and its
velocities frozen via `latch_lane`. The loop ends when **all** lanes are done (or
the `max_steps` safety cap trips). Any lane still unfinished at the cap is latched
where it stands ([lines 164–169](../../include/coulomb/batched_integrator.hpp#L164)).

`done[]` is the mask that drives the whole min-envelope/straggler behavior: it's
passed into `step` so finished lanes neither block step acceptance nor shrink
`dt`.

### 4c. Per-lane finalize: energy redistribution → momenta

[Lines 173–189](../../include/coulomb/batched_integrator.hpp#L173). For each lane,
in `double`:

```cpp
ke = sum_i 0.5 * m_i * |v_i|^2;                     // residual KE at stop
s  = (redistribute_energy && ke>0) ? sqrt(e0/ke) : 1;
momenta[lane,i] = (m_i * s) * v_i;
```

At the stop point a little potential energy remains, so kinetic energy is slightly
short of the true asymptotic value. Rescaling every velocity by
`s = sqrt(E_0 / KE)` makes the final KE exactly equal the initial total energy,
converting that residual PE into KE. Uniform scaling preserves total momentum
(still ~0 for a system started at rest) and the correction vanishes as the stop
threshold tightens. This is the same redistribution the scalar driver does; see
the `redistribute_energy` doc in `RunConfig`.

The `Result` reports per-lane momenta, per-lane converged flag and step count,
and `batch_steps` (how many accepted SIMD iterations the whole batch ran — the
straggler-idle cost is `batch_steps` vs. the per-lane step counts).

---

## 5. `step()` — one accepted shared-dt step

[Lines 228–343](../../include/coulomb/batched_integrator.hpp#L228). This is the
SIMD heart. It performs the 7 RK stages, computes a per-lane error, reduces to the
**worst active lane**, and accepts or rejects — looping (shrinking `h_`) until the
shared step is accepted.

### 5a. FSAL stage-1 and lazy initial step

```cpp
if (!fsal_valid_) {                 // first ever step: build k1 from scratch
  kpos_[0] = vel_;                  // x-derivative = velocity
  accel(pos_, kvel_[0]);            // v-derivative = a(x)
  fsal_valid_ = true;
}
if (h_ <= 0) h_ = select_initial_step(done, max_dt);   // §6, only once
```

On every subsequent step `kpos_[0]/kvel_[0]` are already valid from the previous
accept (FSAL), so stage 1 is free.

### 5b. Pointer tables

```cpp
T* P[3]={pos_.x,...}; T* V[3]={vel_.x,...}; T* ST[3]={stage_.x,...};
T* KP[7][3]; T* KV[7][3];   // raw pointers to every stage's x/y/z
```

Flattening the SoA arrays into `[component]` and `[stage][component]` pointer
tables lets the stage loops index uniformly with a tight `c = 0..2` inner loop.
They're rebuilt each call because `pos_`/`vel_`/`kvel_[0]` pointers move on accept
(the `swap`s) — but since accept returns immediately, they stay valid across the
reject retries within one call.

### 5c. The error functor

```cpp
auto errc = [&](verr, yold, ynew) {
  scale = atol + rtol * max(|yold|, |ynew|);   // per-component error scale
  rr = verr / scale;
  return rr * rr;                              // squared, ready to accumulate
};
```

This is the RMS-norm weighting: each component's error is scaled by
`atol + rtol*max(|y_old|, |y_new|)`, exactly matching the scalar norm. It returns
the *squared* normalized error so the caller can sum directly.

### 5d. Stages 2..7

```cpp
for (s = 1; s < 7; ++s) {
  a = kA_[s-1];                            // this stage's a-row
  for each atom i, each component c:
    dx = sum_{j<s} a[j] * KP[j][c];        // weighted combo of prior x-derivatives
    dv = sum_{j<s} a[j] * KV[j][c];        // ... and v-derivatives
    stage_[c]   = pos_[c] + h * dx;        // trial position
    kpos_[s][c] = vel_[c] + h * dv;        // trial velocity (= this stage's x-deriv)
  accel(stage_, kvel_[s]);                 // this stage's v-deriv = a(trial positions)
}
```

Each stage forms the RK linear combination of all previous stage derivatives
(`MulAdd` accumulation across lanes), produces a trial position and velocity, then
calls the batched force kernel once to get the acceleration at that trial point.
All arithmetic is full-width vector ops — every lane's stage is computed together.
The single `accel()` per stage is the dominant cost, and it's exactly the kernel
from [batched-force-kernel.md](batched-force-kernel.md).

### 5e. Embedded error → worst active lane

```cpp
err_sq = 0 (vector);
for each atom i, component c:
  ep = sum_s kE_[s] * KP[s][c];            // embedded error, position half
  ev = sum_s kE_[s] * KV[s][c];            // ... velocity half
  err_sq += errc(ep*h, pos_[c], stage_[c]);    // y_new positions = stage_
  err_sq += errc(ev*h, vel_[c], kpos_[6][c]);  // y_new velocities = kpos_[6]
enorm = sqrt(err_sq / (6n));               // per-lane RMS error, still a vector
hn::StoreU(enorm, scratch_);               // spill lanes to memory
max_err = max over lanes where !done[lane] of scratch_[lane];   // <-- the reduction
```

`err_sq` accumulates the squared, scaled error across all `6n` state components
(`n` atoms × 3 dims × {pos, vel}) **per lane, in a vector register**. After the
RMS normalization, the per-lane error vector is spilled to `scratch_` and reduced
to a single `max_err` over the **active** lanes only. This one scalar is the whole
batch's accept/reject signal — *the* lockstep coupling. Done lanes are skipped, so
a finished simulation can't force the batch to keep taking tiny steps.

### 5f. Accept / reject and the step controller

```cpp
if (max_err <= 1) {                  // ACCEPT
  pos_.swap(stage_);                 // x_new
  vel_.swap(kpos_[6]);               // v_new
  kvel_[0].swap(kvel_[6]);           // a(x_new) becomes next k1 (FSAL)
  kpos_[0] = vel_;                   // v_new becomes next k1's x-derivative
  factor = (max_err==0) ? maxFactor : min(maxFactor, safety * max_err^(-1/5));
  if (rejected_once) factor = min(factor, 1);   // don't grow after a reject this step
  h_ = min(h * factor, max_dt);      // step size for NEXT call
  return;
}
// REJECT
factor = max(minFactor, safety * max_err^(-1/5));
h_ = h * factor;                     // shrink and retry the SAME step
rejected_once = true;
```

Standard PI-free adaptive control with the I-controller exponent `-1/5` (order 5),
safety 0.9, growth clamped to ×10, shrink to ×0.2 — all identical constants to the
scalar version. The accept path commits the new state with O(1) buffer swaps
(no data copy), wires up FSAL for the next step, and stores the *next* step size.
Reject shrinks `h_` and loops without advancing. A pathological `>100` rejections
throws rather than spinning.

The crucial lockstep property: because `max_err` is the **worst active lane**, the
batch accepts a step only when it's good enough for *every* still-running
simulation, and the next `h_` is sized for that worst lane. That is the min-
envelope, in code.

---

## 6. `select_initial_step()` — Hairer's heuristic, batched

[Lines 348–420](../../include/coulomb/batched_integrator.hpp#L348). Picks the very
first `dt`. It's Hairer's standard starting-step heuristic (the one SciPy uses),
done per lane in the same RMS norm, then reduced by **min over active lanes**
(min-envelope again — start no bigger than the tightest lane wants). Requires
stage-1 already in `kpos_[0]` (= `v0`) / `kvel_[0]` (= `a0`).

The recipe, all vectorized:

1. `d0 = ||y0||`, `d1 = ||f0||` in the scaled norm (`scale = atol + rtol*|y|`),
   per lane.
2. `h0 = 0.01 * d0/d1` (or a `1e-6` floor when either is tiny).
3. Probe one explicit-Euler step `y1 = y0 + h0*f0` (positions only), evaluate
   `a1 = a(stage_)` into `probe_`.
4. `d2 = ||f1 - f0|| / h0` — an estimate of the derivative's rate of change
   (curvature), per lane.
5. `h1 = (0.01 / max(d1, d2))^(1/5)`; the lane's step is
   `min(100*h0, h1, max_dt)`.
6. `return min over active lanes` of that.

The per-lane spills (`scratch_`, `d1_`, `d2_`, `h0_`) are again the bridge from
vector registers to the scalar `for lane` reductions, since the final min and the
elementwise branches (`< 1e-5 ? ...`) are easiest lane-by-lane.

---

## 7. How the two simplifications show up in the code

| Design choice | Where it lives |
|---|---|
| **Shared `dt`** | `step()` reduces the per-lane `enorm` vector to a single `max_err` over active lanes (§5e); one accept/reject for the whole batch; `select_initial_step` takes the `min` over lanes (§6). |
| **No refill** | `latch_lane` freezes a finished lane's velocities into `final_vel_`; `done[]` masks it out of every `max`/`min` reduction; nothing ever writes a fresh geometry into that slot. The loop runs until `n_done == k_`. |
| **Min-envelope cost** | the `max_err` over active lanes in §5e and the `min` over lanes in §6 — the batch always moves at the worst active lane's pace. |
| **Straggler idle cost** | `batch_steps` (total accepted iterations) vs. per-lane `steps[]`: the gap is iterations a converged lane sat idle. |

---

## 8. Cross-references

- Prerequisite — the force/energy kernel this is built on:
  [batched-force-kernel.md](batched-force-kernel.md) /
  [`include/coulomb/batched_force.hpp`](../../include/coulomb/batched_force.hpp).
- Scalar reference it mirrors lane-for-lane:
  `DormandPrince45` in [`src/integrators.cpp`](../../src/integrators.cpp).
- Shared convergence config: `RunConfig` in
  [`include/coulomb/driver.hpp`](../../include/coulomb/driver.hpp).
- Benchmark / framing: [0006](../benchmarks/0006-batched-integrator.md) (realized
  throughput, lockstep efficiency), motivated by the dispersion study
  [0002](../benchmarks/0002-explosion-dispersion.md).
- Precision policy (why finalize is in double): [0004](../decisions/0004-precision-policy.md),
  throughput [0005](../benchmarks/0005-f32-throughput.md).
