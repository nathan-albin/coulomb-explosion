# 0003 — Adaptive RK45, convergence termination, and energy redistribution

- **Status:** accepted
- **Date:** 2026-06-21

## Context

A Coulomb explosion is an unbounded, repulsive N-body problem: atoms start
close (strong forces, fast dynamics) and fly apart forever while the residual
potential energy decays like `1/t`. The validation target is the *asymptotic*
(`t -> infinity`) per-atom momenta. Two things follow:

- A fixed-step integrator cannot do this economically. The close-approach phase
  needs a tiny step; the tail needs an enormous one. Velocity Verlet (already in
  the engine, symplectic, energy-conserving) would have to march the small step
  all the way out — a step-count wall, not an accuracy or stability wall.
- The Python oracle (`python/reference/coulomb.py`) sidesteps "infinity" by
  calling `scipy.solve_ivp` with `RK45` out to a hard-coded `t = 1e10` and a
  single end-point evaluation, then asserts the forces have vanished.

We need the C++ engine to reproduce the oracle's asymptotic momenta without
inheriting its two awkward properties: a magic stopping time, and the energy
drift of a non-symplectic method.

## Decision

- **Port scipy's `RK45` as Dormand–Prince DP5(4).** `IntegratorKind::RK45` is
  the adaptive accuracy reference. It mirrors scipy's controller so the C++
  adaptivity matches the oracle's: the embedded 4th/5th-order pair, the RMS
  error norm with `scale = atol + rtol*max(|y_old|, |y_new|)`, `SAFETY = 0.9`,
  factor clamp `[0.2, 10]`, Hairer initial-step selection, and FSAL. Tolerances
  come from `IntegratorOptions` and default to the oracle's `1e-8 / 1e-16`. The
  dynamics are autonomous, so the stage abscissae `c_i` are never used.
- **The integrator owns its step size; `Integrator::step`'s `dt` is a cap.**
  Fixed-step schemes take exactly `dt`; adaptive schemes take the largest
  accepted step `<= dt`. This lets the driver pass a large cap and let the step
  grow freely into the tail.
- **Terminate on a potential-energy threshold, not a fixed time.** A new driver
  (`run_to_convergence`, `driver.hpp/.cpp`) steps until `PE <= pe_stop_fraction
  * PE_0`. Owning the loop in C++ makes this per-step check free — the concern
  that drove the oracle to a single end-point evaluation (re-entering
  `solve_ivp`) does not exist here.
- **Redistribute residual potential energy into kinetic energy.** After
  termination, scale every velocity by `s = sqrt(E_0 / KE)` so the final kinetic
  energy equals the initial total energy exactly. Uniform scaling is
  momentum-preserving (total momentum is unchanged, so a system started at rest
  stays at zero net momentum) and `s -> 1` as the threshold tightens.

## Consequences

- **The engine is more energy-faithful than the oracle.** scipy's non-symplectic
  RK45 drifts: its recorded `kinetic_energy_final` for `two_protons` is
  `1.0000000490` against `PE_0 = 1.0`. The redistribution pins our KE to `E_0`
  exactly, so the Tier-2 momentum comparison floors at the oracle's own drift
  (`sqrt(1.0000000490) - 1 = 2.4e-8`), not at our error. The reference test
  (`tests/test_reference.cpp`) therefore compares within `1e-6`, comfortably
  above that floor, rather than demanding bit-exact agreement.
- **Early termination is cheap and safe.** With redistribution, terminating at
  `pe_stop_fraction = 1e-6` (~100 adaptive steps, `t ~ 1e8`) already lands within
  ~5e-7 of the oracle; tightening to `1e-8` reaches the drift floor in only a
  handful more steps. `pe_stop_fraction` is the knob for trading steps against
  asymptotic accuracy.
- **Uniform scaling is an approximation to the true asymptotic partition.** It is
  energy-exact and momentum-preserving but does not reproduce the exact infinite-
  time distribution of kinetic energy among atoms; its error vanishes with the
  threshold. If a future need arises for the exact partition, this is the place
  to revisit.
- **Tolerances are shared with the oracle by construction.** `IntegratorOptions`
  defaults track `RKRTOL/RKATOL`; if the oracle's tolerances change, update both
  and regenerate the fixtures.
