# Documentation & reports

This is where design decisions and performance findings are written up — the
rationale behind the code, captured beyond inline comments.

## Structure

- `decisions/` — short, dated design notes (an ADR-style log). One file per
  decision: the question, the options considered, the choice, and why. This is
  also where a *rejected* approach is recorded so the repo stays clean (the code
  for it lives behind a strategy interface or in tagged git history, not as
  commented-out blocks).
- `benchmarks/` — performance reports generated from real runs: baseline vs.
  optimized, accuracy vs. solver, per-architecture tuning notes. Each is framed
  as an experiment: hypothesis → method → result → conclusion.
- `internals/` — annotated code walkthroughs: how a particular hot-path module
  actually works, line by line, for readers diving into the implementation.
  - `batched-force-kernel.md` — the SIMD-over-lanes Coulomb force/energy kernel
    (`include/coulomb/batched_force.hpp`).
  - `batched-integrator.md` — the shared-dt lockstep batched RK45 integrator
    (`include/coulomb/batched_integrator.hpp`) built on that kernel.

See `decisions/0001-build-and-layout.md` for the template and the first entry.
