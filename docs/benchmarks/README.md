# Benchmark reports

Each report captures one performance experiment. Keep them reproducible: record
the hardware, compiler, flags, and git SHA so a number can always be traced back
to the conditions that produced it.

## Capturing the environment

For every run, record:

- CPU model and cache sizes (`lscpu`, `lscpu -C`)
- Compiler + version and build flags (`-march`, `-O`, precision)
- Git SHA of the engine
- Dataset / problem size (N atoms, step count)

## Report template

```
# <experiment name>
- Date / SHA / machine:
- Hypothesis:
- Method:        (how it was measured; commands)
- Result:        (table + plot from python/analysis)
- Conclusion:    (what changed, and the speedup vs. baseline)
- Follow-ups:
```

## Tooling notes

- `bench/coulomb_bench` (Google Benchmark) for kernel microbenchmarks.
- `perf stat -d` for IPC, cache-miss, and branch-miss counters.
- `perf record` + flamegraph for hot-path attribution.
- `valgrind --tool=cachegrind` for cache modeling without hardware counters.
- `llvm-mca` for static instruction-level analysis of a hot kernel.
