include(FetchContent)

# Google Benchmark for microbenchmarks of the hot kernels.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.5)
FetchContent_MakeAvailable(benchmark)

# Google Highway: portable SIMD for the batched (SIMD-over-lanes) force kernel
# and, later, the batched integrator with per-lane masking / refill — control
# flow no autovectorizer can express. See docs/benchmarks/0001 and 0002.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.2.0)
FetchContent_MakeAvailable(highway)

add_executable(coulomb_bench bench_force.cpp)

target_link_libraries(coulomb_bench
  PRIVATE coulomb::engine coulomb_flags benchmark::benchmark)

# SIMD-over-lanes batched force kernel (the production lever). Built with
# -march=native so Highway compiles for the host's widest ISA (AVX-512 -> K=8
# doubles on this Tiger Lake target); the rest of the project stays generic so
# 0001's scalar floor remains the conservative SSE2 baseline.
add_executable(coulomb_force_batch bench_force_batch.cpp)

target_link_libraries(coulomb_force_batch
  PRIVATE coulomb::engine coulomb_flags benchmark::benchmark hwy)
target_compile_options(coulomb_force_batch PRIVATE -march=native)

# Step-count dispersion study for the SIMD-over-lanes decision. A plain driver
# (not a Google Benchmark microbenchmark): it samples many geometries, runs each
# explosion to convergence, and reports the step-count spread plus the shared-dt
# lockstep efficiency that decides whether the wavefront/refill path is worth it.
add_executable(coulomb_explosion bench_explosion.cpp)

target_link_libraries(coulomb_explosion
  PRIVATE coulomb::engine coulomb_flags)

# Precision-robustness sweep: f32 vs f64 asymptotic-momentum drift and failure
# rate across molecule sizes. Precision-agnostic (uses the build's Real); build
# under both relwithdebinfo and relwithdebinfo-f32 and diff the two CSVs.
add_executable(coulomb_precision_sweep bench_precision_sweep.cpp)

target_link_libraries(coulomb_precision_sweep
  PRIVATE coulomb::engine coulomb_flags)
