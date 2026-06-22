include(FetchContent)

# Google Benchmark for microbenchmarks of the hot kernels.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.5)
FetchContent_MakeAvailable(benchmark)

add_executable(coulomb_bench bench_force.cpp)

target_link_libraries(coulomb_bench
  PRIVATE coulomb::engine coulomb_flags benchmark::benchmark)

# Step-count dispersion study for the SIMD-over-lanes decision. A plain driver
# (not a Google Benchmark microbenchmark): it samples many geometries, runs each
# explosion to convergence, and reports the step-count spread plus the shared-dt
# lockstep efficiency that decides whether the wavefront/refill path is worth it.
add_executable(coulomb_explosion bench_explosion.cpp)

target_link_libraries(coulomb_explosion
  PRIVATE coulomb::engine coulomb_flags)
