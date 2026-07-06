include(FetchContent)

# Google Highway: same static-dispatch SIMD-over-lanes kernel the batched
# integrator benchmark uses (bench/CMakeLists.txt). Declared independently here
# so `examples/` builds standalone with just -DCOULOMB_BUILD_EXAMPLES=ON, without
# requiring COULOMB_BUILD_BENCHMARKS. FetchContent no-ops if bench/ already
# populated the same declaration in this configure.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        1.2.0)
FetchContent_MakeAvailable(highway)

# CH4 dataset generator (see docs/benchmarks/0008): drives the batched f32
# lockstep integrator (include/coulomb/batched_integrator.hpp) over millions of
# independent explosions and writes a raw binary + JSON sidecar. Same
# -march=native / static-dispatch Highway regime as coulomb_batched_integrator.
add_executable(coulomb_generate_dataset generate_dataset.cpp)

target_link_libraries(coulomb_generate_dataset
  PRIVATE coulomb::engine coulomb_flags hwy)
target_compile_options(coulomb_generate_dataset PRIVATE -march=native)
