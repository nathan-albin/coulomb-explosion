# The simulation engine, built as a library so it can be exercised by the CLI,
# the tests, and the benchmarks without rebuilding.
add_library(coulomb_engine
  system.cpp
  integrators.cpp
  driver.cpp
  sampler.cpp)

target_include_directories(coulomb_engine
  PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(coulomb_engine
  PUBLIC coulomb_flags)

# Namespaced alias so downstream targets (and a future install/export) use a
# consistent name.
add_library(coulomb::engine ALIAS coulomb_engine)
