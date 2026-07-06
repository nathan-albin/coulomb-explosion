cmake_minimum_required(VERSION 3.20)

project(coulomb
  VERSION 0.1.0
  DESCRIPTION "Coulomb-explosion simulator for ML training-data generation"
  LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Project-wide settings
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for clang-tidy / editors.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to a sensible build type for single-config generators.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
endif()

# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------
option(COULOMB_BUILD_TESTS      "Build unit / correctness tests"  ON)
option(COULOMB_BUILD_BENCHMARKS "Build microbenchmarks"           OFF)
option(COULOMB_BUILD_EXAMPLES   "Build example data-generation drivers" OFF)
option(COULOMB_ENABLE_WARNINGS  "Enable a strict warning set"     ON)
option(COULOMB_ENABLE_SANITIZERS "Build with ASan + UBSan"        OFF)
option(COULOMB_SINGLE_PRECISION "Build the engine in float (Real=float)" OFF)

# Interface target carrying our compile flags, linked by everything we build.
add_library(coulomb_flags INTERFACE)
target_compile_features(coulomb_flags INTERFACE cxx_std_20)

# Precision switch. Applied on the interface target the engine, tests, and
# benchmarks all link, so a given build directory is internally consistent
# (every TU sees the same Real — no ODR split). Configure a separate build dir
# with the `*-f32` preset to get the single-precision binaries alongside the
# default double-precision tree.
if(COULOMB_SINGLE_PRECISION)
  target_compile_definitions(coulomb_flags INTERFACE COULOMB_SINGLE_PRECISION)
endif()

if(COULOMB_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(coulomb_flags INTERFACE /W4 /permissive-)
  else()
    target_compile_options(coulomb_flags INTERFACE
      -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
  endif()
endif()

if(COULOMB_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(coulomb_flags INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(coulomb_flags INTERFACE -fsanitize=address,undefined)
endif()

# ----------------------------------------------------------------------------
# Targets
# ----------------------------------------------------------------------------
add_subdirectory(src)
add_subdirectory(apps)

if(COULOMB_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(COULOMB_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()

if(COULOMB_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
