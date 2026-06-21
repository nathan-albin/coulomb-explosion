include(FetchContent)

# Catch2 v3 as the test framework. Fetched at configure time so a fresh clone
# builds with nothing but a compiler + CMake + network.
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4)
FetchContent_MakeAvailable(Catch2)

# nlohmann/json for reading the reference fixtures emitted by the Python oracle.
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(coulomb_tests
  test_system.cpp
  test_integrator.cpp
  test_reference.cpp)

target_link_libraries(coulomb_tests
  PRIVATE coulomb::engine coulomb_flags
          Catch2::Catch2WithMain nlohmann_json::nlohmann_json)

# Absolute path to the checked-in reference data, generated from the Python
# oracle by tests/reference/gen_reference_cases.py.
target_compile_definitions(coulomb_tests PRIVATE
  COULOMB_REFERENCE_JSON="${CMAKE_CURRENT_SOURCE_DIR}/reference/reference_cases.json")

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(coulomb_tests)
