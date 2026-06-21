add_executable(coulomb_cli main.cpp)

set_target_properties(coulomb_cli PROPERTIES OUTPUT_NAME coulomb)

target_link_libraries(coulomb_cli
  PRIVATE coulomb::engine coulomb_flags)
