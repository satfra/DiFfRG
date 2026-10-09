cmake_minimum_required(VERSION 3.27.0)

project(tut6)

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

# The generated flow equations are built into their own (static) library.
add_subdirectory(flows)

add_executable(tut6 tut6.cc)
target_link_libraries(tut6 PRIVATE DiFfRG::DiFfRG ONFlows)

foreach(pfile parameter.toml parameter_KT.toml)
  execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/${pfile}"
                          "${CMAKE_CURRENT_BINARY_DIR}/${pfile}")
endforeach()
