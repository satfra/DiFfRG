cmake_minimum_required(VERSION 3.27.0)

project(tut4)

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

# The generated, momentum-dependent flow equations are built into their own (static) library.
add_subdirectory(flows)

add_executable(tut4 tut4.cc)
target_link_libraries(tut4 PRIVATE DiFfRG::DiFfRG YangMillsFlows)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
