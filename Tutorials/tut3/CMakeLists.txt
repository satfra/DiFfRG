cmake_minimum_required(VERSION 3.27.0)

project(tut3)

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

# The generated flow equations are built into their own (static) library.
add_subdirectory(flows)

add_executable(tut3 tut3.cc)
target_link_libraries(tut3 PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
