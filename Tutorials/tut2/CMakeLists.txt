cmake_minimum_required(VERSION 3.27.0)

project(tut2)

find_package(DiFfRG REQUIRED HINTS $ENV{HOME}/.local/share/DiFfRG/)

add_executable(tut2 tut2.cc)
target_link_libraries(tut2 PRIVATE DiFfRG::DiFfRG)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
