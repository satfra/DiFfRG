cmake_minimum_required(VERSION 3.27.0)

project(tut5)

find_package(DiFfRG REQUIRED HINTS $ENV{HOME}/.local/share/DiFfRG/)

add_executable(tut5 tut5.cc)
target_link_libraries(tut5 PRIVATE DiFfRG::DiFfRG)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")

# The equivalent TOML parameter file, so that `./tut5 -p parameter.toml` works from the
# build directory too. parameter.json is still the default of the two.
execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.toml"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.toml")
