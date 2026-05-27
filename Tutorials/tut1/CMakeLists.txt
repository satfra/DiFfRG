cmake_minimum_required(VERSION 3.26.4)

project(tut1)

find_package(DiFfRG REQUIRED HINTS $ENV{HOME}/.local/share/DiFfRG)

add_executable(tut1 tut1.cc)
setup_application(tut1)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
