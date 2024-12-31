cmake_minimum_required(VERSION 3.26.4)

project(FourFermi)

find_package(DiFfRG REQUIRED HINTS /opt/DiFfRG_install)

add_executable(FourFermi FourFermi.cc)
setup_application(FourFermi flows)
add_flows(FourFermi flows)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/external_data"
    "${CMAKE_CURRENT_BINARY_DIR}/external_data")
