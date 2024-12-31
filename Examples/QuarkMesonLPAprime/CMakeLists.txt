cmake_minimum_required(VERSION 3.26.4)

project(QuarkMesonLPAprime)

find_package(DiFfRG REQUIRED HINTS /opt/DiFfRG)

add_executable(QuarkMeson QM.cc)
setup_application(QuarkMeson)
add_flows(QuarkMeson flows)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
