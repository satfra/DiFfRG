cmake_minimum_required(VERSION 3.26.4)

project(YangMills)

find_package(DiFfRG REQUIRED HINTS /opt/DiFfRG_install)

add_executable(Yang-Mills Yang-Mills.cc)
setup_application(Yang-Mills)
add_flows(Yang-Mills flows)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
