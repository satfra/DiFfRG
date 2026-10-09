cmake_minimum_required(VERSION 3.26.4)

project(QuarkMesonLPAprime)

find_package(DiFfRG REQUIRED HINTS $ENV{HOME}/.local/share/DiFfRG/)

add_executable(QuarkMesonLPAprime QM.cc)
add_subdirectory(flows)
target_link_libraries(QuarkMesonLPAprime PUBLIC DiFfRG::DiFfRG LPA_QM_flows)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
