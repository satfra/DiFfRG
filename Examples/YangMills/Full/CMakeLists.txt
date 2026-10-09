cmake_minimum_required(VERSION 3.26.4)

project(YangMills)

find_package(DiFfRG REQUIRED HINTS ~/.local/share/DiFfRG)

add_subdirectory(flows)
add_executable(YangMills Yang-Mills.cc)
target_link_libraries(YangMills DiFfRG::DiFfRG YangMillsFlows)
# flow_abort.hh: the runaway criterion, shared with the SP/ truncation next door so that both
# classify a probe the same way.
target_include_directories(YangMills PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.toml"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.toml")
