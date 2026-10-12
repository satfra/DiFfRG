cmake_minimum_required(VERSION 3.27.0)

project(tut8)

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

# The generated flow equations are built into their own (static) library.
add_subdirectory(flows)

add_executable(tut8 tut8.cc)
target_link_libraries(tut8 PRIVATE DiFfRG::DiFfRG LPA_QM_flows)

execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/parameter.toml"
                        "${CMAKE_CURRENT_BINARY_DIR}/parameter.toml")
