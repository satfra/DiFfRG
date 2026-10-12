cmake_minimum_required(VERSION 3.27.0)

project(tut7)

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

add_executable(tut7 tut7.cc)
target_link_libraries(tut7 PRIVATE DiFfRG::DiFfRG)

execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/parameter.toml"
                        "${CMAKE_CURRENT_BINARY_DIR}/parameter.toml")
