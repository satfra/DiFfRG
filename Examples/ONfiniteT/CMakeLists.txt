cmake_minimum_required(VERSION 3.27.0)

project(ONfiniteT LANGUAGES CXX)

# ##############################################################################
# Setup Application
# ##############################################################################

if(NOT DEFINED DiFfRG_DIR)
  set(DiFfRG_DIR /opt/DiFfRG)
endif()
# find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR})
find_package(DiFfRG REQUIRED HINTS ../../DiFfRG_install)

add_subdirectory(flows)

add_executable(LDG LDG.cc ${flow_sources})
setup_application(LDG)

add_executable(CG CG.cc ${flow_sources})
setup_application(CG)

add_executable(dDG dDG.cc ${flow_sources})
setup_application(dDG)

execute_process(
  COMMAND
    ${CMAKE_COMMAND} "-E" "create_symlink"
    "${CMAKE_CURRENT_SOURCE_DIR}/parameter.json"
    "${CMAKE_CURRENT_BINARY_DIR}/parameter.json")
