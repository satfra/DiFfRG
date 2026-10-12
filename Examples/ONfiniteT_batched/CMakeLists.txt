cmake_minimum_required(VERSION 3.27.0)

project(ONfiniteT_batched)

# ##############################################################################
# Setup Application
# ##############################################################################

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

add_subdirectory(flows)

# ##############################################################################
# Batched assembly (model_batched.hh) for CG, DG, LDG and KT
# ##############################################################################

add_executable(flow_batched flow_batched.cc)
target_link_libraries(flow_batched PRIVATE DiFfRG::DiFfRG ONFiniteTBatchedFlows)

# ##############################################################################
# Symlink the parameter files into the build directory
# ##############################################################################

foreach(pfile parameter.toml parameter_KT.toml)
  execute_process(
    COMMAND ${CMAKE_COMMAND} "-E" "create_symlink"
            "${CMAKE_CURRENT_SOURCE_DIR}/${pfile}"
            "${CMAKE_CURRENT_BINARY_DIR}/${pfile}")
endforeach()
