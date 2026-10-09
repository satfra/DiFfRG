cmake_minimum_required(VERSION 3.27.0)

project(ONfiniteT)

# ##############################################################################
# Setup Application
# ##############################################################################

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

add_subdirectory(flows)

# ##############################################################################
# FE discretizations (model.hh / model_LDG.hh, parameter.toml)
# ##############################################################################

add_executable(LDG LDG.cc)
target_link_libraries(LDG PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)

add_executable(CG CG.cc)
target_link_libraries(CG PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)

add_executable(DG DG.cc)
target_link_libraries(DG PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)

# ##############################################################################
# Kurganov-Tadmor finite volume, in rho- and in sigma-coordinates
# ##############################################################################

add_executable(KT KT.cc)
target_link_libraries(KT PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)

add_executable(KT_sigma KT_sigma.cc)
target_link_libraries(KT_sigma PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)

# ##############################################################################
# Symlink the parameter files into the build directory
# ##############################################################################

foreach(pfile parameter.toml parameter_batched.toml parameter_KT.toml parameter_KT_sigma.toml)
  execute_process(
    COMMAND ${CMAKE_COMMAND} "-E" "create_symlink"
            "${CMAKE_CURRENT_SOURCE_DIR}/${pfile}"
            "${CMAKE_CURRENT_BINARY_DIR}/${pfile}")
endforeach()

# ##############################################################################
# Batched assembly pilot (flows_batched/, model_batched.hh) for CG, DG, LDG and KT:
# flow_batched runs a flow, bench_batched times single residual and jacobian calls.
# ##############################################################################

option(ONFINITET_BATCHED "Build flow_batched and bench_batched" OFF)
if(ONFINITET_BATCHED)
  add_subdirectory(flows_batched)

  add_executable(flow_batched flow_batched.cc)
  target_link_libraries(flow_batched PRIVATE DiFfRG::DiFfRG ONFiniteTBatchedFlows)

  add_executable(bench_batched bench_batched.cc)
  target_link_libraries(bench_batched PRIVATE DiFfRG::DiFfRG ONFiniteTBatchedFlows)
endif()
