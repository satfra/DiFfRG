cmake_minimum_required(VERSION 3.27.0)

project(tut9)

find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

# The generated flow equations are built into their own (static) library. Building it first runs the NumTracer
# generators of the flows (the numtrace target).
add_subdirectory(flows)

add_executable(tut9 tut9.cc)
target_link_libraries(tut9 PRIVATE DiFfRG::DiFfRG QMFlows)

# bench compares the codegen options on single evaluations of the flows; its extra flows are a library of their own.
option(TUT9_BENCH "Build the benchmark of the codegen options" OFF)
if(TUT9_BENCH)
  add_subdirectory(bench_flows)
  add_executable(bench bench.cc)
  target_link_libraries(bench PRIVATE DiFfRG::DiFfRG QMFlows QMBenchFlows)
endif()

execute_process(COMMAND ${CMAKE_COMMAND} "-E" "create_symlink" "${CMAKE_CURRENT_SOURCE_DIR}/parameter.toml"
                        "${CMAKE_CURRENT_BINARY_DIR}/parameter.toml")
