cmake_minimum_required(VERSION 3.20)

project(DiFfRG)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/cmake/common.cmake)

message(
  "${BoldWhite}
  __| |________________________________________________| |__
  __   ________________________________________________   __
    | |                                                | |
    | |  ██████╗ ██╗███████╗███████╗██████╗  ██████╗   | |
    | |  ██╔══██╗██║██╔════╝██╔════╝██╔══██╗██╔════╝   | |
    | |  ██║  ██║██║█████╗  █████╗  ██████╔╝██║  ███╗  | |
    | |  ██║  ██║██║██╔══╝  ██╔══╝  ██╔══██╗██║   ██║  | |
    | |  ██████╔╝██║██║     ██║     ██║  ██║╚██████╔╝  | |
    | |  ╚═════╝ ╚═╝╚═╝     ╚═╝     ╚═╝  ╚═╝ ╚═════╝   | |
  __| |________________________________________________| |__
  __   ________________________________________________   __
    | |                                                | |
${ColourReset}

This is the main CMake file for DiFfRG.
It will configure and build all dependencies, including DiFfRG itself.
")

# The dependencies; see superbuild.cmake.
include(${CMAKE_CURRENT_SOURCE_DIR}/superbuild.cmake)

# ##############################################################################
# DiFfRG
# ##############################################################################

option(DiFfRG_TEST "Build DiFfRG tests" OFF)
option(DiFfRG_DOCUMENTATION "Build DiFfRG documentation" ON)

# Only forward a CUDA compiler if one was resolved; passing an empty value would pin the
# cache entry to nothing and break the sub-build's own detection.
if(CMAKE_CUDA_COMPILER)
  set(_diffrg_cuda_compiler_arg
      "-DCMAKE_CUDA_COMPILER:STRING=${CMAKE_CUDA_COMPILER}")
else()
  set(_diffrg_cuda_compiler_arg "")
endif()

# Optional Mathematica package destination, forwarded to the library build
# (which owns the install rule -- see DiFfRG/cmake/install_bundled.cmake).
if(DiFfRG_MATHEMATICA_INSTALL_DIR)
  set(_diffrg_mathematica_arg
      "-DDiFfRG_MATHEMATICA_INSTALL_DIR:PATH=${DiFfRG_MATHEMATICA_INSTALL_DIR}")
else()
  set(_diffrg_mathematica_arg "")
endif()

message("${BoldCyan}[8/8] Configuring DiFfRG...${ColourReset}")
ExternalProject_Add(
  DiFfRG
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/DiFfRG/
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  DEPENDS general_dep ${_boost_dep} deal.II_dep kokkos_dep autodiff_dep
  CMAKE_CACHE_ARGS
    # Forward the toolchain. Without this the sub-build silently falls back to CMake's
    # default compiler (/usr/bin/cc, /usr/bin/c++), which need not be -- and on a machine
    # with a newer system GCC usually is not -- the compiler deal.II and Kokkos were built
    # with. Every other dependency here already gets the compiler passed explicitly; DiFfRG
    # itself did not, so its toolchain depended on whether CC/CXX happened to be exported
    # in the caller's environment.
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    ${_diffrg_cuda_compiler_arg}
    ${_diffrg_mathematica_arg}
    -DMPI:BOOL=${MPI}
    -DHDF5:BOOL=ON
    -DBUNDLED_DIR:PATH=${DEPEND_INSTALL_PREFIX}
    -DBOOST_ROOT:PATH=${_boost_root}
    ${_boost_extra_args}
    -DTBB_DIR:STRING=${_tbb_dir}
    ${_hdf5_extra_args}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DNATIVE:BOOL=${NATIVE}
    -DMARCH:STRING=${MARCH}
    -DDiFfRG_TEST:BOOL=${DiFfRG_TEST}
    -DDiFfRG_DOCUMENTATION:BOOL=${DiFfRG_DOCUMENTATION}
    # lib/ never lib64/ (see general_dep): keeps DiFfRGConfig.cmake at
    # <prefix>/lib/cmake/DiFfRG on every distro, as the docs advertise.
    -DCMAKE_INSTALL_LIBDIR:STRING=lib
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})

# ##############################################################################
# Forwarding DiFfRG tests for CTest to discover them
# ##############################################################################

option(DiFfRG_TEST "Build tests" OFF)
if(DiFfRG_TEST)
  enable_testing()
  add_subdirectory(DiFfRG/tests)
endif()

# ##############################################################################
# Finish
# ##############################################################################

add_custom_target(
  finish ALL
  DEPENDS DiFfRG
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/DiFfRG/cmake/finish.sh
  VERBATIM)
