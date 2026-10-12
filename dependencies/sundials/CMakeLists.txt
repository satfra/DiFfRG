cmake_minimum_required(VERSION 3.20)

project(sundials_dep)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

# if MPI is not set, set it to OFF
if(NOT DEFINED MPI)
  set(MPI
      OFF
      CACHE BOOL "Enable MPI support")
endif()

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.8/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH
    SHA256=78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
if("${CPM_SOURCE_CACHE}" STREQUAL "OFF" OR NOT DEFINED CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE $ENV{HOME}/.cache/CPM)
endif()

# sundials
message(STATUS "Adding SUNDIALS...")
cpmaddpackage(
  NAME
  sundials
  VERSION
  7.7.0
  OPTIONS
  "ENABLE_MPI ${MPI}"
  "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}"
  "EXAMPLES_ENABLE_C OFF"
  "EXAMPLES_ENABLE_CXX OFF"
  GITHUB_REPOSITORY
  LLNL/sundials
  GIT_TAG
  v7.7.0)

# DiFfRG's IDA driver continues IDA warm across mesh adaptations (timestepping/ida_driver.cc), which needs IDA's
# internal state: install idas_impl.h and the private headers it includes.
install(
  FILES ${sundials_SOURCE_DIR}/src/idas/idas_impl.h
        ${sundials_SOURCE_DIR}/src/sundials/sundials_logger_impl.h
        ${sundials_SOURCE_DIR}/src/sundials/sundials_macros.h
        ${sundials_SOURCE_DIR}/src/sundials/sundials_hashmap_impl.h
        ${sundials_SOURCE_DIR}/src/sundials/sundials_utils.h
  DESTINATION include/sundials_internal)
install(FILES ${sundials_SOURCE_DIR}/src/sundials/stl/sunstl_vector.h DESTINATION include/sundials_internal/stl)
