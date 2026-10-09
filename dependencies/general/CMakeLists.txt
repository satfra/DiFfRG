cmake_minimum_required(VERSION 3.20)

project(general_dep)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

# if MPI is not set, set it to OFF
if(NOT DEFINED MPI)
  set(MPI
      OFF
      CACHE BOOL "Enable MPI support")
endif()

# Set up external dependencies

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

# oneTBB — skipped when the top-level build resolved a system TBB
# (GENERAL_BUILD_TBB=OFF). Defaults ON so a standalone general_dep build still
# produces TBB.
option(GENERAL_BUILD_TBB "Build the bundled oneTBB" ON)
if(GENERAL_BUILD_TBB)
  message(STATUS "Adding oneTBB...")
  cpmaddpackage(
    NAME
    oneTBB
    VERSION
    2022.1.0
    OPTIONS
    "CMAKE_CXX_STANDARD 20"
    "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}"
    "TBB_TEST OFF"
    GITHUB_REPOSITORY
    uxlfoundation/oneTBB
    GIT_TAG
    v2022.1.0)
else()
  message(STATUS "Skipping bundled oneTBB build (using system TBB).")
endif()

option(BUILD_OpenBLAS "Build OpenBLAS" ON)

if(BUILD_OpenBLAS)
  message(STATUS "Building OpenBLAS.")

  # OpenBlas
  include(ProcessorCount)
  ProcessorCount(N_PROCESSORS)

  cpmaddpackage(
    NAME
    OpenBLAS
    GITHUB_REPOSITORY
    OpenMathLib/OpenBLAS
    VERSION
    0.3.30
    OPTIONS
    "CMAKE_BUILD_TYPE Release"
    "BUILD_SHARED_LIBS ON"
    "LIBNAMESUFFIX DiFfRG"
    "USE_OPENMP OFF"
    "NUM_PARALLEL ${N_PROCESSORS}"
    "BUILD_TESTING OFF"
    "DYNAMIC_ARCH OFF"
    "NUM_THREADS=64")
else()
  message(STATUS "Skipping OpenBLAS build.")
endif()

# Get Eigen3
message(STATUS "Adding Eigen3...")
cpmaddpackage(
  NAME
  Eigen3
  VERSION
  3.4.0
  URL
  https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  # Eigen's CMakelists are not intended for library use
  DOWNLOAD_ONLY
  YES)
if(Eigen3_ADDED)
  add_library(Eigen3 INTERFACE)
  target_include_directories(
    Eigen3 INTERFACE $<BUILD_INTERFACE:${Eigen3_SOURCE_DIR}>
                     $<INSTALL_INTERFACE:include>)
  add_library(Eigen3::Eigen ALIAS Eigen3)

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/Eigen3ConfigVersion.cmake"
    VERSION 3.4.0
    COMPATIBILITY AnyNewerVersion)

  install(
    TARGETS Eigen3
    EXPORT Eigen3Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib COMPONENT Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib COMPONENT Development
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin COMPONENT Runtime
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
                  COMPONENT Development
    BUNDLE DESTINATION ${CMAKE_INSTALL_PREFIX}/bin COMPONENT Runtime)

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Eigen3Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Eigen3Config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/Eigen3)

  install(EXPORT Eigen3Targets
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/Eigen3)
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/Eigen3ConfigVersion.cmake"
                "${CMAKE_CURRENT_BINARY_DIR}/Eigen3Config.cmake"
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/Eigen3)
  install(DIRECTORY ${Eigen3_SOURCE_DIR}/ DESTINATION include)
endif()

# Get GSL
message(STATUS "Adding GSL...")
find_package(GSL)
if(NOT GSL_FOUND)
  cpmfindpackage(
    NAME
    gsl
    GITHUB_REPOSITORY
    ampl/gsl
    VERSION
    2.5.0
    OPTIONS
    "GSL_DISABLE_TESTS 1"
    "DOCUMENTATION OFF")
  add_library(GSL::gsl ALIAS gsl)
else()
  message(STATUS "GSL found: ${GSL_INCLUDE_DIR}")
endif()

# Get spdlog
message(STATUS "Adding spdlog...")
cpmaddpackage(
  NAME
  spdlog
  GITHUB_REPOSITORY
  gabime/spdlog
  VERSION
  1.17.0
  OPTIONS
  "CMAKE_BUILD_TYPE Release"
  "SPDLOG_INSTALL ON"
  "DSPDLOG_BUILD_SHARED=OFF")

