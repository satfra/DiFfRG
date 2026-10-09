cmake_minimum_required(VERSION 3.20)

project(autodiff_dep)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

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

# Get autodiff
cpmaddpackage(
  NAME
  autodiff
  GITHUB_REPOSITORY
  autodiff/autodiff
  GIT_TAG
  v1.1.0
  PATCHES
  "autodiff.patch"
  OPTIONS
  "CMAKE_CUDA_ARCHITECTURES native"
  "AUTODIFF_BUILD_TESTS OFF"
  "AUTODIFF_BUILD_EXAMPLES OFF"
  "AUTODIFF_BUILD_DOCS OFF"
  "AUTODIFF_BUILD_PYTHON OFF"
  "CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}"
  "Eigen3_DIR ${CMAKE_INSTALL_PREFIX}")
