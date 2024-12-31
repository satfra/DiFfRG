# ##############################################################################
# CMakeLists.txt for the DiFfRG library
# ##############################################################################

cmake_minimum_required(VERSION 3.26.4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(DiFfRG LANGUAGES CXX)

# ##############################################################################
# Documentation
# ##############################################################################

option(DiFfRG_BUILD_DOCUMENTATION "Build documentation" ON)

# check if Doxygen is installed
find_package(Doxygen)
if(DiFfRG_BUILD_DOCUMENTATION AND DOXYGEN_FOUND)
  # set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/documentation/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  message("Doxygen build started")

  add_custom_target(
    documentation
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/documentation
          DESTINATION ${CMAKE_INSTALL_PREFIX}) # target directory
  install(
    CODE "file(CREATE_LINK ${CMAKE_INSTALL_PREFIX}/documentation/html/index.html ${CMAKE_INSTALL_PREFIX}/Documentation.html SYMBOLIC)"
  )

elseif(DiFfRG_BUILD_DOCUMENTATION AND DOXYGEN_FOUND)
  message("Doxygen needs to be installed to generate the doxygen documentation")
endif()

if(DiFfRG_ONLY_DOCUMENTATION)
  return()
endif()

# ##############################################################################
# Setup build system : Libraries
# ##############################################################################

add_subdirectory(${CMAKE_SOURCE_DIR}/cmake)
include(${CMAKE_SOURCE_DIR}/cmake/setup_build_system.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/setup_targets.cmake)

# ##############################################################################
# DiFfRG library and models
# ##############################################################################

include_directories(include)

add_subdirectory(include)
add_subdirectory(src)

# ##############################################################################
# Testing
# ##############################################################################

option(DiFfRG_BUILD_TESTS "Build documentation" ON)

if(DiFfRG_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ##############################################################################
# Packaging and installation
# ##############################################################################

include(cmake/install_bundled.cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  "Config.cmake.in" "DiFfRGConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DiFfRG
  PATH_VARS CMAKE_INSTALL_LIBDIR)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/DiFfRGConfigVersion.cmake
  VERSION 0.1.0
  COMPATIBILITY SameMajorVersion)

export(
  EXPORT DiFfRGTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/DiFfRGTargets.cmake"
  NAMESPACE DiFfRG::)

# Install Config and ConfigVersion files
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/DiFfRGConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/DiFfRGConfigVersion.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/DiFfRGTargets.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/DiFfRG")

# ##############################################################################
# licenses
# ##############################################################################

set(CPM_PACKAGE_deal.II_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/dealii)
set(CPM_PACKAGE_sundials_SOURCE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/sundials)
set(CPM_PACKAGE_boost_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../external/boost)

# Add CPMLicenses for disclaimers
cpmaddpackage(NAME CPMLicenses.cmake GITHUB_REPOSITORY
              cpm-cmake/CPMLicenses.cmake VERSION 0.0.5)

message("deal.II source directory: ${deal.II_SOURCE_DIR}")

message("packages: ${CPM_PACKAGES};deal.II;sundials;boost")

cpm_licenses_create_disclaimer_target(
  write-licenses "${CMAKE_CURRENT_SOURCE_DIR}/../third_party.txt"
  "${CPM_PACKAGES};deal.II;sundials;boost")

# ##############################################################################
# other scripts
# ##############################################################################

cpmaddpackage("gh:StableCoder/cmake-scripts#24.04")

include(${cmake-scripts_SOURCE_DIR}/formatting.cmake)

file(GLOB_RECURSE headers CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh")
file(GLOB_RECURSE sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cc")
file(GLOB_RECURSE cuda_sources CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cu")

clang_format(TARGET_NAME ${headers} ${sources} ${cuda_sources})

file(GLOB_RECURSE CMAKE_FILES CMakeLists.txt)

cmake_format(TARGET_NAME ${CMAKE_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/setup_build_system.cmake)
