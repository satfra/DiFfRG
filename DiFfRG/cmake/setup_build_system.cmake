# ##############################################################################
# Setup directories
# ##############################################################################

# We need to find the bundle directory, which contains several external
# dependencies
if(${CMAKE_PROJECT_NAME} STREQUAL "DiFfRG")
  # If we are building DiFfRG as a standalone project, we need to set the base
  # directory
  set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  if(NOT DEFINED BUNDLED_DIR)
    set(BUNDLED_DIR ${BASE_DIR}/../external)
  endif()
else()
  # If we are building a DiFfRG-based project, we need to set the bundle
  # directory relative to the DiFfRG base directory
  set(BASE_DIR ${DiFfRG_BASE_DIR})
  set(BUNDLED_DIR ${BASE_DIR}/bundled)
endif()
message(STATUS "DiFfRG include directory: ${BASE_DIR}/include")

# No matter what we are building, we need to set the include directory
# include_directories(${BASE_DIR}/include)

# By default, we build in Release mode, i.e. if the user does not make any other
# choice. After all, even if the user is unaware of cmake build types, we want
# to provide optimal performance.
if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not set, defaulting to Release")
endif()

# Set some C++ compiler flags if they are not specified by the user. We simply
# propagate the base library flags to the user's project in that case.
if(NOT DEFINED CMAKE_CXX_FLAGS OR CMAKE_CXX_FLAGS STREQUAL "")
  set(CMAKE_CXX_FLAGS "${DiFfRG_CXX_FLAGS}")
  message(STATUS "CXX flags not set, defaulting to ${CMAKE_CXX_FLAGS}")
else()
  message(STATUS "CXX flags have been set to ${CMAKE_CXX_FLAGS}")
endif()

# CUDA flags are slightly more problematic, as we need them to be compatible
# with the ones used in the base library. Here, we always fix some flags that
# could lead to incompatibilities, and then add the user's flags. Also, by
# default, we use the highest instruction set available on the current machine.
if(NOT DEFINED CMAKE_CUDA_FLAGS OR CMAKE_CUDA_FLAGS STREQUAL "")
  set(CMAKE_CUDA_FLAGS
      "-arch=native --use_fast_math --split-compile=0 --threads=0 --gen-opt-lto"
  )

  message(STATUS "CUDA flags not set, defaulting to ${CMAKE_CUDA_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS
      "--split-compile=0 --threads=0 --gen-opt-lto --use_fast_math")

  # check if CMAKE_CUDA_FLAGS contains the arch keyword (i.e. the user has set
  # what architecture to use). If it does not, set it to the native.
  if(NOT CMAKE_CUDA_FLAGS MATCHES "-arch")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=native")
  endif()

  message(STATUS "CUDA flags have been set to ${CMAKE_CUDA_FLAGS}")
endif()

# By default, enable interprocedural optimization if it is supported.
if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT error)

  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(
      STATUS
        "Interprocedural optimization enabled. To disable, set CMAKE_INTERPROCEDURAL_OPTIMIZATION to FALSE."
    )
  else()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
  endif()
endif()

message(STATUS "Bundle directory: ${BUNDLED_DIR}")

# If CPM_SOURCE_CACHE is set to OFF, we set it to the default cache directory
if("${CPM_SOURCE_CACHE}" STREQUAL "OFF" OR NOT DEFINED CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE $ENV{HOME}/.cache/CPM)
endif()
message(STATUS "CPM source cache directory: ${CPM_SOURCE_CACHE}")

# Get CPM for package management
list(APPEND CMAKE_MODULE_PATH "${BASE_DIR}/cmake")
include(${BASE_DIR}/cmake/CPM.cmake)

# ##############################################################################
# Set standard and language
# ##############################################################################

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
else()
  if(CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "C++ standard must be at least 20")
  endif()
endif()

set(CMAKE_CXX_STANDARD_REQUIRED On)
enable_language(CXX)

# ##############################################################################
# Find packages
# ##############################################################################

# Use CCACHE if not set
if(NOT DEFINED USE_CCACHE)
  set(USE_CCACHE ON)
endif()
# Add CCache.cmake for faster builds
cpmaddpackage(NAME Ccache.cmake GITHUB_REPOSITORY TheLartians/Ccache.cmake
              VERSION 1.2)

# Find deal.II
find_package(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_DIR}
             ${BUNDLED_DIR}/dealii_install)
deal_ii_initialize_cached_variables()

# Find TBB
find_package(TBB 2022.0.0 REQUIRED HINTS ${BUNDLED_DIR}/oneTBB_install)
message(STATUS "TBB dir: ${TBB_DIR}")

# Find Boost
find_package(Boost 1.80 REQUIRED HINTS ${BUNDLED_DIR}/boost_install
             COMPONENTS thread random iostreams math serialization system)
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

# Find Eigen3
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
  add_library(Eigen3 INTERFACE IMPORTED)
  target_include_directories(Eigen3 INTERFACE ${Eigen3_SOURCE_DIR})
  add_library(Eigen3::Eigen ALIAS Eigen3)
endif()

# Find GSL
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

# Find rapidcsv
cpmaddpackage(
  NAME
  rapidcsv
  GITHUB_REPOSITORY
  d99kris/rapidcsv
  VERSION
  8.84
  DOWNLOAD_ONLY
  YES)
include_directories(SYSTEM ${rapidcsv_SOURCE_DIR}/src)

# Find thread-pool
cpmaddpackage(
  NAME
  thread-pool
  GITHUB_REPOSITORY
  bshoshany/thread-pool
  VERSION
  5.0.0
  DOWNLOAD_ONLY
  YES)
include_directories(SYSTEM ${thread-pool_SOURCE_DIR}/include)

# Check if CUDA is available
if(NOT DEFINED USE_CUDA)
  set(USE_CUDA ON)
endif()

if(DEFINED DiFfRG_USE_CUDA)
  if(DiFfRG_USE_CUDA)
    if(NOT USE_CUDA)
      message(
        WARNING "Trying to force CUDA on, as DiFfRG has been built with CUDA")
    endif()
    set(USE_CUDA ON)
  else()
    if(USE_CUDA)
      message(WARNING "Forcing CUDA off, as DiFfRG has been built without CUDA")
    endif()
    set(USE_CUDA OFF)
  endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)

  set(CUDA_NVCC_FLAGS
      -Xcudafe
      "--diag_suppress=20208 --diag_suppress=20012"
      -lineinfo
      --default-stream
      per-thread
      --expt-relaxed-constexpr
      --generate-line-info
      -O3
      -rdc=true
      ${CUDA_NVCC_FLAGS}
      $ENV{CUDA_NVCC_FLAGS})
  message(STATUS "flags for CUDA: ${CUDA_NVCC_FLAGS}")

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  function(setup_target TARGET)
    deal_ii_setup_target(${TARGET})

    target_compile_definitions(${TARGET} PUBLIC USE_CUDA)

    set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_compile_options(
      ${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>")
    target_compile_options(${TARGET} PRIVATE -Wno-misleading-indentation)

    # Check if the target is DiFfRG
    if(${TARGET} STREQUAL "DiFfRG")
      target_include_directories(${TARGET} PUBLIC ${autodiff_SOURCE_DIR})
    else()
      target_link_libraries(${TARGET} autodiff::autodiff)
    endif()

    target_link_libraries(${TARGET} GSL::gsl)
    target_link_libraries(${TARGET} Eigen3)
    target_link_libraries(${TARGET} spdlog::spdlog)
    target_link_libraries(${TARGET} rmm::rmm)
    target_link_libraries(${TARGET} ${Boost_LIBRARIES})
    target_link_libraries(${TARGET} TBB::tbb)
  endfunction()

  message(STATUS "CUDA support enabled.")
else()
  set(USE_CUDA OFF)
  function(setup_target TARGET)
    deal_ii_setup_target(${TARGET})

    set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_compile_options(${TARGET} PRIVATE -Wno-misleading-indentation)

    # Check if the target is DiFfRG
    if(${TARGET} STREQUAL "DiFfRG")
      target_include_directories(${TARGET} PUBLIC ${autodiff_SOURCE_DIR})
    else()
      target_link_libraries(${TARGET} autodiff::autodiff)
    endif()

    target_link_libraries(${TARGET} GSL::gsl)
    target_link_libraries(${TARGET} Eigen3)
    target_link_libraries(${TARGET} TBB::tbb)
    target_link_libraries(${TARGET} spdlog::spdlog)
    target_link_libraries(${TARGET} ${Boost_LIBRARIES})
  endfunction()

  message(STATUS "CUDA support disabled.")
endif()

# Find autodiff
cpmaddpackage(
  NAME
  autodiff
  GITHUB_REPOSITORY
  autodiff/autodiff
  GIT_TAG
  v1.1.0
  OPTIONS
  "AUTODIFF_BUILD_TESTS OFF"
  "AUTODIFF_BUILD_EXAMPLES OFF"
  "AUTODIFF_BUILD_DOCS OFF"
  "AUTODIFF_BUILD_PYTHON OFF"
  "Eigen3_DIR ${Eigen3_BINARY_DIR}")

# This is for spdlog usage. We need to hide the local spdlog installation,
# otherwise we will have problems with the linking process.
set(CMAKE_DISABLE_FIND_PACKAGE_spdlog TRUE)

if(USE_CUDA)
  cpmaddpackage(NAME CCCL GITHUB_REPOSITORY "nvidia/cccl" GIT_TAG "v2.7.0")

  cpmaddpackage(
    NAME
    rmm
    VERSION
    25.02.00a
    GITHUB_REPOSITORY
    rapidsai/rmm
    SYSTEM
    Off
    OPTIONS
    "BUILD_TESTS OFF")
endif()

cpmaddpackage(
  NAME
  spdlog
  GITHUB_REPOSITORY
  gabime/spdlog
  VERSION
  1.14.1
  OPTIONS
  "CMAKE_BUILD_TYPE Release"
  "SPDLOG_INSTALL ON")

# ##############################################################################
# Helper functions
# ##############################################################################

function(setup_application TARGET)
  setup_target(${TARGET})
  target_link_libraries(${TARGET} DiFfRG::DiFfRG)
  target_compile_options(${TARGET} PRIVATE -Wno-unused-parameter)
endfunction()

# Keep track of the flow folders to avoid adding the same folder multiple times
set(FLOW_FOLDERS "")

function(add_flows TARGET DIRECTORY)
  # If the folder is not already in the list, add it
  if(NOT "${FLOW_FOLDERS}" IN_LIST "${DIRECTORY}")
    list(APPEND FLOW_FOLDERS ${DIRECTORY})
    add_subdirectory(${DIRECTORY})
  endif()

  # Add the flow sources to the target
  target_sources(${TARGET} PUBLIC ${flow_sources})

  # Make the list of flow folders available to the parent scope
  set(${FLOW_FOLDERS}
      "${${FLOW_FOLDERS}}"
      PARENT_SCOPE)
endfunction()
