# ##############################################################################
# Setup directories
# ##############################################################################

if(${CMAKE_PROJECT_NAME} STREQUAL "DiFfRG")
  set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  if (NOT DEFINED BUNDLED_DIR)
    set(BUNDLED_DIR ${BASE_DIR}/../external)
  endif()
else()
  set(BASE_DIR ${DiFfRG_BASE_DIR})
  set(BUNDLED_DIR ${BASE_DIR}/bundled)
endif()

message("Bundle directory: ${BUNDLED_DIR}")

# ##############################################################################
# Set standard and language
# ##############################################################################

if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
else()
  if (CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "C++ standard must be at least 20")
  endif()
endif()

set(CMAKE_CXX_STANDARD_REQUIRED On)
enable_language(CXX)

# ##############################################################################
# Find packages
# ##############################################################################

find_package(deal.II 9.5.0 REQUIRED HINTS ${DEAL_II_DIR}
             ${BUNDLED_DIR}/dealii_install)
deal_ii_initialize_cached_variables()

find_package(TBB REQUIRED)
message(STATUS "TBB dir: ${TBB_DIR}")
link_libraries(TBB::tbb)

find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

find_package(Boost 1.80 REQUIRED HINTS ${BUNDLED_DIR}/boost_install COMPONENTS thread random iostreams math serialization system)
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
link_libraries(${Boost_LIBRARIES})

find_package(autodiff REQUIRED PATHS ${BUNDLED_DIR}/autodiff_install)
link_libraries(autodiff::autodiff)

set(Eigen3_DIR ${BUNDLED_DIR}/eigen_install/share/eigen3/cmake)
find_package(Eigen3 3.4 REQUIRED)
link_libraries(Eigen3::Eigen)
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(GSL REQUIRED)
link_libraries(GSL::gsl)
message(STATUS "GSL include dir: ${GSL_INCLUDE_DIR}")
include_directories(SYSTEM ${GSL_INCLUDE_DIR}) 

include_directories(SYSTEM ${BUNDLED_DIR}/rapidcsv_install/include)
include_directories(SYSTEM ${BUNDLED_DIR}/qmc_install)
include_directories(SYSTEM ${BUNDLED_DIR}/thread-pool_install/include)

# Check if CUDA is available
if(NOT DEFINED USE_CUDA)
  set(USE_CUDA ON)
endif()

if(DEFINED DiFfRG_USE_CUDA)
  if(DiFfRG_USE_CUDA)
    if(NOT USE_CUDA)
      message(WARNING "Trying to force CUDA on, as DiFfRG has been built with CUDA")
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
      ${CUDA_NVCC_FLAGS}
      $ENV{CUDA_NVCC_FLAGS})
  message(STATUS "flags for CUDA: ${CUDA_NVCC_FLAGS}")

  function(setup_target TARGET)
    target_compile_definitions(${TARGET} PUBLIC USE_CUDA)
    set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    deal_ii_setup_target(${TARGET})
    #set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES native)
    target_compile_options(
      ${TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>")
    target_compile_options(${TARGET} PRIVATE -Wno-misleading-indentation)
  endfunction()

  message(STATUS "CUDA support enabled.")
else()
  set(USE_CUDA OFF)
  function(setup_target TARGET)
    deal_ii_setup_target(${TARGET})
    set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
    target_compile_options(${TARGET} PRIVATE -Wno-misleading-indentation)
  endfunction()

  message(STATUS "CUDA support disabled.")
endif()

if(USE_CUDA)
  MESSAGE("rmm in ${BUNDLED_DIR}/rmm_build")
  find_package(rmm REQUIRED PATHS ${BUNDLED_DIR}/rmm_build)
  link_libraries(rmm::rmm)
endif()

find_package(spdlog REQUIRED PATHS ${BUNDLED_DIR}/spdlog_install)
link_libraries(spdlog::spdlog)

# ##############################################################################
# Helper functions
# ##############################################################################

function(setup_application TARGET)
  setup_target(${TARGET})
  target_link_libraries(${TARGET} DiFfRG::DiFfRG)
  target_compile_options(${TARGET} PRIVATE -Wno-unused-parameter)
endfunction()

function(setup_test TARGET)
  setup_target(${TARGET})
  target_link_libraries(${TARGET} DiFfRG::DiFfRG Catch2::Catch2WithMain)
  catch_discover_tests(${TARGET})
endfunction()

function(setup_benchmark TARGET)
  setup_target(${TARGET})
  target_link_libraries(${TARGET} DiFfRG::DiFfRG Catch2::Catch2WithMain)
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
  target_sources(${TARGET} PRIVATE ${flow_sources})

  # Make the list of flow folders available to the parent scope
  set(${FLOW_FOLDERS} "${${FLOW_FOLDERS}}" PARENT_SCOPE)
endfunction()
