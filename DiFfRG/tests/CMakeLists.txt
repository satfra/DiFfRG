# ##############################################################################
# Setup additional include directories and Catch2
# ##############################################################################

include_directories(./)
# include(../cmake/setup_build_system.cmake)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.8.1)
FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)

# When building in-tree (as a subdirectory of the library), the DiFfRG target
# already exists. Only call find_package for out-of-tree builds.
if(NOT TARGET DiFfRG)
  find_package(DiFfRG HINTS ${CMAKE_PREFIX_PATH})
else()
  if(NOT TARGET DiFfRG::DiFfRG_boost_json)
    add_library(DiFfRG::DiFfRG_boost_json ALIAS DiFfRG_boost_json)
  endif()
endif()

function(setup_test SOURCE)
  # Create a test executable from the source file, removing the extension
  get_filename_component(TARGET ${SOURCE} NAME_WE)
  add_executable(${TARGET} ${SOURCE})
  target_link_libraries(${TARGET} PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
  catch_discover_tests(${TARGET}
    ADD_TAGS_AS_LABELS
    PROPERTIES TIMEOUT 3600)
endfunction()

# Same as setup_test, but additionally registers the test under mpirun for each rank count in
# MPI_TEST_RANKS. Without this an "MPI test" only ever runs on one rank, which is exactly the
# configuration in which none of the distribution code executes.
set(MPI_TEST_RANKS "2;3;4" CACHE STRING "Rank counts at which MPI tests are additionally run")

function(setup_mpi_test SOURCE)
  get_filename_component(TARGET ${SOURCE} NAME_WE)
  add_executable(${TARGET} ${SOURCE})
  target_link_libraries(${TARGET} PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
  catch_discover_tests(${TARGET}
    ADD_TAGS_AS_LABELS
    PROPERTIES TIMEOUT 3600)

  if(NOT DiFfRG_MPI)
    return()
  endif()
  find_package(MPI QUIET)
  if(NOT MPIEXEC_EXECUTABLE)
    message(STATUS "MPI tests for ${TARGET}: no mpiexec found, single-rank only")
    return()
  endif()

  foreach(NRANKS IN LISTS MPI_TEST_RANKS)
    add_test(NAME ${TARGET}_mpi_${NRANKS}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NRANKS}
              ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${TARGET}> ${MPIEXEC_POSTFLAGS})
    # Ranks sharing one GPU is the normal case on a workstation; it is slow, not wrong.
    set_tests_properties(${TARGET}_mpi_${NRANKS} PROPERTIES
      TIMEOUT 3600
      LABELS "mpi"
      RUN_SERIAL TRUE)
  endforeach()
endfunction()

function(setup_benchmark SOURCE)
  # Create a test executable from the source file, removing the extension
  get_filename_component(TARGET ${SOURCE} NAME_WE)
  add_executable(${TARGET} ${SOURCE})
  target_link_libraries(${TARGET} PRIVATE DiFfRG::DiFfRG Catch2::Catch2WithMain)
endfunction()

include(CTest)
include(Catch)

# ##############################################################################
# Build-system tests
# ##############################################################################
#
# The DiFfRG_CUDA_ARCH helpers decide which nvcc architecture flags the library
# and every application are compiled with, and getting them wrong is invisible
# until an application either fails to compile or silently falls back to
# JIT-compiling every kernel. They are pure CMake, so `cmake -P` exercises them
# without a GPU or a configured build tree.
foreach(_case normalize flags replace)
  add_test(NAME cuda_arch_${_case}
           COMMAND ${CMAKE_COMMAND} -DCASE=${_case} -P
                   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_cuda_arch.cmake)
endforeach()
# Nonsense must abort rather than being passed on to nvcc.
foreach(_bad sm_90 9 abc)
  add_test(NAME cuda_arch_reject_${_bad}
           COMMAND ${CMAKE_COMMAND} -DCASE=invalid -DBAD=${_bad} -P
                   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_cuda_arch.cmake)
  set_tests_properties(cuda_arch_reject_${_bad} PROPERTIES WILL_FAIL TRUE)
endforeach()

# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(common)
add_subdirectory(discretization)
add_subdirectory(hdf5lib)
add_subdirectory(model)
add_subdirectory(physics)
add_subdirectory(timestepping)
add_subdirectory(migration)
