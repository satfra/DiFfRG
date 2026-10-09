# ##############################################################################
# Tests for the header-only hdf5lib wrapper.
# ##############################################################################

# These tests only need the wrapper headers (under DiFfRG/include/hdf5lib) and
# the HDF5 C library. They deliberately do not depend on the rest of DiFfRG, so
# they can validate the wrapper in isolation.

add_executable(hdf5lib_wrapper wrapper.cc)
target_include_directories(hdf5lib_wrapper PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
# DiFfRG_HDF5_LIBRARIES is resolved by find_package(HDF5) in
# setup_build_system.cmake (bundled static -> hdf5-static; system -> hdf5-shared /
# hdf5::hdf5 / HDF5::HDF5).
target_link_libraries(hdf5lib_wrapper PRIVATE
  Catch2::Catch2WithMain
  ${DiFfRG_HDF5_LIBRARIES})
target_include_directories(hdf5lib_wrapper PRIVATE ${HDF5_INCLUDE_DIRS})
catch_discover_tests(hdf5lib_wrapper)
