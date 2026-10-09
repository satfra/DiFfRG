cmake_minimum_required(VERSION 3.20)

project(hdf_dep)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

# if MPI is not set, set it to OFF
if(NOT DEFINED MPI)
  set(MPI
      OFF
      CACHE BOOL "Enable MPI support")
endif()

# Set up external dependencies

# download HDF5, if it is not already downloaded
set(HDF5_TARBALL "${CMAKE_CURRENT_BINARY_DIR}/hdf5.tar.gz")
set(HDF5_TARBALL_VALID FALSE)
if(EXISTS ${HDF5_TARBALL})
  file(SIZE ${HDF5_TARBALL} HDF5_TARBALL_SIZE)
  if(HDF5_TARBALL_SIZE GREATER 0)
    set(HDF5_TARBALL_VALID TRUE)
  else()
    file(REMOVE ${HDF5_TARBALL})
  endif()
endif()

if(NOT HDF5_TARBALL_VALID)
  message(STATUS "Downloading hdf5...")
  file(
    DOWNLOAD https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5.tar.gz
    ${HDF5_TARBALL}
    STATUS HDF5_DOWNLOAD_STATUS
    LOG HDF5_DOWNLOAD_LOG)
  list(GET HDF5_DOWNLOAD_STATUS 0 HDF5_DOWNLOAD_CODE)
  if(NOT HDF5_DOWNLOAD_CODE EQUAL 0)
    file(REMOVE ${HDF5_TARBALL})
    message(FATAL_ERROR "Failed to download HDF5: ${HDF5_DOWNLOAD_STATUS}\n${HDF5_DOWNLOAD_LOG}")
  endif()
  file(SIZE ${HDF5_TARBALL} HDF5_TARBALL_SIZE)
  if(NOT HDF5_TARBALL_SIZE GREATER 0)
    file(REMOVE ${HDF5_TARBALL})
    message(FATAL_ERROR "Failed to download HDF5: archive is empty.")
  endif()
else()
  message(STATUS "hdf5 already downloaded.")
endif()

# extract hdf5, if it is not already extracted
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/hdf5-2.0.0/)
  message(STATUS "Extracting hdf5...")
  execute_process(COMMAND tar -xzf hdf5.tar.gz
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
  message(STATUS "hdf5 already extracted.")
endif()

# configure hdf5 — static-only, minimal-component build
execute_process(
  COMMAND
    cmake -S hdf5-2.0.0 -B ./build
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_STATIC_LIBS=ON
    -DONLY_SHARED_LIBS=OFF
    -DHDF5_EXTERNALLY_CONFIGURED=0
    -DHDF5_ENABLE_PARALLEL=${MPI}
    -DHDF5_BUILD_EXAMPLES=OFF
    -DHDF5_BUILD_TOOLS=OFF
    -DHDF5_BUILD_UTILS=OFF
    -DHDF5_BUILD_HL_LIB=OFF
    -DHDF5_BUILD_CPP_LIB=OFF
    -DHDF5_BUILD_FORTRAN=OFF
    -DHDF5_BUILD_JAVA=OFF
    -DHDF5_BUILD_DOC=OFF
    -DHDF5_ENABLE_SZIP_SUPPORT=OFF
    -DHDF5_ENABLE_SZIP_ENCODING=OFF
    -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF
    -DHDF5_ENABLE_THREADSAFE=OFF
    -DHDF5_ALLOW_EXTERNAL_SUPPORT=NO
    -DBUILD_TESTING=OFF
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/hdf5_configure.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/hdf5_configure.log
  RESULT_VARIABLE HDF5_CONFIGURE_RESULT)
if(NOT HDF5_CONFIGURE_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to configure HDF5. See log: ${CMAKE_CURRENT_BINARY_DIR}/hdf5_configure.log")
endif()

# briefly get the number of processors for parallel build
include(ProcessorCount)
ProcessorCount(PROCESSOR_COUNT)

# build hdf5
execute_process(
  COMMAND make -j ${PROCESSOR_COUNT}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/hdf5_build.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/hdf5_build.log
  RESULT_VARIABLE HDF5_BUILD_RESULT)
if(NOT HDF5_BUILD_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to build HDF5. See log: ${CMAKE_CURRENT_BINARY_DIR}/hdf5_build.log")
endif()

# install hdf5
execute_process(
  COMMAND make install
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/build
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/hdf5_install.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/hdf5_install.log
  RESULT_VARIABLE HDF5_INSTALL_RESULT)
if(NOT HDF5_INSTALL_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to install HDF5. See log: ${CMAKE_CURRENT_BINARY_DIR}/hdf5_install.log")
endif()
