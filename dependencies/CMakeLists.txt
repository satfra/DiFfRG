cmake_minimum_required(VERSION 3.27)

project(DiFfRG_dependencies)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_BUILD_TYPE Release)

# Set up external dependencies

# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.8/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# Find out which backends are available
include(CheckLanguage)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(HAVE_CUDA "ON")
  set(HAVE_HIP "OFF")
else()
  set(HAVE_CUDA "OFF")
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    set(HAVE_HIP "ON")
  else()
    set(HAVE_HIP "OFF")
  endif()
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
endif()
check_language(OpenMP)
find_package( OpenMP )
if ( OpenMP_CXX_FOUND )
  set(HAVE_OPENMP "ON")
  set(HAVE_THREADS "OFF")
else()
  set(HAVE_OPENMP "OFF")
  set(HAVE_THREADS "ON")
endif()

# kokkos
cpmaddpackage(
  NAME
  Kokkos
  VERSION
  4.6.01
  OPTIONS
  "Kokkos_ENABLE_CUDA ${HAVE_CUDA}"
  "Kokkos_ENABLE_CUDA_CONSTEXPR ${HAVE_CUDA}"
  "Kokkos_ENABLE_HIP ${HAVE_HIP}"
  "Kokkos_ENABLE_OPENMP ${HAVE_OPENMP}"
  "Kokkos_ENABLE_THREADS ${HAVE_THREADS}"
  "Kokkos_ENABLE_SERIAL ON"
  "Kokkos_ENABLE_TESTS OFF"
  "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}"
  GITHUB_REPOSITORY
  kokkos/kokkos
  GIT_TAG
  4.6.01)

# oneTBB
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

# sundials
cpmaddpackage(
  NAME
  sundials
  VERSION
  6.7.0
  OPTIONS
  "ENABLE_MPI OFF"
  "CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}"
  "EXAMPLES_ENABLE_C OFF"
  "EXAMPLES_ENABLE_CXX OFF"
  GITHUB_REPOSITORY
  LLNL/sundials
  GIT_TAG
  v6.7.0)

# boost
set(TRY_BOOST_VERSION "1.81.0")

set(PARALLEL_JOBS 8)

message(STATUS "Downloading Boost version ${TRY_BOOST_VERSION}...")
execute_process (
    COMMAND bash -c "git clone --depth 1 --branch boost-1.81.0 https://github.com/boostorg/boost.git"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    OUTPUT_QUIET
)
message(STATUS "Downloading Boost libraries... (this may take a while)")
execute_process (
    COMMAND bash -c "git submodule update --init --depth 1 --recursive --jobs ${PARALLEL_JOBS}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/
    OUTPUT_QUIET
)

add_custom_target(
  build_boost_boostrap
    COMMAND
      bash -c "./bootstrap.sh --prefix=${CMAKE_INSTALL_PREFIX} &> /dev/null"
    VERBATIM
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/
)
add_custom_target(
  build_boost_b2
    COMMAND
      bash -c "./b2 --build-dir=${CMAKE_CURRENT_BINARY_DIR}/boost-build \
        --prefix=${CMAKE_INSTALL_PREFIX} \
        --with-headers \
        --with-iostreams \
        --with-json \
        --with-math \
        --with-serialization \
        --with-system \
        --with-random \
        --with-thread \
        -j ${PARALLEL_JOBS} \
        cxxflags=\"-std=c++20\" \
        install"
    VERBATIM
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/boost/
    DEPENDS
      build_boost_boostrap
)
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target build_boost_b2)")