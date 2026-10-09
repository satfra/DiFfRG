cmake_minimum_required(VERSION 3.20)

project(boost_dep NONE)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_BUILD_TYPE Release)

set(TRY_BOOST_VERSION "1.81.0")
set(TRY_BOOST_VERSION_NUMBER 108100)

# The Boost source ships as one tracked archive rather than an unpacked tree, so
# a checkout costs one file instead of ~74 000 -- which matters both for `git
# status` and for collaborators on slow (network / FUSE) filesystems. Nothing is
# ever downloaded, so offline and firewalled cluster builds keep working.
#
# The archive is upstream boost_1_81_0.tar.gz -- sha256
# 205666dea9f6a7cfed87c7a6dfbeb52a2c1b9de55712c9c1a87735d7181452b6, the digest
# published on boost.org -- with documentation, tests and examples removed
# (74 085 -> 19 498 files, 696 -> 183 MiB). Because it is no longer byte-
# identical to upstream, the upstream digest cannot authenticate it; the recipe
# below is deterministic instead, so anyone can regenerate the archive and get
# the same sha256:
#
#   tar xf boost_1_81_0.tar.gz -C $W && cd $W/boost_1_81_0
#   rm -rf doc more
#   rm -rf libs/*/doc   libs/*/test   libs/*/example   libs/*/examples
#   rm -rf libs/*/*/doc libs/*/*/test libs/*/*/example libs/*/*/examples
#   rm -rf tools/*/doc  tools/*/test  tools/*/example  tools/*/examples
#   cd $W && tar --sort=name --owner=0 --group=0 --numeric-owner -cf - \
#     boost_1_81_0 | xz -9e -T1 > boost_1_81_0-diffrg-slim.tar.xz
#
# Only doc/test/example trees are dropped; boost/ headers, libs/*/src and every
# libs/*/build Jamfile -- all b2 actually consumes -- are untouched.
#
# The archive's single top-level entry is boost_1_81_0/, so unpacking it in
# ${CMAKE_CURRENT_BINARY_DIR} yields exactly ${BOOST_SOURCE_DIR}.
set(BOOST_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/boost_1_81_0-diffrg-slim.tar.xz")
set(BOOST_ARCHIVE_SHA256
    "4d25259d2d81f0349e1944169f63b2d4abc6abe6c1337cb3a90a6269400afb96")
set(BOOST_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/boost_1_81_0")
set(BOOST_PREPARED_STAMP "${BOOST_SOURCE_DIR}/.diffrg_prepared")
set(BOOST_PATCHED_STAMP "${BOOST_SOURCE_DIR}/.diffrg_patched")

include(ProcessorCount)
ProcessorCount(PARALLEL_JOBS)
if(PARALLEL_JOBS EQUAL 0)
  set(PARALLEL_JOBS 1)
endif()

# Extra arch flags (e.g. -march=x86-64-v3 or -march=native), forwarded by the
# superbuild so bundled Boost is compiled for the same target CPU as every
# other dependency.
set(BOOST_ARCH_FLAGS
    ""
    CACHE STRING "Extra architecture flags appended to b2 cxxflags/cflags")

if(NOT EXISTS "${BOOST_ARCHIVE}")
  message(
    FATAL_ERROR
      "Bundled Boost ${TRY_BOOST_VERSION} source archive is missing.\n"
      "Expected ${BOOST_ARCHIVE}.\n"
      "It is tracked in git -- restore it with\n"
      "  git checkout -- dependencies/boost/boost_1_81_0-diffrg-slim.tar.xz\n"
      "or use a system Boost instead via -DBOOST_DIR=<prefix>.")
endif()

# The checksum is what pins the Boost version now: once the archive hashes
# correctly the unpacked tree cannot be anything but Boost ${TRY_BOOST_VERSION}.
# It also catches a truncated checkout, which would otherwise fail deep inside
# tar with an unhelpful message. Costs ~0.05 s for 16 MiB.
file(SHA256 "${BOOST_ARCHIVE}" _boost_archive_sha256)
if(NOT _boost_archive_sha256 STREQUAL "${BOOST_ARCHIVE_SHA256}")
  message(
    FATAL_ERROR
      "Bundled Boost archive checksum mismatch:\n"
      "  file     ${BOOST_ARCHIVE}\n"
      "  expected ${BOOST_ARCHIVE_SHA256}\n"
      "  found    ${_boost_archive_sha256}\n"
      "Restore the tracked archive, or -- if you replaced it deliberately -- "
      "update BOOST_ARCHIVE_SHA256 in this file.")
endif()

if(CCACHE_PROGRAM)
  set(BOOST_CC_ENV "${CCACHE_PROGRAM} ${CMAKE_C_COMPILER}")
  set(BOOST_CXX_ENV "${CCACHE_PROGRAM} ${CMAKE_CXX_COMPILER}")
else()
  set(BOOST_CC_ENV "${CMAKE_C_COMPILER}")
  set(BOOST_CXX_ENV "${CMAKE_CXX_COMPILER}")
endif()

add_custom_command(
  OUTPUT "${BOOST_PREPARED_STAMP}"
  COMMAND ${CMAKE_COMMAND} -E rm -rf "${BOOST_SOURCE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E tar xJf "${BOOST_ARCHIVE}"
  # Layout assertion on the freshly unpacked tree. The archive checksum above
  # already pins the version far more strongly; what this guards is the
  # assumption everything below makes -- that the archive unpacks to a single
  # boost_1_81_0/ in release layout. Runs before the stamp is written, so a
  # failure stays dirty and is retried instead of being silently cached.
  COMMAND
    bash -c
    "grep -qx '#define BOOST_VERSION ${TRY_BOOST_VERSION_NUMBER}' '${BOOST_SOURCE_DIR}/boost/version.hpp' \
     || { echo 'Unpacked Boost is not ${TRY_BOOST_VERSION} (${TRY_BOOST_VERSION_NUMBER}), or the archive has an unexpected layout: ${BOOST_SOURCE_DIR}/boost/version.hpp' >&2; exit 1; }"
  COMMAND ${CMAKE_COMMAND} -E touch "${BOOST_PREPARED_STAMP}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${BOOST_ARCHIVE}"
  COMMENT "Unpacking bundled Boost ${TRY_BOOST_VERSION} into the build tree"
  VERBATIM)

add_custom_command(
  OUTPUT "${BOOST_PATCHED_STAMP}"
  COMMAND
    bash -c
    "if grep -q '__GNUC__ < 14' '${BOOST_SOURCE_DIR}/boost/math/cstdfloat/cstdfloat_limits.hpp'; then \
       echo 'Boost cstdfloat patch already applied.'; \
     else \
       patch --batch --forward '${BOOST_SOURCE_DIR}/boost/math/cstdfloat/cstdfloat_limits.hpp' < '${CMAKE_CURRENT_SOURCE_DIR}/cstdfloat_limits.hpp.diff'; \
     fi"
  COMMAND ${CMAKE_COMMAND} -E touch "${BOOST_PATCHED_STAMP}"
  DEPENDS "${BOOST_PREPARED_STAMP}"
          "${CMAKE_CURRENT_SOURCE_DIR}/cstdfloat_limits.hpp.diff"
  COMMENT "Patching build-local Boost ${TRY_BOOST_VERSION} source copy"
  VERBATIM)

add_custom_target(
  build_boost_boostrap
  COMMAND
    bash -c
    "CC=\"${BOOST_CC_ENV}\" CXX=\"${BOOST_CXX_ENV}\" ./bootstrap.sh --prefix='${CMAKE_INSTALL_PREFIX}' > '${CMAKE_CURRENT_BINARY_DIR}/03_bootstrap.log' 2>&1"
  WORKING_DIRECTORY "${BOOST_SOURCE_DIR}"
  DEPENDS "${BOOST_PATCHED_STAMP}"
  VERBATIM)

add_custom_target(
  build_boost_b2
  COMMAND
    bash -c
    "echo 'Installing Boost headers...' && \
     CC=\"${BOOST_CC_ENV}\" CXX=\"${BOOST_CXX_ENV}\" ./b2 headers \
        > '${CMAKE_CURRENT_BINARY_DIR}/04_headers.log' 2>&1 && \
     echo 'Building Boost...' && \
     CC=\"${BOOST_CC_ENV}\" CXX=\"${BOOST_CXX_ENV}\" ./b2 --build-dir='${CMAKE_CURRENT_BINARY_DIR}/boost-build' \
        --prefix='${CMAKE_INSTALL_PREFIX}' \
        --with-iostreams \
        --with-json \
        --with-math \
        --with-serialization \
        --with-system \
        --with-thread \
        -j ${PARALLEL_JOBS} \
        cxxflags=\"-fPIC -O3 -ffat-lto-objects -std=c++20 ${BOOST_ARCH_FLAGS}\" \
        cflags=\"-fPIC -O3 ${BOOST_ARCH_FLAGS}\" \
        install > '${CMAKE_CURRENT_BINARY_DIR}/04_build.log' 2>&1"
  VERBATIM
  WORKING_DIRECTORY "${BOOST_SOURCE_DIR}"
  DEPENDS build_boost_boostrap)

install(
  CODE [[
    execute_process(
      COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target build_boost_b2
      RESULT_VARIABLE boost_build_result
      COMMAND_ECHO STDOUT)

    if(NOT boost_build_result EQUAL 0)
      message(
        FATAL_ERROR
          "Boost installation failed while running the nested build target build_boost_b2."
      )
    endif()

    if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/include/boost")
      message(
        FATAL_ERROR
          "Boost installation did not produce ${CMAKE_INSTALL_PREFIX}/include/boost."
      )
    endif()

    if(NOT EXISTS "${CMAKE_INSTALL_PREFIX}/lib")
      message(
        FATAL_ERROR
          "Boost installation did not produce ${CMAKE_INSTALL_PREFIX}/lib."
      )
    endif()
  ]])
