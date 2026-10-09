cmake_minimum_required(VERSION 3.20)

project(petsc_dep NONE)

# ##############################################################################
# PETSc
# ##############################################################################
#
# Built the same way as the bundled HDF5 (see ../hdf5/CMakeLists.txt): download a
# release tarball idempotently, extract it idempotently, then run configure/make/
# install as separate steps, each with its own log file and a *checked* result.
# The whole thing runs as this sub-project's configure step, so ExternalProject
# stamps it and it does not re-run on every superbuild configure.
#
# What DiFfRG needs from PETSc is narrow: PETScWrappers::MPI::{Vector,BlockVector,
# SparseMatrix,BlockSparseMatrix}, KSP/PC, and whichever of hypre/MUMPS was asked
# for. Everything else is off, which is the difference between a ~5 minute and a
# ~40 minute build.

set(TRY_PETSC_VERSION "3.23.1")
set(PETSC_TARBALL_SHA256 "")  # set to pin the download; empty disables the check

# ------------------------------------------------------------------ options ---
# All of these are normally supplied by the parent superbuild via CMAKE_CACHE_ARGS;
# the defaults only matter when this directory is configured standalone.
foreach(_opt HAVE_MPI PETSC_GMRES PETSC_MUMPS)
  if(NOT DEFINED ${_opt})
    set(${_opt} OFF CACHE BOOL "")
  endif()
endforeach()
if(NOT DEFINED NATIVE)
  set(NATIVE ON CACHE BOOL "Optimize for the build machine's CPU (-march=native)")
endif()
if(NOT DEFINED MARCH)
  set(MARCH "" CACHE STRING "Explicit -march target (e.g. x86-64-v3, native, none). Overrides NATIVE.")
endif()
if(NOT DEFINED PETSC_JOBS OR PETSC_JOBS LESS 1)
  # Standalone fallback only. The superbuild passes a RAM-aware value; never use a
  # bare -j/-j$(nproc) here, PETSc's make will happily exhaust memory.
  set(PETSC_JOBS 4)
endif()

# Same MARCH/NATIVE resolution as the superbuild (see DiFfRG/cmake/common.cmake):
# a non-empty MARCH wins; empty falls back to the NATIVE bool; "none" = no flag.
if(NOT MARCH STREQUAL "")
  set(_march_value "${MARCH}")
elseif(NATIVE)
  set(_march_value "native")
else()
  set(_march_value "none")
endif()
if(_march_value STREQUAL "none")
  set(PETSC_MARCH "")
else()
  set(PETSC_MARCH "-march=${_march_value}")
endif()

# ----------------------------------------------------------------- download ---
set(PETSC_TARBALL "${CMAKE_CURRENT_BINARY_DIR}/petsc-${TRY_PETSC_VERSION}.tar.gz")
set(PETSC_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}/petsc-${TRY_PETSC_VERSION}")

set(PETSC_TARBALL_VALID FALSE)
if(EXISTS ${PETSC_TARBALL})
  file(SIZE ${PETSC_TARBALL} PETSC_TARBALL_SIZE)
  if(PETSC_TARBALL_SIZE GREATER 0)
    set(PETSC_TARBALL_VALID TRUE)
  else()
    file(REMOVE ${PETSC_TARBALL})
  endif()
endif()

if(NOT PETSC_TARBALL_VALID)
  message(STATUS "Downloading PETSc ${TRY_PETSC_VERSION}...")
  set(_hash_arg "")
  if(PETSC_TARBALL_SHA256)
    set(_hash_arg EXPECTED_HASH SHA256=${PETSC_TARBALL_SHA256})
  endif()
  file(
    DOWNLOAD
    "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${TRY_PETSC_VERSION}.tar.gz"
    ${PETSC_TARBALL}
    ${_hash_arg}
    STATUS PETSC_DOWNLOAD_STATUS
    LOG PETSC_DOWNLOAD_LOG)
  list(GET PETSC_DOWNLOAD_STATUS 0 PETSC_DOWNLOAD_CODE)
  if(NOT PETSC_DOWNLOAD_CODE EQUAL 0)
    file(REMOVE ${PETSC_TARBALL})
    message(FATAL_ERROR "Failed to download PETSc: ${PETSC_DOWNLOAD_STATUS}\n${PETSC_DOWNLOAD_LOG}")
  endif()
  file(SIZE ${PETSC_TARBALL} PETSC_TARBALL_SIZE)
  if(NOT PETSC_TARBALL_SIZE GREATER 0)
    file(REMOVE ${PETSC_TARBALL})
    message(FATAL_ERROR "Failed to download PETSc: archive is empty.")
  endif()
else()
  message(STATUS "PETSc already downloaded.")
endif()

if(NOT EXISTS ${PETSC_SRC_DIR})
  message(STATUS "Extracting PETSc...")
  file(ARCHIVE_EXTRACT INPUT ${PETSC_TARBALL} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
  if(NOT EXISTS ${PETSC_SRC_DIR})
    message(FATAL_ERROR "PETSc archive did not contain the expected directory ${PETSC_SRC_DIR}.")
  endif()
else()
  message(STATUS "PETSc already extracted.")
endif()

# ------------------------------------------------------------ configure args ---
set(PETSC_OPTIONS
    --with-debugging=0
    --with-x=0
    --with-ssl=0
    --with-shared-libraries=1
    # Belt and braces on top of the compiler pinning below: GCC >= 15 defaults to
    # C23, where `bool` is a keyword, and hypre's bundled pilut still does
    # `typedef unsigned char bool;` (distributed_ls/pilut/struct.h) -- an outright
    # compile error. PETSc passes CFLAGS down to every --download-* package, so
    # pinning the standard protects hypre and any other C-of-a-certain-age
    # dependency even if someone points this at a newer compiler. No-op on the
    # gcc-14 the superbuild currently selects, whose default is already gnu17.
    "--CFLAGS=-std=gnu17"
    # Bound the *downloaded packages'* builds too. PETSC_JOBS below only governs
    # PETSc's own make; the --download-* packages are built during configure by
    # PETSc's internal make, which otherwise picks its own width (observed: -j24
    # on a 32-core box, well past what this machine should ever run).
    "--with-make-np=${PETSC_JOBS}"
    "--with-make-load=${PETSC_JOBS}"
    # deal.II is built with DEAL_II_WITH_COMPLEX_VALUES=OFF and
    # DEAL_II_WITH_64BIT_INDICES=OFF; its configure_20_petsc.cmake hard-rejects a
    # complex PETSc, and an index-size mismatch fails at link time or worse.
    --with-scalar-type=real
    --with-precision=double
    --with-64-bit-indices=0
    # DiFfRG's own Kokkos (CUDA-enabled) owns the GPU. deal.II additionally refuses
    # a Kokkos-enabled PETSc when Kokkos is bundled, and a PETSc CUDA build would
    # both contend for the device and add a long compile for nothing.
    --with-cuda=0
    --with-kokkos=0)

# PETSc has to be built with the same compilers as deal.II and DiFfRG, or the
# resulting libraries disagree about the C++ ABI and the standard library. The MPI
# wrappers do NOT give us that for free: mpicc/mpicxx invoke whichever compiler is
# the system default, which is frequently *not* the one the superbuild picked (on
# the reference machine: superbuild gcc-14, system default gcc-16). So we pass the
# wrappers to PETSc, and separately point the wrappers at the right compiler via
# the implementations' environment overrides. OMPI_* is Open MPI, MPICH_* is MPICH
# and its derivatives (Intel MPI, MVAPICH); setting both is harmless.
set(_petsc_env)
if(PETSC_C_COMPILER)
  list(APPEND _petsc_env "OMPI_CC=${PETSC_C_COMPILER}" "MPICH_CC=${PETSC_C_COMPILER}")
endif()
if(PETSC_CXX_COMPILER)
  list(APPEND _petsc_env "OMPI_CXX=${PETSC_CXX_COMPILER}" "MPICH_CXX=${PETSC_CXX_COMPILER}")
endif()
if(PETSC_Fortran_COMPILER)
  list(APPEND _petsc_env "OMPI_FC=${PETSC_Fortran_COMPILER}" "MPICH_FC=${PETSC_Fortran_COMPILER}")
endif()

if(HAVE_MPI)
  list(APPEND PETSC_OPTIONS --with-mpi=1)
  if(MPI_C_COMPILER)
    list(APPEND PETSC_OPTIONS "--with-cc=${MPI_C_COMPILER}")
  endif()
  if(MPI_CXX_COMPILER)
    list(APPEND PETSC_OPTIONS "--with-cxx=${MPI_CXX_COMPILER}")
  endif()
else()
  # No wrappers in play, so name the compilers directly.
  list(APPEND PETSC_OPTIONS --with-mpi=0)
  if(PETSC_C_COMPILER)
    list(APPEND PETSC_OPTIONS "--with-cc=${PETSC_C_COMPILER}")
  endif()
  if(PETSC_CXX_COMPILER)
    list(APPEND PETSC_OPTIONS "--with-cxx=${PETSC_CXX_COMPILER}")
  endif()
endif()

# hypre gives PETScWrappers::PreconditionBoomerAMG. It needs MPI + BLAS/LAPACK +
# C++ but no Fortran (its PETSc recipe passes --disable-fortran when no FC is
# configured), so it stays compatible with the --with-fc=0 fast path below.
if(PETSC_GMRES)
  list(APPEND PETSC_OPTIONS --download-hypre)
endif()

# MUMPS gives PETScWrappers::SparseDirectMUMPS. Its PETSc recipe raises a hard
# error without an F90 compiler and pulls in the ScaLAPACK chain, which is the
# bulk of the build time -- hence opt-in. metis/parmetis are optional for MUMPS
# but materially improve its parallel ordering; PT-Scotch and hwloc (and the bison
# PT-Scotch drags in) are left out deliberately.
if(PETSC_MUMPS)
  if(NOT PETSC_Fortran_COMPILER)
    message(FATAL_ERROR "PETSC_MUMPS=ON requires a Fortran compiler, but none was provided.")
  endif()
  # Which Fortran compiler PETSc is told about matters as much as which one actually runs.
  # With MPI enabled it must be the MPI *wrapper*: handing it the bare gfortran makes PETSc's
  # "can Fortran see MPI?" probe fail with
  #     Fortran error! mpi_init() could not be located!
  # because the bare compiler has no MPI include or link flags. Same split as for C/C++ above:
  # the wrapper is what PETSc invokes, and OMPI_FC/MPICH_FC (set below) decide which compiler
  # the wrapper actually runs, keeping it consistent with the rest of the superbuild.
  set(_petsc_fc_arg "${PETSC_Fortran_COMPILER}")
  if(HAVE_MPI)
    find_program(_petsc_mpifc NAMES mpifort mpif90)
    if(NOT _petsc_mpifc)
      message(
        FATAL_ERROR
          "PETSC_MUMPS=ON with MPI needs an MPI Fortran wrapper (mpifort/mpif90) on PATH; "
          "none was found. Install your MPI implementation's Fortran support, or build with "
          "-DPETSC_MUMPS=OFF and use the hypre/BoomerAMG Krylov path instead.")
    endif()
    set(_petsc_fc_arg "${_petsc_mpifc}")
  endif()
  list(APPEND PETSC_OPTIONS
       "--with-fc=${_petsc_fc_arg}"
       "--FOPTFLAGS=-O3 ${PETSC_MARCH} -ffp-contract=fast"
       --download-mumps
       --download-scalapack
       --download-metis
       --download-parmetis)
else()
  # No Fortran at all: this is what keeps the default build in the minutes range.
  list(APPEND PETSC_OPTIONS --with-fc=0)
endif()

# Point PETSc at the same BLAS/LAPACK the rest of the superbuild uses: the bundled
# OpenBLAS if one was built, otherwise the system library the superbuild found and
# handed down as a comma-separated list. Left to search on its own PETSc can pick a
# different library entirely -- on a cluster, MKL via MKLROOT is the usual one --
# and an index-size or ABI mismatch with deal.II follows.
if(OpenBLAS_LIBRARY)
  list(APPEND PETSC_OPTIONS
       "--with-blas-lib=${OpenBLAS_LIBRARY}"
       "--with-lapack-lib=${OpenBLAS_LIBRARY}")
elseif(PETSC_BLASLAPACK_LIB)
  list(APPEND PETSC_OPTIONS "--with-blaslapack-lib=[${PETSC_BLASLAPACK_LIB}]")
endif()

# PETSc determines the BLAS index size by *running* a test binary, and where it
# cannot run one it assumes 64-bit -- which makes --download-scalapack (i.e. MUMPS)
# fail outright with "Cannot use scalapack with 64-bit BLAS/LAPACK indices". The
# superbuild has already probed the very library pinned above, so hand PETSc the
# answer instead of letting it guess. Only ever passed when the probe produced a
# verdict; `unknown` leaves PETSc's own detection in charge.
if(OpenBLAS_LIBRARY OR BLAS_INDEX_SIZE STREQUAL "32")
  list(APPEND PETSC_OPTIONS --known-64-bit-blas-indices=0)
elseif(BLAS_INDEX_SIZE STREQUAL "64")
  message(FATAL_ERROR "The BLAS/LAPACK in use has 64-bit indices; PETSc here is built with 32-bit ones.")
endif()

list(APPEND PETSC_OPTIONS
     "--COPTFLAGS=-O3 ${PETSC_MARCH} -ffp-contract=fast"
     "--CXXOPTFLAGS=-O3 ${PETSC_MARCH} -ffp-contract=fast"
     "--prefix=${CMAKE_INSTALL_PREFIX}")

# --------------------------------------------------------------- configure -----
message(STATUS "Configuring PETSc (jobs: ${PETSC_JOBS}, hypre: ${PETSC_GMRES}, MUMPS: ${PETSC_MUMPS})...")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env ${_petsc_env} ./configure ${PETSC_OPTIONS}
  WORKING_DIRECTORY ${PETSC_SRC_DIR}
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/01_configure.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/01_configure.log
  RESULT_VARIABLE PETSC_CONFIGURE_RESULT)
if(NOT PETSC_CONFIGURE_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to configure PETSc. See log: ${CMAKE_CURRENT_BINARY_DIR}/01_configure.log")
endif()

# ------------------------------------------------------------------- build -----
# PETSc's top-level make dispatches to its own build system; MAKE_NP is how it is
# told how wide to go. Passing -j as well keeps a plain make from ignoring it.
message(STATUS "Building PETSc...")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env ${_petsc_env} make -j ${PETSC_JOBS} MAKE_NP=${PETSC_JOBS} all
  WORKING_DIRECTORY ${PETSC_SRC_DIR}
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/02_build.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/02_build.log
  RESULT_VARIABLE PETSC_BUILD_RESULT)
if(NOT PETSC_BUILD_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to build PETSc. See log: ${CMAKE_CURRENT_BINARY_DIR}/02_build.log")
endif()

# ----------------------------------------------------------------- install -----
message(STATUS "Installing PETSc...")
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env ${_petsc_env} make install
  WORKING_DIRECTORY ${PETSC_SRC_DIR}
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/03_install.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/03_install.log
  RESULT_VARIABLE PETSC_INSTALL_RESULT)
if(NOT PETSC_INSTALL_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to install PETSc. See log: ${CMAKE_CURRENT_BINARY_DIR}/03_install.log")
endif()
