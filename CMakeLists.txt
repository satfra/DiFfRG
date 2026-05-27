# ##############################################################################
# Top-level build for DiFfRG (superbuild)
#
# This file builds all bundled dependencies (boost, oneTBB, SUNDIALS, deal.II)
# into ${CMAKE_INSTALL_PREFIX}/bundled/ and then builds and installs the 
# DiFfRG library itself.
# ##############################################################################

cmake_minimum_required(VERSION 3.26.4)

project(DiFfRG_superbuild LANGUAGES C CXX)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds are not allowed. Configure into a separate build directory (build.sh uses DiFfRG_build)."
  )
endif()

set(HOME_DIR $ENV{HOME})
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# Use the module-mode FindBoost for our own Boost detection, matching how
# deal.II discovers Boost (its FindDEAL_II_BOOST sets Boost_NO_BOOST_CMAKE).
# Also silences the CMP0167 "FindBoost module is removed" warning on CMake >=
# 3.30 / 4.x.
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

include(ExternalProject)
include(CheckLanguage)
include(ProcessorCount)

# ##############################################################################
# Install locations
# ##############################################################################

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX
      ${HOME_DIR}/.local/share/DiFfRG
      CACHE PATH "Install prefix" FORCE)
endif()

# Each heavy dependency is installed into its own subdirectory, matching the
# find_package(... HINTS ${BUNDLED_DIR}/<lib>_install) calls in
# DiFfRG/cmake/setup_build_system.cmake. Always derived from the install prefix
# (NOT a cache entry), so it tracks changes to CMAKE_INSTALL_PREFIX when an
# existing build tree is reconfigured. (A cached value would silently keep
# pointing at a previous prefix.)
set(DEPEND_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/bundled)
set(BOOST_INSTALL ${DEPEND_INSTALL_PREFIX}/boost_install)
set(TBB_INSTALL ${DEPEND_INSTALL_PREFIX}/oneTBB_install)
set(SUNDIALS_INSTALL ${DEPEND_INSTALL_PREFIX}/sundials_install)
set(DEALII_INSTALL ${DEPEND_INSTALL_PREFIX}/dealii_install)

message(STATUS "DiFfRG install prefix:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Bundled dependency prefix:   ${DEPEND_INSTALL_PREFIX}")

# ##############################################################################
# Build configuration forwarded from build.sh / config
# ##############################################################################

# Compiler/linker flags (sourced from the `config` file by build.sh and passed
# in via -D). They default to empty, in which case the dependency build systems
# pick their own defaults.
set(CMAKE_CXX_FLAGS
    ""
    CACHE STRING "CXX flags for all builds")
set(CMAKE_C_FLAGS
    ""
    CACHE STRING "C flags for all builds")
set(CMAKE_CUDA_FLAGS
    ""
    CACHE STRING "CUDA flags for the DiFfRG library")
set(DIFFRG_EXE_LINKER_FLAGS
    ""
    CACHE STRING "Linker flags for dependency builds")
# Extra deal.II cmake flags (the `config` file's DEAL_II_CMAKE variable).
set(DEAL_II_CMAKE
    ""
    CACHE STRING "Extra cmake flags forwarded to the deal.II build")

option(USE_CUDA "Build the DiFfRG library with CUDA support" OFF)
option(USE_CCACHE "Use ccache for dependency builds if available" ON)

# Boost source selection. By default a system Boost >= 1.80 is used if found,
# otherwise the pinned Boost is built from source. Set BOOST_DIR to point at a
# specific Boost prefix, or BUILD_BOOST=ON to always build the bundled one.
set(BOOST_DIR
    ""
    CACHE PATH
          "Use the Boost installation at this prefix instead of building one")
option(BUILD_BOOST
       "Always build the bundled, pinned Boost (ignore any system Boost)" OFF)

# Same selection scheme for TBB (>= 2021) and SUNDIALS (>= 5.4.0).
set(TBB_DIR
    ""
    CACHE PATH "Use the TBB installation at this prefix instead of building one")
option(BUILD_TBB "Always build the bundled oneTBB (ignore any system TBB)" OFF)
set(SUNDIALS_DIR
    ""
    CACHE PATH
          "Use the SUNDIALS installation at this prefix instead of building one")
option(BUILD_SUNDIALS
       "Always build the bundled SUNDIALS (ignore any system SUNDIALS)" OFF)

set(DEALII_MAX_JOBS
    16
    CACHE STRING "Hard cap on parallel jobs for the deal.II build")

# Number of build threads (forwarded from build.sh -j). Used for boost (b2) and
# as the basis for the RAM-capped deal.II parallelism.
ProcessorCount(_nproc)
if(_nproc LESS 1)
  set(_nproc 1)
endif()
set(BUILD_THREADS
    ${_nproc}
    CACHE STRING "Number of parallel build threads")

# ##############################################################################
# Pre-flight checks
# ##############################################################################

find_program(GIT_EXECUTABLE git)
if(NOT GIT_EXECUTABLE)
  message(
    FATAL_ERROR
      "'git' is required to fetch dependencies but was not found.\n"
      "  Ubuntu/Debian: sudo apt install git | Arch: sudo pacman -S git | macOS: brew install git"
  )
endif()

# A Fortran compiler is optional: deal.II uses enable_language_optional(Fortran)
# and, when none is present, falls back to the standard gfortran name-mangling
# for its LAPACK detection. That works for LAPACK/BLAS built with gfortran
# conventions (OpenBLAS, reference LAPACK), but may fail otherwise.
check_language(Fortran)
if(NOT CMAKE_Fortran_COMPILER)
  message(
    WARNING
      "No Fortran compiler found. deal.II will fall back to standard gfortran "
      "LAPACK name-mangling; the build may fail if your LAPACK/BLAS uses a "
      "different convention. To install one:\n"
      "  Ubuntu/Debian: sudo apt install gfortran | Arch: sudo pacman -S gcc-fortran | macOS: brew install gcc"
  )
endif()

find_package(GSL QUIET)
if(NOT GSL_FOUND)
  message(
    FATAL_ERROR
      "Required system dependency not found: GSL (GNU Scientific Library).\n"
      "  Ubuntu/Debian: sudo apt install libgsl-dev | Arch: sudo pacman -S gsl | macOS: brew install gsl"
  )
endif()

find_package(LAPACK QUIET)
if(NOT LAPACK_FOUND)
  message(
    FATAL_ERROR
      "Required system dependency not found: LAPACK/BLAS (needed by deal.II).\n"
      "  Ubuntu/Debian: sudo apt install libopenblas-dev | Arch: sudo pacman -S blas-openblas | macOS: brew install openblas"
  )
endif()

# CUDA is detected here only to warn early; the DiFfRG library obtains its CUDA
# toolchain (CCCL/RMM) during its own build.
if(USE_CUDA)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(
      WARNING
        "USE_CUDA=ON but no CUDA compiler was found; the DiFfRG build will likely fall back to CPU-only."
    )
  endif()
endif()

# Optional ccache
if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    set(CCACHE_CMAKE_ARGS
        -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CCACHE_PROGRAM}
        -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CCACHE_PROGRAM})
  endif()
endif()

message(STATUS "All pre-flight checks passed.")

# ##############################################################################
# Pinned dependency commits (kept identical to the DiFfRG 1.0 submodules so the
# build stays consistent with the publication).
# ##############################################################################

# Boost is fetched as the official release tarball rather than a recursive git
# clone of the boostorg superproject: the latter pulls ~2 GB (full history +
# 150+ library submodules) and dominates the build time. 1.88.0 is the release
# the pinned 1.0 submodule built as.
set(BOOST_VERSION 1.88.0)
set(BOOST_URL
    https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.gz)
set(BOOST_SHA256
    3621533e820dcab1e8012afd583c0c73cf0f77694952b81352bf38c1488f9cb4)
set(TBB_REPO https://github.com/uxlfoundation/oneTBB.git)
set(TBB_TAG 8659619fbd73361a129e02df19d24aa5054649cb)
set(SUNDIALS_REPO https://github.com/LLNL/sundials.git)
set(SUNDIALS_TAG v7.2.1)
set(DEALII_REPO https://github.com/dealii/dealii.git)
set(DEALII_TAG 88cb8ba3cb5c4ffd287785e4af3fd8c634613fb2)

# ##############################################################################
# [1/5] Boost
#
# Boost is needed by BOTH deal.II and the DiFfRG library, and deal.II asserts at
# compile time that an application links the exact Boost version it was
# configured with. We therefore resolve a single Boost (BOOST_PREFIX) and point
# both at it. DiFfRG requires >= 1.80.
#
# IMPORTANT viability constraint: deal.II's FindDEAL_II_BOOST forces the legacy
# module-mode FindBoost (it sets Boost_NO_BOOST_CMAKE) and requires a COMPILED
# Boost.System library. Modern Boost (>= ~1.69) ships Boost.System as
# header-only, so its packages no longer provide libboost_system -> deal.II's
# find then fails and silently falls back to its bundled Boost, mismatching the
# one DiFfRG links. We therefore only adopt an external Boost when this exact
# deal.II-style, module-mode find succeeds; otherwise we build the bundled
# Boost.
#
# Selection: - BOOST_DIR set      -> use that prefix (FATAL if deal.II can't
# consume it) - BUILD_BOOST=ON     -> build the pinned Boost from source -
# otherwise (default)-> use a deal.II-consumable system Boost if found, else
# build it
# ##############################################################################

set(_build_boost OFF)
set(BOOST_PREFIX "")
set(_boost_dep_target "") # ExternalProject dependency, only when building Boost

# Replicate deal.II's Boost discovery so "viable" means "deal.II can use it":
# module mode (CMP0167 OLD is set above), config disabled, compiled components.
set(Boost_NO_BOOST_CMAKE ON)
set(Boost_NO_WARN_NEW_VERSIONS TRUE)
set(_boost_required_components thread iostreams serialization system)

set(_boost_incompat_hint
    "deal.II uses the legacy module-mode FindBoost and needs a COMPILED Boost.System library, but modern Boost (>= ~1.69) ships Boost.System header-only. Use an older Boost / one with compiled component libraries, or build the bundled Boost (-DBUILD_BOOST=ON)."
)

if(NOT "${BOOST_DIR}" STREQUAL "")
  # Explicit, user-given Boost prefix: must be consumable by deal.II.
  set(BOOST_ROOT "${BOOST_DIR}")
  find_package(Boost 1.80 QUIET COMPONENTS ${_boost_required_components})
  if(NOT Boost_FOUND)
    message(
      FATAL_ERROR
        "BOOST_DIR=${BOOST_DIR} does not provide a Boost (>= 1.80) that deal.II can use.\n"
        "  ${_boost_incompat_hint}")
  endif()
  set(BOOST_PREFIX "${BOOST_DIR}")
  message(
    STATUS "[1/5] Using Boost ${Boost_VERSION} from BOOST_DIR=${BOOST_DIR}")
elseif(BUILD_BOOST)
  set(_build_boost ON)
  message(
    STATUS "[1/5] BUILD_BOOST=ON: building the bundled Boost from source.")
else()
  # Default: use a deal.II-consumable system Boost if present, else build it.
  find_package(Boost 1.80 QUIET COMPONENTS ${_boost_required_components})
  if(Boost_FOUND)
    # Derive the install prefix from the discovered include dir.
    get_filename_component(BOOST_PREFIX "${Boost_INCLUDE_DIRS}" DIRECTORY)
    message(
      STATUS "[1/5] Using system Boost ${Boost_VERSION} (${BOOST_PREFIX}). "
             "Set -DBUILD_BOOST=ON to build the bundled Boost instead.")
  else()
    set(_build_boost ON)
    message(
      STATUS
        "[1/5] No deal.II-consumable system Boost >= 1.80 found; building the bundled Boost from source."
    )
    message(STATUS "      (${_boost_incompat_hint})")
  endif()
endif()

if(_build_boost)
  set(BOOST_PREFIX ${BOOST_INSTALL})
  set(_boost_dep_target boost_dep)
  set(_boost_cxxflags "${CMAKE_CXX_FLAGS} -std=c++20")

  message(
    STATUS
      "[1/5] Configuring Boost (bundled, replicating external/build_boost.sh)..."
  )
  ExternalProject_Add(
    boost_dep
    URL ${BOOST_URL}
    URL_HASH SHA256=${BOOST_SHA256}
    PREFIX ${CMAKE_BINARY_DIR}/boost
    INSTALL_DIR ${BOOST_INSTALL}
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON # send bootstrap.sh output to a log file, not the console
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
    BUILD_IN_SOURCE 1
    # Run bootstrap.sh / b2 directly (no `bash -c` wrapper) so the cxxflags
    # value, which contains spaces, is passed to b2 as a single argument without
    # any shell-quoting that could swallow the trailing `install` target.
    CONFIGURE_COMMAND <SOURCE_DIR>/bootstrap.sh --prefix=<INSTALL_DIR>
    BUILD_COMMAND
      <SOURCE_DIR>/b2 --build-dir=<BINARY_DIR> --prefix=<INSTALL_DIR>
      --with-headers --with-iostreams --with-json --with-math
      --with-serialization --with-system --with-thread -j ${BUILD_THREADS}
      "cxxflags=${_boost_cxxflags}" install
    INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Boost installed by b2.")
endif()

# ##############################################################################
# [2/5] oneTBB
#
# Needed by BOTH DiFfRG (find_package(TBB 2021 CONFIG)) and deal.II (path-based,
# floor 4.2); a single TBB is shared by both. There is no version-match
# assertion, and modern TBB ships TBBConfig.cmake, so a system TBB is usable.
# Selection mirrors Boost:
#   TBB_DIR set       -> use that prefix (FATAL if it lacks TBB >= 2021)
#   BUILD_TBB=ON      -> build the bundled oneTBB
#   otherwise default -> use a system TBB >= 2021 if found, else build it
# A system TBB in standard paths is found by both consumers without a hint, so
# TBB_PREFIX is left empty in that case (routed only for bundled / user prefix).
# ##############################################################################

set(_build_tbb OFF)
set(TBB_PREFIX "")
set(_tbb_dep_target "")

# Capture the user-given prefix BEFORE find_package(TBB), which overwrites the
# TBB_DIR variable with the location of the discovered TBBConfig.cmake.
set(_tbb_user_dir "${TBB_DIR}")

if(NOT "${_tbb_user_dir}" STREQUAL "")
  # Restrict the search to the given prefix (NO_DEFAULT_PATH) so a wrong prefix
  # fails instead of silently picking up a system TBB.
  find_package(
    TBB 2021 QUIET CONFIG
    PATHS "${_tbb_user_dir}"
    PATH_SUFFIXES lib/cmake/TBB lib64/cmake/TBB cmake/TBB
    NO_DEFAULT_PATH)
  if(NOT TBB_FOUND)
    message(
      FATAL_ERROR
        "TBB_DIR=${_tbb_user_dir} does not provide TBB >= 2021. Point it at a prefix "
        "containing TBBConfig.cmake, or use -DBUILD_TBB=ON to build the bundled oneTBB."
    )
  endif()
  set(TBB_PREFIX "${_tbb_user_dir}")
  message(
    STATUS "[2/5] Using TBB ${TBB_VERSION} from TBB_DIR=${_tbb_user_dir}")
elseif(BUILD_TBB)
  set(_build_tbb ON)
  message(STATUS "[2/5] BUILD_TBB=ON: building the bundled oneTBB from source.")
else()
  find_package(TBB 2021 QUIET CONFIG)
  if(TBB_FOUND)
    message(
      STATUS
        "[2/5] Using system TBB ${TBB_VERSION} (${TBB_DIR}). "
        "Set -DBUILD_TBB=ON to build the bundled oneTBB instead.")
  else()
    set(_build_tbb ON)
    message(
      STATUS "[2/5] No system TBB >= 2021 found; building the bundled oneTBB.")
  endif()
endif()

if(_build_tbb)
  set(TBB_PREFIX ${TBB_INSTALL})
  set(_tbb_dep_target oneTBB_dep)
  message(STATUS "[2/5] Configuring oneTBB (bundled, replicating external/build_oneTBB.sh)...")
  ExternalProject_Add(
    oneTBB_dep
    DOWNLOAD_COMMAND
      ${CMAKE_COMMAND} -DGIT_EXECUTABLE=${GIT_EXECUTABLE} -DREPO=${TBB_REPO}
      -DTAG=${TBB_TAG} -DDEST=<SOURCE_DIR> -P
      ${CMAKE_SOURCE_DIR}/cmake/fetch_commit.cmake
    PREFIX ${CMAKE_BINARY_DIR}/oneTBB
    INSTALL_DIR ${TBB_INSTALL}
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
    CMAKE_CACHE_ARGS
      -DCMAKE_BUILD_TYPE:STRING=Release
      -DTBB_TEST:BOOL=OFF
      -DCMAKE_CXX_STANDARD:STRING=20
      -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
      -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
      "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}"
      "-DCMAKE_EXE_LINKER_FLAGS:STRING=${DIFFRG_EXE_LINKER_FLAGS}"
      -DCMAKE_INSTALL_PREFIX:PATH=${TBB_INSTALL}
      ${CCACHE_CMAKE_ARGS})
endif()

# ##############################################################################
# [3/5] SUNDIALS
#
# SUNDIALS is needed ONLY by deal.II (DiFfRG never references it). deal.II finds
# it by library/path search (sundials_arkode/kinsol/nvecserial/ida(s), + core
# for >= 7) with floor 5.4.0 and no version-match assertion. We adopt a system
# SUNDIALS that provides those libraries at version >= 5.4.0; otherwise build the
# bundled one. As with TBB, a system install in standard paths is found by
# deal.II without a hint, so SUNDIALS_PREFIX is only routed for bundled / an
# explicit SUNDIALS_DIR prefix.
# Selection: SUNDIALS_DIR set -> use it (FATAL if unusable); BUILD_SUNDIALS=ON ->
# build bundled; otherwise -> system if usable, else build bundled.
# ##############################################################################

set(_build_sundials OFF)
set(SUNDIALS_PREFIX "")
set(_sundials_dep_target "")
set(_sun_hint "")
set(_sun_explicit OFF)

if(NOT "${SUNDIALS_DIR}" STREQUAL "")
  set(_sun_hint "${SUNDIALS_DIR}")
  set(_sun_explicit ON)
elseif(BUILD_SUNDIALS)
  set(_build_sundials ON)
  message(
    STATUS "[3/5] BUILD_SUNDIALS=ON: building the bundled SUNDIALS from source.")
endif()

if(NOT _build_sundials)
  # Replicate deal.II's SUNDIALS discovery: required libs + version >= 5.4.0.
  find_library(_SUN_ARKODE NAMES sundials_arkode HINTS ${_sun_hint}
               PATH_SUFFIXES lib lib64)
  find_library(_SUN_KINSOL NAMES sundials_kinsol HINTS ${_sun_hint}
               PATH_SUFFIXES lib lib64)
  find_library(_SUN_NVEC NAMES sundials_nvecserial HINTS ${_sun_hint}
               PATH_SUFFIXES lib lib64)
  # The version macros live in sundials_config.h on most installs
  # (sundials_version.h only declares the runtime accessors and #includes the
  # config -- e.g. on Fedora it carries no macros), so locate the config header
  # and read whichever of the two actually defines SUNDIALS_VERSION_MAJOR.
  find_path(_SUN_INC sundials/sundials_config.h HINTS ${_sun_hint}
            PATH_SUFFIXES include)

  set(_sun_ok FALSE)
  set(_sun_ver "")
  set(_sun_mpi OFF)
  if(_SUN_ARKODE
     AND _SUN_KINSOL
     AND _SUN_NVEC
     AND _SUN_INC)
    # DiFfRG builds deal.II without MPI (DEAL_II_WITH_MPI=OFF below). A SUNDIALS
    # built with MPI has SUNDIALS_MPI_ENABLED=1 in sundials_config.h, which makes
    # sundials_types.h #include <mpi.h> and define MPI_Comm as a real pointer type.
    # That collides with deal.II's mpi_stub.h (MPI_Comm = int), so such a system
    # SUNDIALS is unusable here -- detect it and fall back to the bundled build.
    if(EXISTS "${_SUN_INC}/sundials/sundials_config.h")
      file(STRINGS "${_SUN_INC}/sundials/sundials_config.h" _sun_mpi_probe
           REGEX "#define[ \t]+SUNDIALS_MPI_ENABLED[ \t]+1")
      if(_sun_mpi_probe)
        set(_sun_mpi ON)
      endif()
    endif()
    set(_sun_verhdr "")
    foreach(_h sundials/sundials_config.h sundials/sundials_version.h)
      if(EXISTS "${_SUN_INC}/${_h}")
        file(STRINGS "${_SUN_INC}/${_h}" _sun_probe
             REGEX "#define[ \t]+SUNDIALS_VERSION_MAJOR")
        if(_sun_probe)
          set(_sun_verhdr "${_SUN_INC}/${_h}")
          break()
        endif()
      endif()
    endforeach()
    if(_sun_verhdr)
      file(STRINGS "${_sun_verhdr}" _sun_mj REGEX "#define[ \t]+SUNDIALS_VERSION_MAJOR")
      file(STRINGS "${_sun_verhdr}" _sun_mn REGEX "#define[ \t]+SUNDIALS_VERSION_MINOR")
      file(STRINGS "${_sun_verhdr}" _sun_pt REGEX "#define[ \t]+SUNDIALS_VERSION_PATCH")
      string(REGEX REPLACE "^.*MAJOR[ \t]+([0-9]+).*" "\\1" _sun_mj "${_sun_mj}")
      string(REGEX REPLACE "^.*MINOR[ \t]+([0-9]+).*" "\\1" _sun_mn "${_sun_mn}")
      string(REGEX REPLACE "^.*PATCH[ \t]+([0-9]+).*" "\\1" _sun_pt "${_sun_pt}")
      set(_sun_ver "${_sun_mj}.${_sun_mn}.${_sun_pt}")
      if(_sun_ver MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$"
         AND NOT _sun_ver VERSION_LESS 5.4.0)
        set(_sun_ok TRUE)
      endif()
    endif()
  endif()

  if(_sun_ok AND _sun_mpi)
    # Usable version, but built with MPI -> incompatible with our MPI-less deal.II.
    if(_sun_explicit)
      message(
        FATAL_ERROR
          "SUNDIALS_DIR=${SUNDIALS_DIR} provides SUNDIALS ${_sun_ver} built WITH MPI "
          "(SUNDIALS_MPI_ENABLED=1), which is incompatible with DiFfRG's MPI-less "
          "deal.II (DEAL_II_WITH_MPI=OFF): the conflicting MPI_Comm definitions break "
          "the deal.II build. Use -DBUILD_SUNDIALS=ON to build the bundled MPI-less "
          "SUNDIALS, or point -DSUNDIALS_DIR at a SUNDIALS built without MPI.")
    else()
      set(_build_sundials ON)
      message(
        STATUS
          "[3/5] System SUNDIALS ${_sun_ver} (${_SUN_INC}) was built with MPI "
          "(SUNDIALS_MPI_ENABLED=1); incompatible with DiFfRG's MPI-less deal.II. "
          "Building the bundled SUNDIALS instead.")
    endif()
  elseif(_sun_ok)
    if(_sun_explicit)
      set(SUNDIALS_PREFIX "${SUNDIALS_DIR}")
    endif() # system-in-standard-paths: leave empty, deal.II finds it unhinted
    message(STATUS "[3/5] Using system SUNDIALS ${_sun_ver} (${_SUN_INC}). "
                   "Set -DBUILD_SUNDIALS=ON to build the bundled one instead.")
  elseif(_sun_explicit)
    message(
      FATAL_ERROR
        "SUNDIALS_DIR=${SUNDIALS_DIR} does not provide SUNDIALS >= 5.4.0 with the "
        "required libraries (arkode, kinsol, nvecserial). Use -DBUILD_SUNDIALS=ON "
        "to build the bundled SUNDIALS.")
  else()
    set(_build_sundials ON)
    message(
      STATUS "[3/5] No system SUNDIALS >= 5.4.0 found; building the bundled SUNDIALS."
    )
  endif()
endif()

if(_build_sundials)
  set(SUNDIALS_PREFIX ${SUNDIALS_INSTALL})
  set(_sundials_dep_target sundials_dep)
  message(STATUS "[3/5] Configuring SUNDIALS (bundled, replicating external/build_sundials.sh)...")
  ExternalProject_Add(
    sundials_dep
    GIT_REPOSITORY ${SUNDIALS_REPO}
    GIT_TAG ${SUNDIALS_TAG}
    GIT_SHALLOW ON
    PREFIX ${CMAKE_BINARY_DIR}/sundials
    INSTALL_DIR ${SUNDIALS_INSTALL}
    LOG_DOWNLOAD ON
    LOG_CONFIGURE ON
    LOG_BUILD ON
    LOG_INSTALL ON
    LOG_OUTPUT_ON_FAILURE ON
    CMAKE_CACHE_ARGS
      -DENABLE_OPENMP:BOOL=OFF
      -DENABLE_MPI:BOOL=OFF
      -DCMAKE_BUILD_TYPE:STRING=Release
      -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
      -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
      "-DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS} -O3 -DNDEBUG"
      "-DCMAKE_EXE_LINKER_FLAGS:STRING=${DIFFRG_EXE_LINKER_FLAGS}"
      -DCMAKE_INSTALL_PREFIX:PATH=${SUNDIALS_INSTALL}
      ${CCACHE_CMAKE_ARGS})
endif()

# ##############################################################################
# [4/5] deal.II (replicating external/build_dealii.sh)
# ##############################################################################

# RAM-aware cap on deal.II parallelism (linking can OOM). Use the smallest of:
# the requested thread count, ~one job per 2 GB of RAM, and DEALII_MAX_JOBS.
cmake_host_system_information(RESULT _total_ram_mb QUERY TOTAL_PHYSICAL_MEMORY)
math(EXPR _max_threads_by_ram "${_total_ram_mb} / 2000")
if(_max_threads_by_ram LESS 1)
  set(_max_threads_by_ram 1)
endif()
set(_dealii_jobs ${BUILD_THREADS})
if(_max_threads_by_ram LESS _dealii_jobs)
  set(_dealii_jobs ${_max_threads_by_ram})
endif()
if(_dealii_jobs GREATER ${DEALII_MAX_JOBS})
  set(_dealii_jobs ${DEALII_MAX_JOBS})
endif()
message(
  STATUS
    "deal.II build parallelism: ${_dealii_jobs} (RAM ${_total_ram_mb} MB, cap ${DEALII_MAX_JOBS})"
)

# Normalize the `config` file's DEAL_II_CMAKE string into proper -D cmake args.
# (The historical default omits the leading dash on the first flag.)
set(_dealii_extra_args "")
if(NOT "${DEAL_II_CMAKE}" STREQUAL "")
  separate_arguments(_dealii_tokens UNIX_COMMAND "${DEAL_II_CMAKE}")
  foreach(_tok IN LISTS _dealii_tokens)
    if(NOT _tok MATCHES "^-")
      set(_tok "-${_tok}")
    endif()
    list(APPEND _dealii_extra_args "${_tok}")
  endforeach()
endif()

# deal.II build type. Default to Release for faster compilation; enable DEVEL to
# get a DebugRelease build (debug + release libraries) for library development.
option(DEVEL "Build deal.II in DebugRelease mode (slower; for development)" OFF)
if(DEVEL)
  set(_dealii_build_type DebugRelease)
else()
  set(_dealii_build_type Release)
endif()
message(STATUS "deal.II build type: ${_dealii_build_type}")

# Route TBB/SUNDIALS to deal.II only when a specific prefix is known (bundled
# build or an explicit *_DIR). For a system install in standard paths the prefix
# is empty and deal.II finds it by its own path search without a hint.
set(_dealii_tbb_arg "")
if(NOT "${TBB_PREFIX}" STREQUAL "")
  set(_dealii_tbb_arg -DTBB_DIR:PATH=${TBB_PREFIX})
endif()
set(_dealii_sundials_arg "")
if(NOT "${SUNDIALS_PREFIX}" STREQUAL "")
  set(_dealii_sundials_arg -DSUNDIALS_DIR:PATH=${SUNDIALS_PREFIX})
endif()
# DiFfRG's find_package(TBB ... HINTS .../oneTBB_install) is config mode; route a
# bundled/user TBB via TBB_ROOT (a system TBB in standard paths is found anyway).
set(_diffrg_tbb_arg "")
if(NOT "${TBB_PREFIX}" STREQUAL "")
  set(_diffrg_tbb_arg -DTBB_ROOT:PATH=${TBB_PREFIX})
endif()

message(STATUS "[4/5] Configuring deal.II...")
ExternalProject_Add(
  deal.II_dep
  DOWNLOAD_COMMAND
    ${CMAKE_COMMAND} -DGIT_EXECUTABLE=${GIT_EXECUTABLE} -DREPO=${DEALII_REPO}
    -DTAG=${DEALII_TAG} -DDEST=<SOURCE_DIR> -P
    ${CMAKE_SOURCE_DIR}/cmake/fetch_commit.cmake
  PREFIX ${CMAKE_BINARY_DIR}/deal.II
  INSTALL_DIR ${DEALII_INSTALL}
  DEPENDS ${_boost_dep_target} ${_tbb_dep_target} ${_sundials_dep_target}
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_OUTPUT_ON_FAILURE ON
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> -j ${_dealii_jobs}
  CMAKE_ARGS ${_dealii_extra_args}
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${_dealii_build_type}
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    -DCMAKE_Fortran_COMPILER:STRING=${CMAKE_Fortran_COMPILER}
    "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}"
    "-DCMAKE_EXE_LINKER_FLAGS:STRING=${DIFFRG_EXE_LINKER_FLAGS}"
    -DCMAKE_CXX_STANDARD:STRING=17
    -DCMAKE_WARN_DEPRECATED:BOOL=OFF
    # deal.II's FindDEAL_II_BOOST sets Boost_NO_BOOST_CMAKE and expects the
    # classic module-mode FindBoost (which creates the Boost::<component>
    # targets). Force CMP0167 OLD so find_package(Boost) uses that module rather
    # than BoostConfig.
    -DCMAKE_POLICY_DEFAULT_CMP0167:STRING=OLD
    -DDEAL_II_COMPONENT_EXAMPLES:BOOL=OFF
    -DDEAL_II_COMPONENT_DOCUMENTATION:BOOL=OFF
    -DDEAL_II_WITH_MPI:BOOL=OFF
    -DDEAL_II_WITH_UMFPACK:BOOL=ON
    -DDEAL_II_WITH_TASKFLOW:BOOL=OFF
    -DDEAL_II_WITH_VTK:BOOL=OFF
    ${_dealii_tbb_arg}
    ${_dealii_sundials_arg}
    # deal.II and DiFfRG must use the SAME Boost (deal.II asserts this). Point
    # deal.II at the resolved Boost prefix (bundled build or system/user Boost).
    # CMP0167 OLD (above) makes find_package(Boost) use the module-mode
    # FindBoost that deal.II's FindDEAL_II_BOOST expects.
    -DBOOST_DIR:PATH=${BOOST_PREFIX}
    -DBOOST_INCLUDEDIR:PATH=${BOOST_PREFIX}/include
    -DBOOST_LIBRARYDIR:PATH=${BOOST_PREFIX}/lib
    -DCMAKE_PREFIX_PATH:STRING=${BOOST_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${DEALII_INSTALL}
    ${CCACHE_CMAKE_ARGS})

# ##############################################################################
# [5/5] DiFfRG library
# ##############################################################################

# Build the documentation only if Doxygen is available.
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
  set(_diffrg_doc ON)
else()
  set(_diffrg_doc OFF)
  message(
    STATUS
      "Doxygen not found; DiFfRG documentation will not be built. (Install doxygen + graphviz to enable.)"
  )
endif()

# Build the default targets, then (optionally) the documentation target before
# the install step picks up the generated docs.
set(_diffrg_build_cmd ${CMAKE_COMMAND} --build <BINARY_DIR> -j ${BUILD_THREADS})
if(_diffrg_doc)
  set(_diffrg_doc_cmd COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target
                      documentation)
else()
  set(_diffrg_doc_cmd "")
endif()

message(STATUS "[5/5] Configuring DiFfRG...")
ExternalProject_Add(
  DiFfRG
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/DiFfRG
  PREFIX ${CMAKE_BINARY_DIR}/DiFfRG
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  DEPENDS ${_boost_dep_target} ${_tbb_dep_target} ${_sundials_dep_target}
          deal.II_dep
  BUILD_ALWAYS ON
  LOG_CONFIGURE ON
  LOG_OUTPUT_ON_FAILURE ON
  BUILD_COMMAND ${_diffrg_build_cmd} ${_diffrg_doc_cmd}
  CMAKE_CACHE_ARGS
    -DBUNDLED_DIR:PATH=${DEPEND_INSTALL_PREFIX}
    # Resolve the SAME Boost as deal.II. setup_build_system.cmake searches
    # ${BUNDLED_DIR}/boost_install first (present only for a bundled build);
    # these hints route it to a system/user Boost otherwise.
    -DBoost_ROOT:PATH=${BOOST_PREFIX}
    -DCMAKE_PREFIX_PATH:STRING=${BOOST_PREFIX}
    ${_diffrg_tbb_arg}
    -DUSE_CUDA:BOOL=${USE_CUDA}
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
    "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}"
    "-DCMAKE_CUDA_FLAGS:STRING=${CMAKE_CUDA_FLAGS}"
    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
    -DDiFfRG_BUILD_TESTS:BOOL=OFF
    -DDiFfRG_BUILD_DOCUMENTATION:BOOL=${_diffrg_doc}
    -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX})

# ##############################################################################
# Finish
# ##############################################################################

add_custom_target(
  finish ALL
  DEPENDS DiFfRG
  COMMAND ${CMAKE_COMMAND} -E echo ""
  COMMAND ${CMAKE_COMMAND} -E echo "DiFfRG installed to ${CMAKE_INSTALL_PREFIX}"
  VERBATIM)
