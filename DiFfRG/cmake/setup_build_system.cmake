# ##############################################################################
# Setup directories
# ##############################################################################

# We need to find the bundle directory, which contains several external
# dependencies
if(${CMAKE_PROJECT_NAME} STREQUAL "DiFfRG")
  # If we are building DiFfRG as a standalone project, we need to set the base
  # directory. (Its MPI option is resolved below, once the bundle's pin is loaded.)
  set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
else()
  # If we are building a DiFfRG-based project, we need to set the bundle
  # directory relative to the DiFfRG base directory
  set(BASE_DIR ${DiFfRG_BASE_DIR})
endif()

# Target CPU architecture. MARCH selects an explicit -march= value
# ("x86-64-v3", "native", or "none" for no arch flag); when empty (default) the
# legacy NATIVE bool decides. The superbuild forwards both so a standalone
# library build behaves identically. Resolved inline (not via
# diffrg_resolve_march) because consumer projects include this file without
# common.cmake.
option(NATIVE "Optimize for the build machine's CPU (-march=native). Disable for portable binaries." ON)
set(MARCH
    ""
    CACHE STRING
          "Explicit -march target (e.g. x86-64-v3, native, none). Overrides NATIVE.")
if(NOT MARCH STREQUAL "")
  set(DiFfRG_MARCH_VALUE "${MARCH}")
elseif(NATIVE)
  set(DiFfRG_MARCH_VALUE "native")
else()
  set(DiFfRG_MARCH_VALUE "none")
endif()
if(COMMAND diffrg_report_arch)
  diffrg_report_arch("${DiFfRG_MARCH_VALUE}" "${CMAKE_CXX_COMPILER}")
endif()

# ##############################################################################
# Validate BUNDLED_DIR
# ##############################################################################

if(NOT DEFINED BUNDLED_DIR OR "${BUNDLED_DIR}" STREQUAL "")
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  BUNDLED_DIR is not set.\n"
      "======================================================================\n"
      "  DiFfRG needs to know where its bundled dependencies are installed.\n"
      "\n"
      "  If you have not built the dependencies yet, build from the top-level\n"
      "  repository directory first:\n"
      "    mkdir build && cd build\n"
      "    cmake .. -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG\n"
      "    cmake --build . -- -j8\n"
      "\n"
      "  Then configure DiFfRG with:\n"
      "    cmake .. -DBUNDLED_DIR=~/.local/share/DiFfRG/bundled\n"
      "======================================================================\n"
  )
endif()

if(NOT EXISTS "${BUNDLED_DIR}")
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  BUNDLED_DIR does not exist: ${BUNDLED_DIR}\n"
      "======================================================================\n"
      "  The specified dependency directory was not found. This usually means\n"
      "  the dependencies have not been built yet.\n"
      "\n"
      "  Build from the top-level repository directory:\n"
      "    mkdir build && cd build\n"
      "    cmake .. -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG\n"
      "    cmake --build . -- -j8\n"
      "\n"
      "  Then re-run this cmake configuration.\n"
      "======================================================================\n"
  )
endif()

set(CMAKE_PREFIX_PATH "${BUNDLED_DIR};${BUNDLED_DIR}/lib;${CMAKE_PREFIX_PATH}")

# ##############################################################################
# Pinned dependency configuration
# ##############################################################################
#
# The superbuild records the Boost/TBB/HDF5 it resolved (system vs bundled) in
# DiFfRG_bundled_config.cmake inside the bundle dir. Load it before the
# find_package calls below so this build -- whether a standalone library rebuild
# or a downstream find_package(DiFfRG) -- reuses exactly those dependencies
# instead of re-resolving and possibly picking a different system install or a
# stray copy. The pin uses if(NOT DEFINED) guards, so an explicit -DX= still
# wins. It also records DiFfRG_PINNED_<X>_VERSION, checked after each find below.
set(_diffrg_pin "${BUNDLED_DIR}/DiFfRG_bundled_config.cmake")
if(EXISTS "${_diffrg_pin}")
  message(STATUS "Loading pinned dependency configuration: ${_diffrg_pin}")
  include("${_diffrg_pin}")
endif()

# A standalone library build defaults MPI to whatever the bundle was built with: deal.II's
# DEAL_II_WITH_MPI has to match it anyway (checked below). An explicit -DMPI= still wins.
if(${CMAKE_PROJECT_NAME} STREQUAL "DiFfRG")
  set(_diffrg_mpi_default OFF)
  if(DEFINED DiFfRG_PINNED_MPI)
    set(_diffrg_mpi_default ${DiFfRG_PINNED_MPI})
  endif()
  set(MPI
      ${_diffrg_mpi_default}
      CACHE BOOL "Whether to build with MPI support (default: as the dependency bundle)")
  set(DiFfRG_MPI ${MPI})
endif()

# The pin sets the upper-case BOOST_ROOT (the convention used by the superbuild
# and deal.II). CMake >= 3.27 only honors upper-case <PKG>_ROOT when CMP0144 is
# NEW; otherwise find_package ignores it and warns. Opt in here so BOOST_ROOT is
# respected. Set before find_package(Boost) is invoked below.
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()
if(POLICY CMP0156)
  cmake_policy(SET CMP0156 NEW)
endif()
if(POLICY CMP0179)
  cmake_policy(SET CMP0179 NEW)
endif()

message(STATUS "DiFfRG include directory: ${BASE_DIR}/include")
message(STATUS "DiFfRG bundle directory: ${BUNDLED_DIR}")
message(STATUS "MPI support has been set to ${DiFfRG_MPI}")

# ##############################################################################
# Guard: environment include paths that shadow the bundle
# ##############################################################################
#
# GCC and Clang search the directories named by CPATH / C_INCLUDE_PATH /
# CPLUS_INCLUDE_PATH *before* every -isystem directory, and setup_target() below
# adds the bundle with -isystem. So anything on those variables -- a cluster
# environment module, a conda environment, a hand-set CPATH -- quietly takes
# precedence over ${BUNDLED_DIR}/include for every header it also provides,
# while the link still resolves against the bundled libraries. The build then
# compiles and links cleanly against two different copies of one dependency.
#
# Warn here, where it is still cheap to fix, and name the shadowed headers.
# Only headers the bundle actually provides are reported, so an unrelated CPATH
# entry stays quiet.
function(_diffrg_warn_shadowed_includes)
  set(_sentinels
      "hdf5.h"
      "boost/version.hpp"
      "Eigen/Core"
      "tbb/version.h"
      "spdlog/version.h"
      "deal.II/base/config.h"
      "autodiff/forward/dual.hpp"
      "sundials/sundials_version.h"
      "Kokkos_Core.hpp")

  if(NOT IS_DIRECTORY "${BUNDLED_DIR}/include")
    return()
  endif()
  file(REAL_PATH "${BUNDLED_DIR}" _real_bundle)

  set(_hits "")
  foreach(_var CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH)
    if("$ENV{${_var}}" STREQUAL "")
      continue()
    endif()
    string(REPLACE ":" ";" _dirs "$ENV{${_var}}")
    foreach(_dir IN LISTS _dirs)
      if("${_dir}" STREQUAL "" OR NOT IS_DIRECTORY "${_dir}")
        continue()
      endif()
      # A directory inside the bundle cannot shadow the bundle.
      file(REAL_PATH "${_dir}" _real_dir)
      string(FIND "${_real_dir}" "${_real_bundle}" _inside)
      if(_inside EQUAL 0)
        continue()
      endif()
      foreach(_s IN LISTS _sentinels)
        if(EXISTS "${_dir}/${_s}" AND EXISTS "${BUNDLED_DIR}/include/${_s}")
          list(APPEND _hits
               "    ${_s}\n        shadowed by ${_dir}/${_s}\n        (via ${_var})")
        endif()
      endforeach()
    endforeach()
  endforeach()

  if(_hits)
    list(REMOVE_DUPLICATES _hits)
    string(REPLACE ";" "\n" _hits "${_hits}")
    message(
      WARNING
        "\n"
        "======================================================================\n"
        "  Environment include paths shadow the DiFfRG bundle\n"
        "======================================================================\n"
        "  The bundle provides these headers, but the compiler will open the\n"
        "  copy reached through the environment instead -- CPATH and friends\n"
        "  are searched before every -isystem directory:\n"
        "\n"
        "${_hits}\n"
        "\n"
        "  The link still resolves against ${BUNDLED_DIR},\n"
        "  so the two halves of the build can disagree at runtime.\n"
        "\n"
        "  Unload the environment module or deactivate the environment that\n"
        "  adds these paths, then reconfigure in a clean build directory.\n"
        "======================================================================\n")
  endif()
endfunction()

_diffrg_warn_shadowed_includes()

# ##############################################################################
# Set standard and language
# ##############################################################################

set(CMAKE_CXX_STANDARD_REQUIRED On)
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
else()
  if(CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "C++ standard must be at least 20")
  endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_CXX_EXTENSIONS OFF)
enable_language(CXX)

# By default, we build in Release mode, i.e. if the user does not make any other
# choice. After all, even if the user is unaware of cmake build types, we want
# to provide optimal performance.
if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type not set, defaulting to Release")
endif()

# ##############################################################################
# Helper macro for find_package with actionable errors
# ##############################################################################

macro(diffrg_find_package pkg)
  # Parse optional arguments: version and extra hints
  cmake_parse_arguments(_DFP "" "VERSION" "HINTS;COMPONENTS" ${ARGN})

  set(_dfp_args "")
  if(_DFP_VERSION)
    list(APPEND _dfp_args "${_DFP_VERSION}")
  endif()
  list(APPEND _dfp_args QUIET)
  if(_DFP_HINTS)
    list(APPEND _dfp_args HINTS ${_DFP_HINTS})
  endif()
  if(_DFP_COMPONENTS)
    list(APPEND _dfp_args COMPONENTS ${_DFP_COMPONENTS})
  endif()

  find_package(${pkg} ${_dfp_args})

  if(NOT ${pkg}_FOUND)
    message(
      FATAL_ERROR
        "\n"
        "======================================================================\n"
        "  Required dependency not found: ${pkg}\n"
        "======================================================================\n"
        "  CMake could not find '${pkg}' in BUNDLED_DIR=${BUNDLED_DIR}\n"
        "\n"
        "  This usually means the bundled dependencies need to be (re)built.\n"
        "  Build from the top-level repository directory:\n"
        "    mkdir build && cd build\n"
        "    cmake .. -DCMAKE_INSTALL_PREFIX=~/.local/share/DiFfRG\n"
        "    cmake --build . -- -j8\n"
        "\n"
        "  Then re-run this cmake configuration.\n"
        "======================================================================\n"
    )
  endif()
endmacro()

# ##############################################################################
# Direct external dependencies
# ##############################################################################

if(APPLE)
  include(CheckLinkerFlag)
  check_linker_flag(CXX "LINKER:-no_warn_duplicate_libraries"
                    DiFfRG_LINKER_SUPPORTS_NO_WARN_DUPLICATE_LIBRARIES)
endif()

# Find deal.II
diffrg_find_package(deal.II VERSION 9.4.2 HINTS ${BUNDLED_DIR})
deal_ii_initialize_cached_variables()
message(STATUS "Found deal.II in  ${deal.II_DIR}")

# ----------------------------------------------------------------------------
# Drop deal.II's optimization level from its imported target
# ----------------------------------------------------------------------------
# dealii::dealii carries deal.II's build flags in INTERFACE_COMPILE_OPTIONS,
# including a per-config optimization level (e.g. "$<$<CONFIG:Release>:-O2;
# -funroll-loops;...>"). Every target that links it therefore receives that -O
# *in addition to* the one CMake itself puts in CMAKE_CXX_FLAGS_<CONFIG>
# (-O3 -DNDEBUG for Release), so the compile line ends up with two -O flags.
#
# CMake de-duplicates the other repeated deal.II flags against the copy that
# setup_dealii() adds by hand, but not the -O: setup_dealii() strips it from its
# copy (see there), leaving the interface's -O as the sole survivor. On the GCC +
# CUDA path -- where Kokkos routes compilation through nvcc_wrapper -- nvcc then
# warns ("you have set multiple optimization flags ... only the last is used")
# and silently keeps the *last* one, downgrading Release objects to deal.II's
# -O2 instead of CMake's -O3.
#
# Strip the optimization level here, once, so CMAKE_CXX_FLAGS_<CONFIG> is the
# single source of truth for it. Everything else deal.II exports is kept.
#
# The property is a ";"-list that also contains generator expressions, so an -O
# token may be glued to a genex head ("$<$<CONFIG:Release>>:-O2"). Delete the
# tokens textually, then collapse the separators they leave behind -- an empty
# list entry would otherwise reach the compiler as an empty argument.
get_target_property(_dealii_iface_opts dealii::dealii INTERFACE_COMPILE_OPTIONS)
if(_dealii_iface_opts)
  string(REGEX REPLACE "^-O[0-9a-zA-Z]+;" "" _dealii_iface_opts
                       "${_dealii_iface_opts}")
  string(REGEX REPLACE "([:;])-O[0-9a-zA-Z]+" "\\1" _dealii_iface_opts
                       "${_dealii_iface_opts}")
  string(REPLACE ":;" ":" _dealii_iface_opts "${_dealii_iface_opts}")
  string(REPLACE ";;" ";" _dealii_iface_opts "${_dealii_iface_opts}")
  string(REPLACE ";>" ">" _dealii_iface_opts "${_dealii_iface_opts}")
  set_target_properties(dealii::dealii PROPERTIES INTERFACE_COMPILE_OPTIONS
                                                  "${_dealii_iface_opts}")
endif()

# Find TBB. TBB_DIR (set by the top-level build, or by the user) selects bundled
# vs system; DiFfRG requires oneTBB >= 2021.
diffrg_find_package(TBB VERSION 2021 HINTS ${BUNDLED_DIR})
message(STATUS "Found TBB in ${TBB_DIR}")
if(DEFINED DiFfRG_PINNED_TBB_VERSION
   AND NOT TBB_VERSION VERSION_EQUAL DiFfRG_PINNED_TBB_VERSION)
  message(
    WARNING
      "TBB version drift: the superbuild pinned ${DiFfRG_PINNED_TBB_VERSION} but this build "
      "found ${TBB_VERSION} (${TBB_DIR}). The dependency changed since the bundle was built. "
      "If you hit link/ABI errors, rebuild the bundled dependencies.")
endif()

# Find Kokkos
diffrg_find_package(Kokkos HINTS ${BUNDLED_DIR})
message(STATUS "Found Kokkos in ${Kokkos_DIR}")

# ##############################################################################
# CUDA architecture selection
# ##############################################################################
#
# The bundled Kokkos bakes its GPU architecture into its interface as
# "-arch=sm_XX", and deal.II flattens that into dealii::interface_kokkos. Left
# alone, every application therefore compiles for whatever architecture the
# *dependency bundle* was built with -- typically not the GPU it will run on.
# nvcc then emits no loadable SASS for the local device and the driver falls
# back to JIT-compiling the embedded PTX at module load. DiFfRG's generated flow
# kernels are far too large for that to be acceptable: they overflow the
# driver's 256 MB JIT cache (~/.nv/ComputeCache), so the compile is repeated on
# every single run.
#
# DiFfRG_CUDA_ARCH retargets that inherited flag, so the library and every
# application are compiled as native SASS for the GPUs actually in use, while
# the bundle stays one portable prebuilt artifact. Only the bundle's own (small)
# Kokkos and deal.II kernels are left to JIT.
#
# A lower-arch bundle on a newer GPU is a supported combination: Kokkos aborts
# only when its compiled architecture *exceeds* the device
# (Kokkos_Cuda_Instance.cpp), and merely warns in the other direction.

set(DiFfRG_CUDA_ARCH
    ""
    CACHE
      STRING
      "CUDA compute capabilities to compile for, e.g. \"90\" or \"80;90\". Also accepts \"native\" (detect the local GPUs) and \"bundled\" (keep the dependency bundle's architecture)."
)

include(${CMAKE_CURRENT_LIST_DIR}/cuda_arch.cmake)

# The inherited flag is also how we tell whether this bundle has CUDA at all --
# there is no other record of it in the installed dependency tree.
set(DiFfRG_CUDA_ARCH_RESOLVED "")
set(DiFfRG_CUDA_ARCH_BUNDLED "")
get_target_property(_diffrg_kokkos_opts dealii::interface_kokkos
                    INTERFACE_COMPILE_OPTIONS)
if(_diffrg_kokkos_opts MATCHES "-arch=sm_([0-9]+)")
  set(DiFfRG_CUDA_ARCH_BUNDLED "${CMAKE_MATCH_1}")

  set(_requested "${DiFfRG_CUDA_ARCH}")
  if(_requested STREQUAL "")
    # An application defaults to whatever the library it links was built for, so
    # the two always agree without probing again; the library itself detects.
    if(DEFINED DiFfRG_CUDA_ARCH_LIBRARY AND NOT DiFfRG_CUDA_ARCH_LIBRARY
                                            STREQUAL "")
      set(_requested "${DiFfRG_CUDA_ARCH_LIBRARY}")
    else()
      set(_requested "native")
    endif()
  endif()

  if(_requested STREQUAL "native")
    _diffrg_detect_cuda_arch(_requested)
    if(_requested STREQUAL "")
      message(
        STATUS
          "${BoldYellow}[CUDA] No GPU could be queried on this machine, so the dependency bundle's "
          "sm_${DiFfRG_CUDA_ARCH_BUNDLED} is used. Kernels will be JIT-compiled at startup on any other "
          "GPU -- pass -DDiFfRG_CUDA_ARCH=<capability> (e.g. 90) to compile for the target directly.${ColourReset}"
      )
      set(_requested "bundled")
    endif()
  endif()

  if(NOT _requested STREQUAL "bundled")
    _diffrg_normalize_cuda_arch(DiFfRG_CUDA_ARCH_RESOLVED ${_requested})

    # Retargeting moves DiFfRG's kernels but not the bundle's, and Kokkos aborts
    # rather than JITs when its own architecture is above the device. Asking for
    # a GPU older than the bundle therefore builds cleanly and then fails at
    # startup with a Kokkos abort, well away from the cause.
    list(GET DiFfRG_CUDA_ARCH_RESOLVED 0 _diffrg_arch_min)
    if(_diffrg_arch_min LESS DiFfRG_CUDA_ARCH_BUNDLED)
      message(
        WARNING
          "DiFfRG_CUDA_ARCH asks for sm_${_diffrg_arch_min}, but the dependency bundle was built for "
          "sm_${DiFfRG_CUDA_ARCH_BUNDLED}. DiFfRG's own kernels will be correct, but Kokkos aborts at "
          "startup on a GPU older than the architecture it was compiled for, so this build will not run "
          "there. Install a bundle whose architecture is at most sm_${_diffrg_arch_min} (the pre-built "
          "CUDA bundle is sm_75), or build the dependencies from source with -DKokkos_ARCH_LIST=.")
    endif()

    _diffrg_cuda_arch_flags(_diffrg_arch_flags ${DiFfRG_CUDA_ARCH_RESOLVED})
    foreach(_target dealii::interface_kokkos Kokkos::kokkoscore)
      foreach(_property INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_OPTIONS)
        _diffrg_retarget_cuda_arch(${_target} ${_property}
                                   "${_diffrg_arch_flags}")
      endforeach()
    endforeach()
  else()
    set(DiFfRG_CUDA_ARCH_RESOLVED "${DiFfRG_CUDA_ARCH_BUNDLED}")
  endif()

  # "sm_80,sm_90" -- for the summary table and for DiFfRG::Init()'s startup
  # line, which has to correct Kokkos' own architecture warning: Kokkos probes
  # with a kernel that lives in libkokkoscore, so it reports the bundle's
  # architecture and says nothing about the kernels that actually matter.
  # No spaces: this also becomes a preprocessor define, and a space would split
  # the compiler argument in two.
  set(_diffrg_arch_names "")
  foreach(_cc IN LISTS DiFfRG_CUDA_ARCH_RESOLVED)
    list(APPEND _diffrg_arch_names "sm_${_cc}")
  endforeach()
  list(JOIN _diffrg_arch_names "," DiFfRG_CUDA_ARCH_DISPLAY)
endif()

# CUDA stub-RUNPATH guard. Offline (driverless) builds link the CUDA driver
# API against the toolkit's *stub* libcuda.so; that is fine -- dynamic linking
# records only the SONAME -- unless the stubs directory leaks into a RUNPATH,
# in which case the runtime loader picks the stub over the real driver and
# every CUDA call fails (CUDA_ERROR_STUB_LIBRARY). That poisoning is silent at
# build time and only explodes on the compute node, so refuse it here.
if(UNIX
   AND NOT APPLE
   AND Kokkos_ENABLE_CUDA)
  find_program(_diffrg_readelf readelf)
  if(_diffrg_readelf)
    file(GLOB _diffrg_bundle_sos "${BUNDLED_DIR}/lib/*.so*"
         "${BUNDLED_DIR}/lib64/*.so*")
    set(_diffrg_stub_poisoned "")
    foreach(_so ${_diffrg_bundle_sos})
      if(NOT IS_SYMLINK "${_so}")
        execute_process(
          COMMAND ${_diffrg_readelf} -d "${_so}"
          OUTPUT_VARIABLE _dyn
          ERROR_QUIET)
        if(_dyn MATCHES "(RPATH|RUNPATH)[^\n]*stubs")
          list(APPEND _diffrg_stub_poisoned "${_so}")
        endif()
      endif()
    endforeach()
    if(_diffrg_stub_poisoned)
      list(JOIN _diffrg_stub_poisoned "\n    " _diffrg_stub_poisoned)
      message(
        FATAL_ERROR
          "\n"
          "======================================================================\n"
          "  CUDA stub directory found in the RUNPATH of bundled libraries:\n"
          "    ${_diffrg_stub_poisoned}\n"
          "======================================================================\n"
          "  At runtime the loader would pick the CUDA driver *stub* over the\n"
          "  real driver and every CUDA call would fail. Fix the libraries with\n"
          "    patchelf --set-rpath '<rpath without the stubs entry>' <lib>\n"
          "  and, when building offline, expose the stubs to the linker only\n"
          "  via LIBRARY_PATH or -Wl,-rpath-link -- never -Wl,-rpath or\n"
          "  LD_LIBRARY_PATH (see the installation documentation).\n"
          "======================================================================\n"
      )
    endif()
  endif()
endif()

# Find Boost. find_package also honors BOOST_ROOT/Boost_DIR and standard system
# paths, so a system Boost (selected via BOOST_DIR/BUILD_BOOST in the top-level
# build) is picked up here when BUNDLED_DIR does not contain one. Use Boost's own
# BoostConfig.cmake (config mode); the legacy FindBoost module is removed in
# CMake >= 3.30. Boost has shipped BoostConfig.cmake since 1.70, and DiFfRG
# requires >= 1.81, so config mode always applies.
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
diffrg_find_package(
  Boost
  VERSION
  1.81
  HINTS
  "${BUNDLED_DIR}/"
  "${BUNDLED_DIR}/boost_install/lib/"
  COMPONENTS
  thread
  iostreams
  serialization
  system)
message(STATUS "Boost version: ${Boost_VERSION}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
# Boost is ABI-critical: a version divergence from what the superbuild pinned
# (e.g. a system Boost upgraded in place after the bundle was built) is a hard
# error rather than a warning.
if(DEFINED DiFfRG_PINNED_BOOST_VERSION
   AND NOT Boost_VERSION VERSION_EQUAL DiFfRG_PINNED_BOOST_VERSION)
  message(
    FATAL_ERROR
      "Boost version mismatch: the superbuild pinned ${DiFfRG_PINNED_BOOST_VERSION} but this "
      "build found ${Boost_VERSION} (${Boost_DIR}). The dependency changed since the bundle was "
      "built (e.g. a system upgrade). Rebuild the bundled dependencies, or pass an explicit "
      "-DBoost_DIR= / -DBOOST_ROOT= pointing at Boost ${DiFfRG_PINNED_BOOST_VERSION}.")
endif()

# Find Eigen3
diffrg_find_package(Eigen3 VERSION 3.4.0 HINTS ${BUNDLED_DIR})
if(TARGET Eigen3::Eigen)
  set(DiFfRG_EIGEN_TARGET Eigen3::Eigen)
else()
  set(DiFfRG_EIGEN_TARGET Eigen3)
endif()

# Find GSL (system dependency)
find_package(GSL QUIET)
if(NOT GSL_FOUND)
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  Required system dependency not found: GSL\n"
      "======================================================================\n"
      "  The GNU Scientific Library (GSL) must be installed on your system.\n"
      "\n"
      "  Install it using your package manager:\n"
      "    Ubuntu/Debian:  sudo apt install libgsl-dev\n"
      "    Arch Linux:     sudo pacman -S gsl\n"
      "    Rocky/RHEL:     sudo dnf install gsl-devel\n"
      "    macOS:          brew install gsl\n"
      "======================================================================\n"
  )
endif()

# Find autodiff
diffrg_find_package(autodiff VERSION 1.1.0 HINTS ${BUNDLED_DIR})

# Find spdlog
diffrg_find_package(spdlog VERSION 1.14.1 HINTS ${BUNDLED_DIR})

# Find HDF5. DiFfRG uses only the HDF5 C API, so 1.12 is the floor. Prefer config
# mode so the imported targets are exported; HDF5_DIR (set by the top-level build,
# or by the user) selects bundled vs system. Do not pass the version to
# find_package: HDF5's config-version file uses a same-major-version policy, so
# requesting 1.12 would reject a newer 2.x install; gate the version manually.
# Config mode first (bundled static build + distros that ship a CMake config,
# e.g. Arch); then module mode (FindHDF5) for config-less system installs
# (Fedora/Debian/Ubuntu). HDF5_DIR/HDF5_ROOT are set by the top-level build.
option(HDF5 "Enable HDF5 support" ON)
find_package(HDF5 CONFIG QUIET COMPONENTS C HINTS ${BUNDLED_DIR})
if(NOT HDF5_FOUND OR HDF5_VERSION VERSION_LESS 1.12.0)
  find_package(HDF5 MODULE QUIET COMPONENTS C)
endif()
if(NOT HDF5_FOUND OR HDF5_VERSION VERSION_LESS 1.12.0)
  message(
    FATAL_ERROR
      "\n"
      "======================================================================\n"
      "  Required dependency not found: HDF5 >= 1.12 (found '${HDF5_VERSION}')\n"
      "======================================================================\n"
      "  CMake could not find an HDF5 (>= 1.12) config in BUNDLED_DIR=${BUNDLED_DIR}\n"
      "  or via HDF5_DIR. Build the bundled dependencies, install a system HDF5,\n"
      "  or pass -DHDF5_DIR=<prefix-with-hdf5-config.cmake>.\n"
      "======================================================================\n")
endif()
message(STATUS "HDF5 version: ${HDF5_VERSION}")
message(STATUS "HDF5 include dir: ${HDF5_INCLUDE_DIRS}")

# Resolve the HDF5 link target: the bundled static build exports hdf5-static;
# system installs vary (hdf5-shared / hdf5::hdf5 / HDF5::HDF5), or only set vars.
if(TARGET hdf5-static)
  set(DiFfRG_HDF5_LIBRARIES hdf5-static)
elseif(TARGET hdf5::hdf5-static)
  set(DiFfRG_HDF5_LIBRARIES hdf5::hdf5-static)
elseif(TARGET hdf5-shared)
  set(DiFfRG_HDF5_LIBRARIES hdf5-shared)
elseif(TARGET hdf5::hdf5)
  set(DiFfRG_HDF5_LIBRARIES hdf5::hdf5)
elseif(TARGET HDF5::HDF5)
  set(DiFfRG_HDF5_LIBRARIES HDF5::HDF5)
else()
  set(DiFfRG_HDF5_LIBRARIES ${HDF5_C_LIBRARIES} ${HDF5_LIBRARIES})
  set(DiFfRG_HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIRS})
endif()
message(STATUS "HDF5 link target(s): ${DiFfRG_HDF5_LIBRARIES}")
if(DEFINED DiFfRG_PINNED_HDF5_VERSION
   AND NOT HDF5_VERSION VERSION_EQUAL DiFfRG_PINNED_HDF5_VERSION)
  message(
    WARNING
      "HDF5 version drift: the superbuild pinned ${DiFfRG_PINNED_HDF5_VERSION} but this build "
      "found ${HDF5_VERSION}. The dependency changed since the bundle was built. "
      "If you hit link/ABI errors, rebuild the bundled dependencies.")
endif()

# ##############################################################################
# Guard: the hdf5.h the compiler opens must match the HDF5 we link
# ##############################################################################
#
# find_package() above resolved which HDF5 will be *linked*. It says nothing
# about which hdf5.h the compiler will actually *open*: setup_target() adds the
# bundle with -isystem, and CPATH / C_INCLUDE_PATH / CPLUS_INCLUDE_PATH are
# searched first (see the shadowing warning near the top of this file). Nothing
# goes wrong at compile or link time either, because H5public.h only bakes the
# header's version numbers into the object file. The mismatch surfaces at
# *runtime*:
#
#   Warning! ***HDF5 library version mismatched error***
#   Headers are 1.14.6, library is 2.0.0
#
# and HDF5 means it -- public struct layouts and the versioned-API macro mapping
# differ across major versions, so this is an ABI break with data corruption and
# segfaults as the documented consequences, not a cosmetic banner.
#
# So do not trust find_package here: ask the preprocessor what it resolves,
# using the same -isystem flags setup_target() applies, and refuse to configure
# when it disagrees with the library. Runs for the library build and, through
# Config.cmake.in, for every downstream application too.
option(DiFfRG_CHECK_HDF5_HEADERS
       "Verify at configure time that the resolved hdf5.h matches the linked HDF5"
       ON)

if(HDF5
   AND DiFfRG_CHECK_HDF5_HEADERS
   AND NOT "${HDF5_VERSION}" STREQUAL "")
  set(_h5_probe_flags "")
  foreach(_d "${BUNDLED_DIR}/include" ${HDF5_INCLUDE_DIRS}
              ${DiFfRG_HDF5_INCLUDE_DIRS})
    if(IS_DIRECTORY "${_d}")
      list(APPEND _h5_probe_flags "-isystem" "${_d}")
    endif()
  endforeach()

  set(_h5_probe_src "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/diffrg_hdf5_probe.cc")
  file(WRITE "${_h5_probe_src}" "#include <hdf5.h>\n")
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" ${_h5_probe_flags} -E -dM "${_h5_probe_src}"
    OUTPUT_VARIABLE _h5_macros
    ERROR_VARIABLE _h5_probe_err
    RESULT_VARIABLE _h5_probe_rc
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT _h5_probe_rc EQUAL 0)
    # Fail open: an unusual toolchain must not block the build, and a genuinely
    # broken <hdf5.h> will produce a far clearer error at compile time.
    message(
      STATUS
        "HDF5 header check skipped: could not preprocess <hdf5.h> "
        "(exit ${_h5_probe_rc})")
  else()
    set(_h5_hdr_version "")
    if(_h5_macros MATCHES "#define[ \t]+H5_VERS_MAJOR[ \t]+([0-9]+)")
      set(_h5_major "${CMAKE_MATCH_1}")
      string(REGEX MATCH "#define[ \t]+H5_VERS_MINOR[ \t]+([0-9]+)" _h5_ignored
                   "${_h5_macros}")
      set(_h5_minor "${CMAKE_MATCH_1}")
      string(REGEX MATCH "#define[ \t]+H5_VERS_RELEASE[ \t]+([0-9]+)" _h5_ignored
                   "${_h5_macros}")
      set(_h5_hdr_version "${_h5_major}.${_h5_minor}.${CMAKE_MATCH_1}")
    endif()

    if("${_h5_hdr_version}" STREQUAL "")
      message(STATUS "HDF5 header check skipped: <hdf5.h> declared no version")
    elseif(_h5_hdr_version VERSION_EQUAL "${HDF5_VERSION}")
      message(STATUS "HDF5 headers resolve to ${_h5_hdr_version} (matches the library)")
    else()
      # Name the culprit rather than making the reader hunt for it.
      set(_h5_culprits "")
      foreach(_var CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH)
        if(NOT "$ENV{${_var}}" STREQUAL "")
          string(REPLACE ":" ";" _dirs "$ENV{${_var}}")
          foreach(_dir IN LISTS _dirs)
            if(NOT "${_dir}" STREQUAL "" AND EXISTS "${_dir}/hdf5.h")
              list(APPEND _h5_culprits "    ${_dir}/hdf5.h  (via ${_var})")
            endif()
          endforeach()
        endif()
      endforeach()
      if(_h5_culprits)
        list(REMOVE_DUPLICATES _h5_culprits)
        string(REPLACE ";" "\n" _h5_culprits "${_h5_culprits}")
        set(_h5_culprits
            "  Reached before any -isystem directory:\n\n${_h5_culprits}\n\n")
      endif()

      message(
        FATAL_ERROR
          "\n"
          "======================================================================\n"
          "  HDF5 headers and library disagree\n"
          "======================================================================\n"
          "  headers the compiler resolves : ${_h5_hdr_version}\n"
          "  library that will be linked   : ${HDF5_VERSION}\n"
          "                                  ${HDF5_INCLUDE_DIRS}\n"
          "\n"
          "${_h5_culprits}"
          "  This would not fail at compile or link time. It produces a binary\n"
          "  that prints 'HDF5 library version mismatched error' on every run,\n"
          "  with data corruption and segfaults as the documented consequences.\n"
          "\n"
          "  Remove the shadowing HDF5 from the environment, e.g.\n"
          "      module unload lib/hdf5        # or: conda deactivate\n"
          "  then reconfigure in a clean build directory. To use that HDF5 on\n"
          "  purpose instead, point DiFfRG at it:\n"
          "      -DHDF5_DIR=<prefix containing hdf5-config.cmake>\n"
          "\n"
          "  Override with -DDiFfRG_CHECK_HDF5_HEADERS=OFF (not recommended).\n"
          "======================================================================\n")
    endif()
  endif()
endif()

if(${DiFfRG_MPI})
  find_package(MPI REQUIRED)
endif()

# DiFfRG's HAVE_MPI and deal.II's DEAL_II_WITH_MPI must agree: deal.II is what declares MPI_Comm
# and friends (real types with MPI, dummies in namespace dealii without), so a mixed configuration
# compiles the two halves of DiFfRG::MPI against different type sets. common/mpi.hh turns this into
# a #error as well, but catching it at configure time is much easier to act on.
if(DEFINED DEAL_II_WITH_MPI)
  if(${DiFfRG_MPI} AND NOT DEAL_II_WITH_MPI)
    message(FATAL_ERROR
      "DiFfRG is configured with MPI=ON but the deal.II found at ${DEAL_II_PATH} was built without "
      "DEAL_II_WITH_MPI. MPI is a single switch for the whole superbuild -- reconfigure the "
      "superbuild with -DMPI=ON so deal.II and SUNDIALS are rebuilt too, or set -DMPI=OFF.")
  endif()
  if(NOT ${DiFfRG_MPI} AND DEAL_II_WITH_MPI)
    message(WARNING
      "deal.II was built with MPI but DiFfRG is configured with MPI=OFF. This builds and runs, but "
      "every DiFfRG::MPI call degrades to a single-rank no-op, so a multi-rank launch would give "
      "each rank the full problem rather than a share of it.")
  endif()
endif()

# ##############################################################################
# Dependency summary
# ##############################################################################

message("")
message(
  "${BoldWhite}======================================================================${ColourReset}"
)
message("${BoldWhite}  DiFfRG Dependency Summary${ColourReset}")
message(
  "${BoldWhite}======================================================================${ColourReset}"
)

# Resolve a path string for Boost: prefer Boost_DIR (CMake config package),
# otherwise fall back to the first include directory.
if(DEFINED Boost_DIR AND NOT "${Boost_DIR}" STREQUAL "Boost_DIR-NOTFOUND")
  set(_boost_path "${Boost_DIR}")
else()
  list(GET Boost_INCLUDE_DIRS 0 _boost_path)
endif()

# Resolve a path string for HDF5: prefer HDF5_DIR if it points at a CMake
# config package, otherwise fall back to the first include directory.
if(DEFINED HDF5_DIR AND NOT "${HDF5_DIR}" STREQUAL "HDF5_DIR-NOTFOUND")
  set(_hdf5_path "${HDF5_DIR}")
else()
  list(GET HDF5_INCLUDE_DIRS 0 _hdf5_path)
endif()

# Pretty-print one summary row with fixed-width name and version columns so
# the (path) field aligns identically across every dependency. ARGN is an
# optional trailing annotation (e.g. "[static]").
function(_diffrg_summary_row name version path)
  set(_name_width 10)
  set(_version_width 14)

  string(LENGTH "${name}" _nlen)
  string(LENGTH "${version}" _vlen)
  math(EXPR _npad "${_name_width} - ${_nlen}")
  math(EXPR _vpad "${_version_width} - ${_vlen}")
  if(_npad LESS 1)
    set(_npad 1)
  endif()
  if(_vpad LESS 1)
    set(_vpad 1)
  endif()
  string(REPEAT " " ${_npad} _nspaces)
  string(REPEAT " " ${_vpad} _vspaces)

  set(_suffix "")
  if(ARGN)
    set(_suffix " ${ARGN}")
  endif()

  message(
    "  ${BoldGreen}${name}${ColourReset}${_nspaces}${version}${_vspaces}(${path})${_suffix}"
  )
endfunction()

_diffrg_summary_row("deal.II"  "${deal.II_VERSION}"  "${deal.II_DIR}")
_diffrg_summary_row("TBB"      "${TBB_VERSION}"      "${TBB_DIR}")
_diffrg_summary_row("Kokkos"   "${Kokkos_VERSION}"   "${Kokkos_DIR}")
_diffrg_summary_row("Boost"    "${Boost_VERSION}"    "${_boost_path}")
_diffrg_summary_row("Eigen3"   "${Eigen3_VERSION}"   "${Eigen3_DIR}")
_diffrg_summary_row("GSL"      "${GSL_VERSION}"      "${GSL_INCLUDE_DIRS}")
_diffrg_summary_row("autodiff" "${autodiff_VERSION}" "${autodiff_DIR}")
_diffrg_summary_row("spdlog"   "${spdlog_VERSION}"   "${spdlog_DIR}")
_diffrg_summary_row("HDF5"     "${HDF5_VERSION}"     "${_hdf5_path}" "[static]")
if(${DiFfRG_MPI})
  _diffrg_summary_row("MPI"    "${MPI_CXX_VERSION}"  "${MPI_CXX_INCLUDE_DIRS}")
endif()
if(NOT DiFfRG_CUDA_ARCH_BUNDLED STREQUAL "")
  _diffrg_summary_row("CUDA arch" "${DiFfRG_CUDA_ARCH_DISPLAY}"
                      "bundled Kokkos: sm_${DiFfRG_CUDA_ARCH_BUNDLED}")
endif()
message(
  "${BoldWhite}======================================================================${ColourReset}"
)
message("")

# ##############################################################################
# Convenience functions
# ##############################################################################

# We redefine the deal_ii_setup_target function here such that we can choose
# precisely how to propagate flags and other details
function(setup_dealii TARGET)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(_build "DEBUG")
  else()
    set(_build "RELEASE")
  endif()

  # deal.II >= 9.7 renamed its imported targets; the pre-9.7 names deal_II /
  # deal_II.g no longer exist. Link the config-aware umbrella target
  # dealii::dealii, which (unlike DEAL_II_INCLUDE_DIRS) also propagates the
  # include dirs of optional features such as UMFPACK/suitesparse.
  target_link_libraries(${TARGET} PUBLIC dealii::dealii)

  target_include_directories(${TARGET} SYSTEM PUBLIC ${DEAL_II_INCLUDE_DIRS})

  set(_cflags "${DEAL_II_CXX_FLAGS} ${DEAL_II_CXX_FLAGS_${_build}}")
  # Remove the c++20 and the optimization flag: CMake adds the standard itself
  # and CMAKE_CXX_FLAGS_<CONFIG> is the single source of truth for the -O level
  # (the same reason dealii::dealii's INTERFACE_COMPILE_OPTIONS is sanitized
  # after find_package(deal.II) above). Keeping either here would put two -O
  # flags on the compile line, which makes nvcc_wrapper warn and silently keep
  # the last one. The leading space lets the regex match a flag in first
  # position too.
  string(REPLACE "-std=c++20" "" _cflags ${_cflags})
  string(REGEX REPLACE " -O[0-9a-zA-Z]+" "" _cflags " ${_cflags}")
  separate_arguments(_cflags)
  # deal.II built through nvcc/nvcc_wrapper (the GCC + CUDA path) emits repeated
  # "-Xcudafe <code>" pairs. Added as plain compile options these get
  # de-duplicated by CMake, which collapses the identical -Xcudafe tokens and
  # orphans the trailing --diag_suppress=NNN, so the host compiler receives them
  # raw and errors out. Fuse each "-Xcudafe <arg>" into a single SHELL: fragment
  # so the pair stays intact and is exempt from de-duplication. No-op on the
  # clang-CUDA path, which emits no -Xcudafe.
  set(_cxx_opts "")
  set(_pending_xcudafe FALSE)
  foreach(_tok IN LISTS _cflags)
    if(_tok STREQUAL "")
      continue()
    elseif(_pending_xcudafe)
      list(APPEND _cxx_opts "SHELL:-Xcudafe ${_tok}")
      set(_pending_xcudafe FALSE)
    elseif(_tok STREQUAL "-Xcudafe")
      set(_pending_xcudafe TRUE)
    else()
      list(APPEND _cxx_opts "${_tok}")
    endif()
  endforeach()
  target_compile_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${_cxx_opts}>)

  set(_lflags "${DEAL_II_LINKER_FLAGS} ${DEAL_II_LINKER_FLAGS_${_build}}")
  separate_arguments(_lflags)
  target_link_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${_lflags}>)
endfunction()

function(setup_target TARGET)
  setup_dealii(${TARGET})

  # Check if the target is DiFfRG
  if(${TARGET} STREQUAL "DiFfRG")
    target_include_directories(${TARGET} PRIVATE ${autodiff_SOURCE_DIR})
  endif()

  # Do not warn about missing braces
  target_compile_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:
                                          -Wno-missing-braces>)

  target_include_directories(${TARGET} SYSTEM PUBLIC ${BUNDLED_DIR}/include)
  target_link_libraries(${TARGET} PUBLIC autodiff::autodiff)
  target_link_libraries(${TARGET} PUBLIC GSL::gsl)
  target_link_libraries(${TARGET} PUBLIC ${DiFfRG_EIGEN_TARGET})
  target_link_libraries(${TARGET} PUBLIC spdlog::spdlog)
  target_link_libraries(${TARGET} PUBLIC ${Boost_LIBRARIES})
  target_link_libraries(${TARGET} PUBLIC TBB::tbb)
  # deal.II's exported target already carries the concrete Kokkos archives.
  # Keep Kokkos discovered above, but do not add the same archives a second time.
  target_link_libraries(${TARGET} PUBLIC ${DiFfRG_HDF5_LIBRARIES})
  if(DiFfRG_HDF5_INCLUDE_DIRS)
    target_include_directories(${TARGET} SYSTEM PUBLIC ${DiFfRG_HDF5_INCLUDE_DIRS})
  endif()
  # target_link_libraries(${TARGET} PUBLIC petsc)

  if(${DiFfRG_MPI})
    target_link_libraries(${TARGET} PUBLIC MPI::MPI_CXX)
    target_compile_definitions(${TARGET} PUBLIC HAVE_MPI)
  endif()

  if(NOT ${CMAKE_BUILD_TYPE} STREQUAL Debug)
    # -march follows the resolved MARCH/NATIVE value (see the option block at the
    # top of this file); the fast-math flags are CPU-portable and always applied
    # in non-Debug builds.
    if(DiFfRG_MARCH_VALUE STREQUAL "none")
      set(_arch_flag)
    else()
      set(_arch_flag -march=${DiFfRG_MARCH_VALUE})
    endif()
    target_compile_options(
      ${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${_arch_flag} -ffast-math
                       -ffp-contract=fast -fno-finite-math-only >)
    target_compile_options(${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
                                            --use_fast_math>)
  endif()

  target_compile_definitions(${TARGET} PUBLIC _HAS_AUTO_PTR_ETC=0)

  if(NOT DiFfRG_CUDA_ARCH_BUNDLED STREQUAL "")
    target_compile_definitions(
      ${TARGET}
      PUBLIC DiFfRG_KERNEL_ARCH="${DiFfRG_CUDA_ARCH_DISPLAY}"
             DiFfRG_BUNDLE_ARCH="sm_${DiFfRG_CUDA_ARCH_BUNDLED}")
  endif()

  # Workaround: spdlog's bundled fmt uses consteval for format-string checking,
  # which breaks on newer compilers. constexpr is functionally equivalent.
  target_compile_definitions(${TARGET} PUBLIC FMT_CONSTEVAL=constexpr)
  # Force-include the workaround prelude (a missing <cassert> in deal.II's
  # tensor.h, and an nvcc defect around std::iterator_traits<char *>; see the
  # header for the details). It has to be a single -include: CMake de-duplicates
  # the repeated "-include" token of a second one and orphans its file name,
  # which the compiler then treats as an additional input file. The header
  # resolves through the DiFfRG include directory, which precedes the flags on
  # the command line in both the build and the install interface.
  target_compile_options(
    ${TARGET} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-include DiFfRG/common/prelude.hh>)

  if(HDF5)
    target_compile_definitions(${TARGET} PUBLIC H5CPP)
  endif()

  if(DiFfRG_LINKER_SUPPORTS_NO_WARN_DUPLICATE_LIBRARIES)
    target_link_options(${TARGET} PUBLIC
                        "LINKER:-no_warn_duplicate_libraries")
  endif()
endfunction()
