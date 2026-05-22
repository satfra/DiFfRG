#!/bin/bash

usage_msg="Build script for setting up the DiFfRG library and its dependencies.
For configuration of build flags (compiler, CUDA architecture, ...), edit the config file.

Usage: build.sh [options]
Options:
  -f               Perform a full build and install of everything without confirmations.
  -c               Use CUDA when building the DiFfRG library.
  -i <directory>   Set the installation directory for the library.
  -j <threads>     Set the number of threads passed to the build.
  -b <directory>   Use the Boost installation at this prefix instead of building one.
                   By default a system Boost (>= 1.80) is used if found, otherwise
                   Boost is built from source. Set the environment variable
                   BUILD_BOOST=1 to always build the bundled Boost.
  --help           Display this information.
"

# ##############################################################################
# Script setup
# ##############################################################################

# Long-option help (getopts below only handles single-letter flags).
for arg in "$@"; do
  if [[ "${arg}" == "--help" ]]; then
    printf "%s" "${usage_msg}"
    exit 0
  fi
done

threads='1'
INSTALLPATH=''
USE_CUDA_OPT='OFF'
config_file='config'
config_flag=''
full_build='false'
boost_dir=''
while getopts :i:j:b:fcd flag; do
  case "${flag}" in
  d)
    config_file="config_docker"
    config_flag="-d"
    ;;
  i) INSTALLPATH=${OPTARG} ;;
  j) threads=${OPTARG} ;;
  b) boost_dir=${OPTARG} ;;
  c) USE_CUDA_OPT='ON' ;;
  f) full_build='true' ;;
  ?)
    printf "%s" "${usage_msg}"
    exit 2
    ;;
  esac
done

# Get the path where this script is located
SCRIPTPATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
LOGPATH=${SCRIPTPATH}/logs
BUILDPATH=${SCRIPTPATH}/DiFfRG_build
mkdir -p ${LOGPATH}

# Obtain possibly user-defined configuration (compiler/linker flags, CUDA arch,
# deal.II flags, ...). See the `config` file for documentation.
source ${SCRIPTPATH}/${config_file}
source ${SCRIPTPATH}/build_scripts/expand_path.sh

# ##############################################################################
# Determine installation directory
# ##############################################################################

if [[ -z ${INSTALLPATH} ]]; then
  if [[ ${full_build} == "true" ]]; then
    INSTALLPATH="${HOME}/.local/share/DiFfRG"
  else
    echo
    read -p "Install DiFfRG library to ${HOME}/.local/share/DiFfRG? [Y/n/path] " answer
    answer=${answer:-Y}
    case "${answer}" in
    y | Y) INSTALLPATH="${HOME}/.local/share/DiFfRG" ;;
    n | N)
      echo "Aborting."
      exit 1
      ;;
    *) INSTALLPATH="${answer}" ;;
    esac
  fi
fi

# Make sure the install directory is absolute
idir=$(expandPath "${INSTALLPATH}")
echo "DiFfRG library will be installed in ${idir}"

# Boost source selection forwarded to CMake:
#   -b <dir>      -> use that Boost prefix
#   BUILD_BOOST=1 -> always build the bundled Boost
# (default: use a system Boost >= 1.80 if present, otherwise build it).
boost_cmake_args=()
if [[ -n ${boost_dir} ]]; then
  boost_cmake_args+=("-DBOOST_DIR=$(expandPath "${boost_dir}")")
  echo "  Using Boost from ${boost_dir}"
fi
if [[ -n ${BUILD_BOOST} ]] && [[ ${BUILD_BOOST} != "0" ]]; then
  boost_cmake_args+=("-DBUILD_BOOST=ON")
  echo "  Forcing a bundled Boost build (BUILD_BOOST=${BUILD_BOOST})."
fi

if [[ ${USE_CUDA_OPT} == "ON" ]]; then
  echo "  Using CUDA to build the DiFfRG library."
else
  echo "  Not using CUDA to build the DiFfRG library. To switch it on, use the -c flag!"
fi
echo

# ##############################################################################
# Build dependencies and library via the top-level CMake orchestrator
# ##############################################################################

# Each dependency is built by an ExternalProject sub-build that bakes the
# install prefix into its own CMake cache. Reusing a build tree that was
# configured for a different install prefix would therefore install the
# dependencies to the OLD location. If the prefix changed, start fresh.
if [[ -f "${BUILDPATH}/CMakeCache.txt" ]]; then
  prev_prefix=$(grep -E "^CMAKE_INSTALL_PREFIX:" "${BUILDPATH}/CMakeCache.txt" | head -1 | cut -d= -f2)
  if [[ -n ${prev_prefix} ]] && [[ ${prev_prefix} != "${idir}" ]]; then
    echo "Install prefix changed (${prev_prefix} -> ${idir}); removing ${BUILDPATH} for a clean build."
    rm -rf "${BUILDPATH}"
  fi
fi

# Let nested `cmake --build` (dependency) steps pick up the requested thread
# count without an explicit -j on every command.
export CMAKE_BUILD_PARALLEL_LEVEL=${threads}

start=$(date +%s)
echo "Configuring DiFfRG and its dependencies..."
cmake -S "${SCRIPTPATH}" -B "${BUILDPATH}" \
  -DCMAKE_INSTALL_PREFIX="${idir}" \
  -DUSE_CUDA="${USE_CUDA_OPT}" \
  -DBUILD_THREADS="${threads}" \
  -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
  -DCMAKE_C_FLAGS="${C_FLAGS}" \
  -DCMAKE_CUDA_FLAGS="${CUDA_FLAGS}" \
  -DDIFFRG_EXE_LINKER_FLAGS="${EXE_LINKER_FLAGS}" \
  -DDEAL_II_CMAKE="${DEAL_II_CMAKE}" \
  "${boost_cmake_args[@]}" \
  2>&1 | tee ${LOGPATH}/DiFfRG_cmake.log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "    Failed to configure DiFfRG, see ${LOGPATH}/DiFfRG_cmake.log."
  exit 1
fi

echo "Building and installing DiFfRG and its dependencies (this can take 10-40 minutes)..."
cmake --build "${BUILDPATH}" -j "${threads}" 2>&1 | tee ${LOGPATH}/DiFfRG_build.log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "    Failed to build DiFfRG, see ${LOGPATH}/DiFfRG_build.log."
  exit 1
fi

end=$(date +%s)
runtime=$((end - start))
echo "    Done. (Elapsed: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec)"
echo

# ##############################################################################
# Register the Mathematica package locally
# ##############################################################################

# The library, Python and Mathematica packages are already installed into the
# prefix by the CMake install step. This additionally registers the Mathematica
# package into the user's Wolfram Applications directory so it can be loaded.
if [[ ${full_build} == "true" ]]; then
  bash "${SCRIPTPATH}/update_DiFfRG.sh" ${config_flag} -m </dev/null || true
else
  bash "${SCRIPTPATH}/update_DiFfRG.sh" ${config_flag} -m
fi
