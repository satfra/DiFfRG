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
  -t <directory>   Use the TBB installation at this prefix instead of building one.
  -s <directory>   Use the SUNDIALS installation at this prefix instead of building one.
                   By default a compatible system Boost (>= 1.80), TBB (>= 2021) and
                   SUNDIALS (>= 5.4.0) are used if found, otherwise they are built from
                   source. Set BUILD_BOOST=1 / BUILD_TBB=1 / BUILD_SUNDIALS=1 to always
                   build the bundled one.
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

threads=''
INSTALLPATH=''
USE_CUDA_OPT='OFF'
config_file='config'
config_flag=''
full_build='false'
boost_dir=''
tbb_dir=''
sundials_dir=''
while getopts :i:j:b:t:s:fcd flag; do
  case "${flag}" in
  d)
    config_file="config_docker"
    config_flag="-d"
    ;;
  i) INSTALLPATH=${OPTARG} ;;
  j) threads=${OPTARG} ;;
  b) boost_dir=${OPTARG} ;;
  t) tbb_dir=${OPTARG} ;;
  s) sundials_dir=${OPTARG} ;;
  c) USE_CUDA_OPT='ON' ;;
  f) full_build='true' ;;
  ?)
    printf "%s" "${usage_msg}"
    exit 2
    ;;
  esac
done

# Auto-detect the number of build threads if -j was not given: use half the
# available cores (rounded down, at least 1) to limit peak RAM usage.
if [[ -z ${threads} ]]; then
  ncores=''
  if command -v nproc >/dev/null 2>&1; then
    ncores=$(nproc)
  elif command -v sysctl >/dev/null 2>&1; then
    ncores=$(sysctl -n hw.ncpu 2>/dev/null)
  fi
  ncores=${ncores:-2}
  threads=$((ncores / 2))
  [[ ${threads} -lt 1 ]] && threads=1
  echo "No -j given; using ${threads} build threads (half of ${ncores} cores)."
fi

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

# Dependency source selection forwarded to CMake. For each of Boost/TBB/SUNDIALS:
#   -b/-t/-s <dir>            -> use that prefix
#   BUILD_BOOST/TBB/SUNDIALS=1 -> always build the bundled one
# Default: use a compatible system copy if present, otherwise build it.
dep_cmake_args=()
if [[ -n ${boost_dir} ]]; then
  dep_cmake_args+=("-DBOOST_DIR=$(expandPath "${boost_dir}")")
  echo "  Using Boost from ${boost_dir}"
fi
if [[ -n ${tbb_dir} ]]; then
  dep_cmake_args+=("-DTBB_DIR=$(expandPath "${tbb_dir}")")
  echo "  Using TBB from ${tbb_dir}"
fi
if [[ -n ${sundials_dir} ]]; then
  dep_cmake_args+=("-DSUNDIALS_DIR=$(expandPath "${sundials_dir}")")
  echo "  Using SUNDIALS from ${sundials_dir}"
fi
for _dep in BOOST TBB SUNDIALS; do
  _v="BUILD_${_dep}"
  if [[ -n ${!_v} ]] && [[ ${!_v} != "0" ]]; then
    dep_cmake_args+=("-DBUILD_${_dep}=ON")
    echo "  Forcing a bundled ${_dep} build (BUILD_${_dep}=${!_v})."
  fi
done

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
  "${dep_cmake_args[@]}" \
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
