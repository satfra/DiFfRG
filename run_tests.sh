#!/bin/bash

# ##############################################################################
# Script setup
# ##############################################################################

threads=''
force='y'
while getopts j:f flag; do
    case "${flag}" in
    j) threads=${OPTARG} ;;
    f) force='y' ;;
    esac
done
scriptpath="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
)"

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

# The top-level build (build.sh) is a superbuild: DiFfRG_build is the orchestrator
# build directory, and the DiFfRG library itself is built by an ExternalProject in
# the sub-build directory below. Tests and the CTest registration live there.
buildpath="${scriptpath}/DiFfRG_build/DiFfRG/src/DiFfRG-build"
if [ ! -d "${buildpath}" ]; then
    echo "DiFfRG library build not found at ${buildpath}."
    echo "Please run build.sh first."
    exit 1
fi

if [[ "$OSTYPE" =~ ^darwin ]]; then
    export OpenMP_ROOT=$(brew --prefix)/opt/libomp
fi

mkdir -p "${scriptpath}/logs"

# ##############################################################################
# Build tests
# ##############################################################################

# Re-configure the DiFfRG library sub-build with tests enabled (it keeps the
# BUNDLED_DIR / compiler / flags the superbuild configured it with), then build.
cd "${buildpath}"
cmake -DDiFfRG_BUILD_TESTS=ON . || { echo "Failed to configure tests."; exit 1; }
cmake --build . -j${threads} || { echo "Failed to build tests."; exit 1; }

# ##############################################################################
# Run tests
# ##############################################################################

{ ctest | tee "${scriptpath}/logs/DiFfRG_tests.log"; } || { echo "Tests failed."; exit 1; }