#!/bin/bash

# ##############################################################################
# Script setup
# ##############################################################################

threads='1'
while getopts j: flag; do
  case "${flag}" in
  j) threads=${OPTARG} ;;
  esac
done
scriptpath="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

if [[ "$OSTYPE" =~ ^darwin ]]; then
  export OpenMP_ROOT=$(brew --prefix)/opt/libomp
fi

# ##############################################################################
# Build application
# ##############################################################################

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-O3 -march=native -ffast-math -fno-finite-math-only -Wl,--demangle' -DCMAKE_CUDA_FLAGS="-arch=native" ..
make -j"$threads"
