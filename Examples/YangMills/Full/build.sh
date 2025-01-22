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
cmake ..
make -j"$threads"
