#!/bin/bash

# ##############################################################################
# Script setup
# ##############################################################################

threads='1'
while getopts j:c flag; do
  case "${flag}" in
  j) threads=${OPTARG} ;;
  c) cuda="-c" ;;
  ?)
    exit 2
    ;;
  esac
done

source ../config

################################################################################
# This script builds all the dependencies for the project.
################################################################################

echo "    Building Boost..."
bash -i ./build_boost.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build Boost, aborting."
  exit 1
}

echo "    Building oneTBB..."
bash -i ./build_oneTBB.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build oneTBB, aborting."
  exit 1
}

echo "    Building kokkos..."
bash -i ./build_kokkos.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build kokkos, aborting."
  exit 1
}

echo "    Building sundials..."
bash -i ./build_sundials.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build SUNDIALS, aborting."
  exit 1
}

echo "    Building deal.II..."
bash -i ./build_dealii.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build deal.ii, aborting."
  exit 1
}
