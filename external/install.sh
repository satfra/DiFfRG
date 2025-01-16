#!/bin/bash

# ##############################################################################
# Script setup
# ##############################################################################

threads='1'
while getopts i:j:c flag; do
  case "${flag}" in
  i) install_dir=${OPTARG} ;;
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
./build_boost.sh -j ${threads} -i ${install_dir} || {
  echo "    Failed to build Boost, aborting."
  exit 1
}

echo "    Building oneTBB..."
./build_oneTBB.sh -j ${threads} -i ${install_dir} || {
  echo "    Failed to build oneTBB, aborting."
  exit 1
}

echo "    Building kokkos..."
./build_kokkos.sh -j ${threads} -i ${install_dir} || {
  echo "    Failed to build kokkos, aborting."
  exit 1
}

echo "    Building sundials..."
./build_sundials.sh -j ${threads} -i ${install_dir} || {
  echo "    Failed to build SUNDIALS, aborting."
  exit 1
}

echo "    Building deal.II..."
./build_dealii.sh -j ${threads} -i ${install_dir} &>/dev/null || {
  echo "    Failed to build deal.ii, aborting."
  exit 1
}
