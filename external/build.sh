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

echo "    Building QMC..."
bash -i ./build_qmc.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build qmc, aborting."
  exit 1
}

echo "    Building autodiff..."
bash -i ./build_autodiff.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build autodiff, aborting."
  exit 1
}

echo "    Building rapidcsv..."
bash -i ./build_rapidcsv.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build rapidcsv, aborting."
  exit 1
}

echo "    Building Boost..."
bash -i ./build_boost.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build Boost, aborting."
  exit 1
}

echo "    Building Catch2..."
bash -i ./build_Catch2.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build Catch2, tests will not work. Continuing setup process."
}

echo "    Building Eigen3..."
bash -i ./build_eigen.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build Eigen, aborting."
  exit 1
}

echo "    Building thread-pool..."
bash -i ./build_thread-pool.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build thread-pool, aborting."
  exit 1
}

echo "    Building spdlog..."
bash -i ./build_spdlog.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build spdlog, aborting."
  exit 1
}

if [[ "${cuda}" == "-c" ]]; then
  echo "    Building rmm..."
  bash -i ./build_rmm.sh -j ${threads} &>/dev/null || {
    echo "    Failed to build rmm, CUDA will not work. Continuing setup process."
  }
fi

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
