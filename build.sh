#!/bin/bash

# ##############################################################################
# Script setup
# ##############################################################################

usage_msg="Build script for setting up the DiFfRG library and its dependencies. 
For configuration of build flags, please edit the config file.

Usage: build.sh [options]
Options:
  -f               Perform a full build and install of everything without confirmations.
  -c               Use CUDA when building the DiFfRG library.
  -i <directory>   Set the installation directory for the library.
  -j <threads>     Set the number of threads passed to make and git fetch.
  --help           Display this information.
"

threads='1'
full_inst='n'
install_dir=''
cuda=''
while getopts :i:j:fc flag; do
  case "${flag}" in
  i) install_dir=${OPTARG} ;;
  j) threads=${OPTARG} ;;
  f) full_inst='y' ;;
  c) cuda="-c" ;;
  ?)
    printf "${usage_msg}"
    exit 2
    ;;
  esac
done

# Get the path where this script is located
scriptpath="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

# Obtain possibly user-defined configuration
source config

# ##############################################################################
# Build bundled libraries
# ##############################################################################

git submodule update --init --recursive --jobs ${threads}
echo
echo "Building bundled libraries..."
cd ${scriptpath}/external
start=$(date +%s)
echo "    Building QMC..."
bash -i ./build_qmc.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build qmc, aborting."
  exit 1
}
echo "    Building autodiff..."
bash -i ./build_autodiff.sh -j ${threads} || {
  echo "    Failed to build autodiff, aborting."
  exit 1
}
echo "    Building Boost..."
bash -i ./build_boost.sh -j ${threads} &>/dev/null || {
  echo "    Failed to build Boost, aborting."
  exit 1
}
echo "    Building Catch2..."
bash -i ./build_Catch2.sh -j ${threads} &>/dev/null || { echo "    Failed to build Catch2, tests will not work. Continuing setup process."; }
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
echo "    Building rmm..."
bash -i ./build_rmm.sh -j ${threads} &>/dev/null || { echo "    Failed to build rmm, CUDA will not work. Continuing setup process."; }
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
end=$(date +%s)
runtime=$((end - start))
elapsed="Elapsed: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "    Done. (${elapsed})"
cd ${scriptpath}
echo

# ##############################################################################
# Setup and build library
# ##############################################################################

if [[ "$full_inst" == "y" ]] && [[ "$install_dir" != "" ]]; then
  bash -i ${scriptpath}/update_DiFfRG.sh -j ${threads} -ml ${cuda} -i ${install_dir}
else
  if [[ "$full_inst" == "y" ]]; then
    bash -i ${scriptpath}/update_DiFfRG.sh -j ${threads} -ml ${cuda}
  else
    bash -i ${scriptpath}/update_DiFfRG.sh -j ${threads}
  fi
fi

# ##############################################################################
# Finish
# ##############################################################################

echo "DiFfRG setup complete."
echo
