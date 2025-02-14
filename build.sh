#!/bin/bash

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

# ##############################################################################
# Script setup
# ##############################################################################

threads='1'
full_inst='n'
INSTALLPATH=''
cuda_flag=''
config_file='config'
config_flag=''
while getopts :i:j:fcd flag; do
  case "${flag}" in
  d)
    config_file="config_docker"
    config_flag="-d"
    ;;
  i) INSTALLPATH=${OPTARG} ;;
  j) threads=${OPTARG} ;;
  f) full_inst='y' ;;
  c) cuda_flag="-c" ;;
  ?)
    printf "${usage_msg}"
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
mkdir -p ${LOGPATH}

# Obtain possibly user-defined configuration
source ${SCRIPTPATH}/${config_file}

source ${SCRIPTPATH}/build_scripts/expand_path.sh

# ##############################################################################
# Build bundled libraries
# ##############################################################################

git submodule update --init --recursive --jobs ${threads}

if [[ -z ${INSTALLPATH} ]]; then
  echo
  read -p "Install DiFfRG library globally to /opt/DiFfRG? [y/N/path] " INSTALLPATH
  INSTALLPATH=${INSTALLPATH:-N}
fi

if [[ ${INSTALLPATH} == "y" ]]; then
  INSTALLPATH="/opt/DiFfRG"
fi

if [[ ${INSTALLPATH} != "n" ]] && [[ ${INSTALLPATH} != "N" ]]; then
  cd ${SCRIPTPATH}
  # Make sure the install directory is absolute
  idir=$(expandPath ${INSTALLPATH})
  #idir=$(readlink --canonicalize ${idir})
  echo "DiFfRG library will be installed in ${idir}"

  # Install dependencies
  start=$(date +%s)
  cd ${SCRIPTPATH}/external
  echo "Installing dependencies..."
  bash ./install.sh -i ${idir}/bundled -j ${threads} ${cuda_flag} 2>&1 | tee ${LOGPATH}/DiFfRG_dependencies_install.log
  end=$(date +%s)
  runtime=$((end - start))
  elapsed="Elapsed: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
  echo "    Done. (${elapsed})"
  echo
fi

# ##############################################################################
# Setup and build library
# ##############################################################################

if [[ "$full_inst" == "y" ]] && [[ "$INSTALLPATH" != "" ]]; then
  bash ${SCRIPTPATH}/update_DiFfRG.sh -j ${threads} -m ${cuda_flag} ${config_flag} -i ${INSTALLPATH}
else
  if [[ "$full_inst" == "y" ]]; then
    bash ${SCRIPTPATH}/update_DiFfRG.sh -j ${threads} -m ${cuda_flag} ${config_flag}
  else
    bash ${SCRIPTPATH}/update_DiFfRG.sh -j ${threads} ${config_flag}
  fi
fi

# ##############################################################################
# Finish
# ##############################################################################
