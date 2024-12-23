#!/bin/bash

# ##############################################################################
# Utility
# ##############################################################################

expandPath() {
  local path
  local -a pathElements resultPathElements
  IFS=':' read -r -a pathElements <<<"$1"
  : "${pathElements[@]}"
  for path in "${pathElements[@]}"; do
    : "$path"
    case $path in
    "~+"/*)
      path=$PWD/${path#"~+/"}
      ;;
    "~-"/*)
      path=$OLDPWD/${path#"~-/"}
      ;;
    "~"/*)
      path=$HOME/${path#"~/"}
      ;;
    "~"*)
      username=${path%%/*}
      username=${username#"~"}
      IFS=: read -r _ _ _ _ _ homedir _ < <(getent passwd "$username")
      if [[ $path = */* ]]; then
        path=${homedir}/${path#*/}
      else
        path=$homedir
      fi
      ;;
    esac
    resultPathElements+=("$path")
  done
  local result
  printf -v result '%s:' "${resultPathElements[@]}"
  printf '%s\n' "${result%:}"
}

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
install_dir=''
cuda_flag=''
config_file='config'
config_flag=''
while getopts :i:j:fcd flag; do
  case "${flag}" in
  d)
    config_file="config_docker"
    config_flag="-d"
    ;;
  i) install_dir=${OPTARG} ;;
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
SOURCEPATH=${SCRIPTPATH}/DiFfRG
BUILDPATH=${SCRIPTPATH}/DiFfRG_build
LOGPATH=${SCRIPTPATH}/logs
mkdir -p ${LOGPATH}

# Obtain possibly user-defined configuration
source ${SCRIPTPATH}/${config_file}

# ##############################################################################
# Build bundled libraries
# ##############################################################################

git submodule update --init --recursive --jobs ${threads}

if [[ -z ${install_dir+x} ]]; then
  echo
  read -p "Install DiFfRG library globally to /opt/DiFfRG? [y/N/path] " install_dir
  install_dir=${install_dir:-N}
fi

if [[ ${install_dir} != "n" ]] && [[ ${install_dir} != "N" ]]; then
  cd ${SCRIPTPATH}

  # Make sure the install directory is absolute
  idir=$(expandPath ${install_dir}/)
  idir=$(readlink --canonicalize ${idir})
  echo "DiFfRG library will be installed in ${idir}"

  # Check if the install directory is writable
  failed_first=0
  mkdir -p ${idir} &>/dev/null && touch ${idir}/_permission_test &>/dev/null || {
    failed_first=1
  }

  # Install dependencies
  start=$(date +%s)
  cd ${SCRIPTPATH}/external
  if [[ $failed_first == 0 ]]; then
    rm -f ${idir}/_permission_test
    echo
    echo "Installing dependencies..."
    bash -i ./install.sh -i ${idir}/bundled -j ${threads} ${cuda_flag} # &>${LOGPATH}/DiFfRG_dependencies_install.log
  else
    echo "Elevated permissions required for install path ${idir}."
    sudo mkdir -p ${idir}
    echo
    echo "Installing dependencies..."
    sudo -E bash -i ./install.sh -i ${idir}/bundled -j ${threads} ${cuda_flag} # &>${LOGPATH}/DiFfRG_dependencies_install.log
  fi
  end=$(date +%s)
  runtime=$((end - start))
  elapsed="Elapsed: $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
  echo "    Done. (${elapsed})"
  echo
fi

# ##############################################################################
# Setup and build library
# ##############################################################################

if [[ "$full_inst" == "y" ]] && [[ "$install_dir" != "" ]]; then
  bash -i ${SCRIPTPATH}/update_DiFfRG.sh -j ${threads} -m ${cuda_flag} ${config_flag} -i ${install_dir}
else
  if [[ "$full_inst" == "y" ]]; then
    bash -i ${SCRIPTPATH}/update_DiFfRG.sh -j ${threads} -m ${cuda_flag} ${config_flag}
  else
    bash -i ${SCRIPTPATH}/update_DiFfRG.sh -j ${threads} ${config_flag}
  fi
fi

# ##############################################################################
# Finish
# ##############################################################################

echo "DiFfRG setup complete."
echo
