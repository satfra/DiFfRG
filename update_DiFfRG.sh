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

usage_msg="Build script for building and installing the DiFfRG library. 
For configuration of build flags, please edit the config file.

Usage: update_DiFfRG.sh [options]
Options:
  -c               Use CUDA when building the DiFfRG library.
  -i <directory>   Set the installation directory for the library.
  -j <threads>     Set the number of threads passed to make and git fetch.
  -m               Install the Mathematica package locally.
  --help           Display this information.
"

# ##############################################################################
# Script setup
# ##############################################################################

THREADS='1'
CUDA_OPT="-DUSE_CUDA=OFF"
cuda_flag=""
config_file="config"
config_flag=''
while getopts :j:mcdi: flag; do
  case "${flag}" in
  d)
    config_file="config_docker"
    config_flag="-d"
    ;;
  j)
    THREADS=${OPTARG}
    ;;
  m)
    option_setup_library=${option_setup_library:-n}
    option_install_library=${option_install_library:-n}
    option_setup_mathematica="Y"
    ;;
  i)
    option_setup_library="Y"
    option_install_library=${OPTARG}
    option_setup_mathematica=${option_setup_mathematica:-n}
    ;;
  c)
    CUDA_OPT="-DUSE_CUDA=ON"
    cuda_flag="-c"
    ;;
  ?)
    echo "${usage_msg}"
    exit 2
    ;;
  esac
done
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

echo

# ##############################################################################
# Setup and build library
# ##############################################################################

if [[ -z ${option_install_library+x} ]]; then
  echo
  read -p "Install DiFfRG library globally to /opt/DiFfRG? [y/N/path] " option_install_library
  option_install_library=${option_install_library:-N}
fi

if [[ -z ${option_setup_library+x} ]]; then
  read -p "Build DiFfRG library? [Y/n] " option_setup_library
  option_setup_library=${option_setup_library:-Y}
fi

if [[ ${option_install_library} != "n" ]] && [[ ${option_install_library} != "N" ]]; then
  cd ${SCRIPTPATH}

  # Make sure the install directory is absolute
  idir=$(expandPath ${option_install_library}/)
  idir=$(readlink --canonicalize ${idir})
  echo "DiFfRG library will be installed in ${idir}"

  # Check if the install directory is writable
  mkdir -p ${idir} &>/dev/null && touch ${idir}/_permission_test &>/dev/null || {
    failed_first=1
  }

  if [[ ${CUDA_OPT} == "-DUSE_CUDA=ON" ]]; then
    echo "Using CUDA to build DiFfRG library."
  else
    echo "Not using CUDA to build DiFfRG library. To switch it on, use the -c flag!"
  fi

  echo "Running CMake..."
  mkdir -p ${BUILDPATH}
  cd $BUILDPATH
  cmake \
    -DCMAKE_INSTALL_PREFIX=${idir} \
    -DBUNDLED_DIR=${idir}/bundled \
    ${CUDA_OPT} \
    -DCMAKE_CUDA_FLAGS=${CUDA_FLAGS} \
    -DCMAKE_CXX_FLAGS=${CXXFLAGS} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DDiFfRG_BUILD_TESTS=OFF \
    -DDiFfRG_BUILD_DOCUMENTATION=ON \
    -S ${SOURCEPATH} &>${LOGPATH}/DiFfRG_cmake.log || {
    echo "    Failed to configure DiFfRG, aborting."
    exit 1
  }
  make release_additional_flags &>/dev/null

  echo "Updating DiFfRG..."
  make -j ${THREADS} &>${LOGPATH}/DiFfRG_make.log || {
    echo "    Failed to build DiFfRG, aborting."
    exit 1
  }

  echo "Updating documentation..."
  make -j ${THREADS} documentation &>${LOGPATH}/DiFfRG_documentation.log || { echo "    Failed to build DiFfRG documentation."; }

  echo "Installing..."
  if [[ -w ${idir} ]]; then
    make install -j ${THREADS} &>${LOGPATH}/DiFfRG_install.log
    cp -r ${SCRIPTPATH}/python ${idir}/
  else
    if ((failed_first == 0)); then
      echo "Elevated permissions required to write into ${idir}."
    fi
    sudo -E make install -j ${THREADS} &>${LOGPATH}/DiFfRG_install.log
    sudo cp -r ${SCRIPTPATH}/python ${idir}/
  fi

  echo "Done."
  echo

  cd ${SCRIPTPATH}
fi

# ##############################################################################
# Setup mathematica environment
# ##############################################################################

if [[ -z ${option_setup_mathematica+x} ]]; then
  echo
  read -p "Install DiFfRG mathematica package locally? [Y/n] " option_setup_mathematica
  option_setup_mathematica=${option_setup_mathematica:-Y}
fi
if [[ ${option_setup_mathematica} != "n" ]] && [[ ${option_setup_mathematica} != "N" ]]; then
  echo "    Checking for mathematica installation..."
  if command -v math &>/dev/null; then

    # We use the math command to determine the mathematica applications folder
    math_app_folder=$(math -run 'FileNameJoin[{$UserBaseDirectory,"Applications"}]//Print;Exit[]' | tail -n1)
    mkdir -p ${math_app_folder}

    # Check if a DiFfRG mathematica package is already installed
    if [[ -d ${math_app_folder}/DiFfRG ]]; then
      echo "    DiFfRG mathematica package already installed in ${math_app_folder}"
      read -p "        Do you want to overwrite it? [y/N] " option_overwrite_mathematica
      option_overwrite_mathematica=${option_overwrite_mathematica:-N}
      if [[ ${option_overwrite_mathematica} == "n" ]] || [[ ${option_overwrite_mathematica} == "N" ]]; then
        echo "    Aborting."
        exit 0
      fi
      rm -rf ${math_app_folder}/DiFfRG
    fi

    # Copy the DiFfRG mathematica package to the mathematica applications folder
    echo "    Installing DiFfRG mathematica package to ${math_app_folder}"
    cp -r ${SCRIPTPATH}/Mathematica/DiFfRG ${math_app_folder} || {
      echo "    Failed to install DiFfRG mathematica package, aborting."
      exit 1
    }
  else
    echo "Mathematica: 'math' command could not be found"
    exit 1
  fi
fi

echo
