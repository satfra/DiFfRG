#!/bin/bash

LIBRARY_NAME="boost"
SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

source $SCRIPT_PATH/build_scripts/parse_flags.sh
source $SCRIPT_PATH/build_scripts/populate_paths.sh
source $SCRIPT_PATH/build_scripts/cleanup_build_if_asked.sh
source $SCRIPT_PATH/build_scripts/setup_folders.sh
source $SCRIPT_PATH/../config

cd $BUILD_PATH

COMPILER_CXX=$(cmake -E environment | grep "CXX" | awk -F'=' '{print $3}')

cd $SOURCE_PATH

$SuperUser ./bootstrap.sh --prefix=${INSTALL_PATH} &>/dev/null
$SuperUser ./b2 --build-dir=${BUILD_PATH} \
  --prefix=${INSTALL_PATH} \
  -j ${THREADS} \
  cxxflags="${CXX_FLAGS} -std=c++20" \
  install &> $MAKE_LOG_FILE || {
    cat $MAKE_LOG_FILE
    exit 1
  }
