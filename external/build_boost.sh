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
if [ -z "$COMPILER_CXX" ]; then
  COMPILER_CXX=$(which g++)
  echo "No compiler for boost build specified, choosing ${COMPILER_CXX}"
fi

COMPILER=$(cmake -E environment | grep "CC" | awk -F'=' '{print $2}')
if [ -z "$COMPILER" ]; then
  COMPILER=$(which gcc)
  echo "No compiler for boost build specified, choosing ${COMPILER}"
fi

cd $SOURCE_PATH

$SuperUser ./bootstrap.sh --prefix=${INSTALL_PATH} &>/dev/null
$SuperUser ./b2 --build-dir=${BUILD_PATH} \
  --prefix=${INSTALL_PATH} \
  --with-headers \
  --with-iostreams \
  --with-json \
  --with-math \
  --with-serialization \
  --with-system \
  --with-thread \
  -j ${THREADS} \
  cxxflags="${CXX_FLAGS} -std=c++20" \
  install &> $MAKE_LOG_FILE
