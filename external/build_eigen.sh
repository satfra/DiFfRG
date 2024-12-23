#!/bin/bash

LIBRARY_NAME="eigen"
SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

source $SCRIPT_PATH/build_scripts/parse_flags.sh
source $SCRIPT_PATH/build_scripts/populate_paths.sh
source $SCRIPT_PATH/build_scripts/cleanup_build_if_asked.sh
source $SCRIPT_PATH/build_scripts/setup_folders.sh

cd $BUILD_PATH

cmake -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
      -S ${SOURCE_PATH} \
      2>&1 | tee $CMAKE_LOG_FILE

make -j $THREADS 2>&1 | tee $MAKE_LOG_FILE
make -j $THREADS install
