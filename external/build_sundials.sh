#!/bin/bash

LIBRARY_NAME="sundials"
SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

source $SCRIPT_PATH/build_scripts/parse_flags.sh
source $SCRIPT_PATH/build_scripts/populate_paths.sh
source $SCRIPT_PATH/build_scripts/cleanup_build_if_asked.sh
source $SCRIPT_PATH/build_scripts/setup_folders.sh

cd $BUILD_PATH

cmake -DENABLE_OPENMP=ON \
  -DENABLE_MPI=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE="${C_FLAGS} -O3 -DNDEBUG" \
  -DCMAKE_EXE_LINKER_FLAGS="${EXE_LINKER_FLAGS}" \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -S ${SOURCE_PATH} \
  2>&1 | tee $CMAKE_LOG_FILE

make -j $THREADS 2>&1 | tee $MAKE_LOG_FILE
$SuperUser make -j $THREADS install
