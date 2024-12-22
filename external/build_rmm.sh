#!/bin/bash

LIBRARY_NAME="rmm"

source ./build_scripts/populate_paths.sh
source ./build_scripts/parse_flags.sh
source ./build_scripts/cleanup_build_if_asked.sh
source ./build_scripts/setup_folders.sh

cd $BUILD_PATH

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=OFF \
    -DCMAKE_CXX_FLAGS="${CXX_FLAGS} -O3 -DNDEBUG" \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
    -DCMAKE_EXE_LINKER_FLAGS="${EXE_LINKER_FLAGS}" \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    -S ${SOURCE_PATH} \
    2>&1 | tee $CMAKE_LOG_FILE

make -j $THREADS 2>&1 | tee $MAKE_LOG_FILE
make -j $THREADS install
