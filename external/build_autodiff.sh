#!/bin/bash

LIBRARY_NAME="autodiff"

source ./build_scripts/populate_paths.sh
source ./build_scripts/parse_flags.sh
source ./build_scripts/cleanup_build_if_asked.sh
source ./build_scripts/setup_folders.sh

cd $BUILD_PATH

cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS=${CUDA_FLAGS} \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    -DCMAKE_CXX_FLAGS="${CXX_FLAGS} -O3 -DNDEBUG" \
    -DCMAKE_EXE_LINKER_FLAGS="${EXE_LINKER_FLAGS}" \
    -DAUTODIFF_BUILD_TESTS=OFF \
    -DAUTODIFF_BUILD_PYTHON=OFF \
    -DAUTODIFF_BUILD_EXAMPLES=OFF \
    -DAUTODIFF_BUILD_DOCS=OFF \
    -DEigen3_DIR=${SCRIPT_PATH}/eigen_install \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
    -S ${SOURCE_PATH} \
    2>&1 | tee $CMAKE_LOG_FILE

make -j $THREADS 2>&1 | tee $MAKE_LOG_FILE
make -j $THREADS install
