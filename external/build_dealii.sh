#!/bin/bash

LIBRARY_NAME="dealii"

source ./build_scripts/populate_paths.sh
source ./build_scripts/parse_flags.sh
source ./build_scripts/cleanup_build_if_asked.sh
source ./build_scripts/setup_folders.sh

cd $BUILD_PATH

cmake -DCMAKE_BUILD_TYPE=DebugRelease \
  -DDEAL_II_COMPONENT_EXAMPLES=OFF \
  -DDEAL_II_COMPONENT_DOCUMENTATION=OFF \
  -DDEAL_II_ALLOW_PLATFORM_INTROSPECTION=ON \
  -DDEAL_II_WITH_CUDA=OFF \
  -DDEAL_II_WITH_MPI=OFF \
  -DDEAL_II_WITH_UMFPACK=ON \
  -DDEAL_II_WITH_TASKFLOW=OFF \
  -DDEAL_II_WITH_VTK=OFF \
  -DDEAL_II_USE_LTO=ON \
  -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
  -DCMAKE_EXE_LINKER_FLAGS="${EXE_LINKER_FLAGS}" \
  -DCMAKE_CXX_STANDARD=17 \
  -DKOKKOS_DIR=${SCRIPT_PATH}/kokkos_install \
  -DSUNDIALS_DIR=${SCRIPT_PATH}/sundials_install \
  ${DEAL_II_CMAKE} \
  -DBOOST_DIR=${SCRIPT_PATH}/boost_install \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -S ${SOURCE_PATH} \
  2>&1 | tee $CMAKE_LOG_FILE

make -j $THREADS 2>&1 | tee $MAKE_LOG_FILE
make -j $THREADS install
