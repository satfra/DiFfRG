#!/bin/bash

LIBRARY_NAME="dealii"
SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

source $SCRIPT_PATH/build_scripts/parse_flags.sh
source $SCRIPT_PATH/build_scripts/populate_paths.sh
source $SCRIPT_PATH/build_scripts/cleanup_build_if_asked.sh
source $SCRIPT_PATH/build_scripts/setup_folders.sh

cd $BUILD_PATH

cmake -DCMAKE_BUILD_TYPE=DebugRelease \
  -DDEAL_II_COMPONENT_EXAMPLES=OFF \
  -DDEAL_II_COMPONENT_DOCUMENTATION=OFF \
  -DDEAL_II_WITH_MPI=OFF \
  -DDEAL_II_WITH_UMFPACK=ON \
  -DDEAL_II_WITH_TASKFLOW=OFF \
  -DDEAL_II_WITH_VTK=OFF \
  -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
  -DCMAKE_EXE_LINKER_FLAGS="${EXE_LINKER_FLAGS}" \
  -DCMAKE_CXX_STANDARD=17 \
  -DKOKKOS_DIR=${TOP_INSTALL_PATH}/kokkos_install \
  -DTBB_DIR=${TOP_INSTALL_PATH}/oneTBB_install \
  -DSUNDIALS_DIR=${TOP_INSTALL_PATH}/sundials_install \
  -DBOOST_DIR=${TOP_INSTALL_PATH}/boost_install \
  ${DEAL_II_CMAKE} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
  -S ${SOURCE_PATH} \
  2>&1 | tee $CMAKE_LOG_FILE

make -j $THREADS 2>&1 | tee $MAKE_LOG_FILE
make -j $THREADS install
