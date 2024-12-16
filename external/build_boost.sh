#!/bin/bash

LIBRARY_NAME="boost"

source ./build_scripts/populate_paths.sh
source ./build_scripts/parse_flags.sh
source ./build_scripts/cleanup_build_if_asked.sh
source ./build_scripts/setup_folders.sh

cd $BUILD_PATH

COMPILER_CXX=`cmake -E environment | grep "CXX" | awk -F'=' '{print $3}'`
if [ -z "$COMPILER_CXX" ]; then
    COMPILER_CXX=`which g++`
    echo "No compiler for boost build specified, choosing ${COMPILER_CXX}";
fi

COMPILER=`cmake -E environment | grep "CC" | awk -F'=' '{print $2}'`
if [ -z "$COMPILER" ]; then
    COMPILER=`which gcc`
    echo "No compiler for boost build specified, choosing ${COMPILER}";
fi

echo $COMPILER
cd $SOURCE_PATH

./bootstrap.sh --prefix=${INSTALL_PATH}
./b2 --build-dir=${BUILD_PATH} \
     --prefix=${INSTALL_PATH} \
     --with-json \
     --with-math \
     --with-serialization \
     --with-system \
     --with-headers \
     -j ${THREADS} \
     cxxflags="-std=c++20" \
     install | tee $MAKE_LOG_FILE