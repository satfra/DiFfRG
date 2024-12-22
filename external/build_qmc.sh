#!/bin/bash

LIBRARY_NAME="qmc"
SCRIPT_PATH="$(
  cd -- "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"

source $SCRIPT_PATH/build_scripts/populate_paths.sh
source $SCRIPT_PATH/build_scripts/parse_flags.sh
source $SCRIPT_PATH/build_scripts/cleanup_build_if_asked.sh
source $SCRIPT_PATH/build_scripts/setup_folders.sh

cd $SOURCE_PATH

python3 ./make_singleheader.py
mv qmc.hpp ${INSTALL_PATH}/qmc.hpp
