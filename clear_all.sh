#!/bin/bash

read -p "Delete full build tree of DiFfRG library? [y/N] " option_clear
option_clear=${option_clear:-N}

if [[ ${option_clear} = "y" ]] || [[ ${option_clear} = "Y" ]]; then
  echo "Deleting..."
  cd external
  ./build_boost.sh -c
  ./build_oneTBB.sh -c
  ./build_kokkos.sh -c
  ./build_sundials.sh -c
  ./build_dealii.sh -c
  cd ..
  rm -rf ./DiFfRG_build ./DiFfRG_install
  rm -rf ./logs
  rm -f ./external/*.log
fi

echo "    Done"
