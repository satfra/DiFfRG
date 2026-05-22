#!/bin/bash

read -p "Delete full build tree of DiFfRG library? [y/N] " option_clear
option_clear=${option_clear:-N}

if [[ ${option_clear} = "y" ]] || [[ ${option_clear} = "Y" ]]; then
  echo "Deleting..."
  SCRIPTPATH="$(
    cd -- "$(dirname "$0")" >/dev/null 2>&1
    pwd -P
  )"
  # The build tree holds the ExternalProject sources/builds of all bundled
  # dependencies as well as the DiFfRG library build. Removing it forces a full
  # rebuild on the next invocation of build.sh.
  rm -rf "${SCRIPTPATH}/DiFfRG_build" "${SCRIPTPATH}/DiFfRG_install"
  rm -rf "${SCRIPTPATH}/logs"
fi

echo "    Done"
