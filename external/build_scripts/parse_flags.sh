THREADS='1'
CLEANUP_FLAG=false
TOP_INSTALL_PATH=${SCRIPT_PATH}
TOP_BUILD_PATH=${SCRIPT_PATH}

while getopts "cj:i:b:" flag; do
  case "${flag}" in
  j) THREADS=${OPTARG} ;;
  c) CLEANUP_FLAG=true ;;
  i) TOP_INSTALL_PATH=${OPTARG} ;;
  b) TOP_BUILD_PATH=${OPTARG} ;;
  esac
done
