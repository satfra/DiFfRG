mkdir -p $SCRIPT_PATH/logs
mkdir -p $BUILD_PATH

source $SCRIPT_PATH/../build_scripts/setup_permissions.sh
SuperUser=$(get_execution_permissions $INSTALL_PATH)
$SuperUser mkdir -p $INSTALL_PATH
