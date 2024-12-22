if [ "$CLEANUP_FLAG" = true ] ; then
    echo "Cleaning up ${LIBRARY_NAME}:"
    if [ -f "$CMAKE_LOG_FILE" ]; then
        echo "  Deleting ${CMAKE_LOG_FILE}"
        rm -rf $CMAKE_LOG_FILE
    fi
    if [ -f "$MAKE_LOG_FILE" ]; then
        echo "  Deleting ${MAKE_LOG_FILE}"
        rm -rf $MAKE_LOG_FILE
    fi
    if [ -d "$BUILD_PATH" ]; then
        echo "  Deleting ${BUILD_PATH}"
        rm -rf $BUILD_PATH
    fi
    if [ -d "$INSTALL_PATH" ]; then
        echo "  Deleting ${INSTALL_PATH}"
        rm -rf $INSTALL_PATH
    fi

    git submodule update --init $SOURCE_PATH
    cd $SOURCE_PATH
    git reset --hard

    exit 0;
fi
