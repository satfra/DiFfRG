configure_file(setup_build_system.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/setup_build_system.cmake COPYONLY)
install(FILES setup_build_system.cmake CPM.cmake
        DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake")
