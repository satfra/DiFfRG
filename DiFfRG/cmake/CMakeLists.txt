configure_file(setup_build_system.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/setup_build_system.cmake COPYONLY)
configure_file(cuda_arch.cmake ${CMAKE_CURRENT_BINARY_DIR}/cuda_arch.cmake
               COPYONLY)
configure_file(verify_install.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/verify_install.cmake COPYONLY)
# cuda_arch.cmake is include()d by setup_build_system.cmake from its own
# directory, so it has to travel with it into the install tree.
install(FILES setup_build_system.cmake cuda_arch.cmake verify_install.cmake
        DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake")
