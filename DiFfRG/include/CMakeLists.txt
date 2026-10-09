install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/include" # source directory
  DESTINATION "${CMAKE_INSTALL_PREFIX}" # target directory
  FILES_MATCHING # install only matched files
  PATTERN "*.hh" # select header files
)

install(
  DIRECTORY "${CMAKE_SOURCE_DIR}/include" # source directory
  DESTINATION "${CMAKE_INSTALL_PREFIX}" # target directory
  FILES_MATCHING # install only matched files
  PATTERN "*.hpp" # select header files
)

# Ship the licenses of the vendored header-only libraries next to their headers.
install(FILES "${CMAKE_SOURCE_DIR}/include/toml11/LICENSE"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/toml11")