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