add_custom_target(
  debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug")
add_custom_target(
  release_additional_flags
  COMMAND
    ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-march=native
    -ffast-math -fno-finite-math-only -Wno-misleading-indentation'
    ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release")
add_custom_target(
  release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_CXX_FLAGS='-Wno-misleading-indentation' ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release")
add_custom_target(
  profile
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=RelWithDebInfo
          -DCMAKE_CXX_FLAGS='-Wno-misleading-indentation' ${CMAKE_SOURCE_DIR}
  COMMENT "Switch CMAKE_BUILD_TYPE to Release")
