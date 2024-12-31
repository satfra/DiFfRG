# ##############################################################################
# Setup additional include directories and Catch2
# ##############################################################################

include_directories(./)

# Find Catch2 Testing Framework
cpmaddpackage(
  NAME
  Catch2
  GITHUB_REPOSITORY
  catchorg/Catch2
  VERSION
  3.7.1
  OPTIONS
  "CATCH_INSTALL_DOCS OFF"
  "CATCH_INSTALL_HELPERS OFF")

function(setup_test TARGET)
  setup_target(${TARGET})
  target_link_libraries(${TARGET} DiFfRG::DiFfRG Catch2::Catch2WithMain)
  catch_discover_tests(${TARGET})
endfunction()

function(setup_benchmark TARGET)
  setup_target(${TARGET})
  target_link_libraries(${TARGET} DiFfRG::DiFfRG Catch2::Catch2WithMain)
endfunction()

list(APPEND CMAKE_MODULE_PATH ${lib_Catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(common)
add_subdirectory(discretization)
add_subdirectory(model)
add_subdirectory(physics)
add_subdirectory(timestepping)
