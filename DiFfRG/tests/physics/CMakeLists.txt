# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(integration)
add_subdirectory(integration_finiteT)
add_subdirectory(interpolation)

# ##############################################################################
# Test loop integrals
# ##############################################################################

add_executable(test_loop_integrals test_loop_integrals.cc)
setup_test(test_loop_integrals)
