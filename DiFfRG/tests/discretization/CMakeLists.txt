# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(DG)
add_subdirectory(CG)
add_subdirectory(grid)

# ##############################################################################
# Test mesh
# ##############################################################################

add_executable(test_mesh test_mesh.cc)
setup_test(test_mesh)

# ##############################################################################
# Test mesh adaptor
# ##############################################################################

add_executable(test_mesh_adaptor test_mesh_adaptor.cc)
setup_test(test_mesh_adaptor)

# ##############################################################################
# Test the EoM finding
# ##############################################################################

add_executable(test_EoM_finding test_EoM_finding.cc)
setup_test(test_EoM_finding)
