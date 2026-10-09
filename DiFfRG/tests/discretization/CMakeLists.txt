# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(FEM)
add_subdirectory(FV)
add_subdirectory(grid)
add_subdirectory(data)
add_subdirectory(coordinates)

# ##############################################################################
# Tests
# ##############################################################################

setup_test(mesh.cc)
setup_mpi_test(mesh_parallel.cc)
setup_mpi_test(assembly_parallel.cc)
setup_mpi_test(snapshot_io.cc)
setup_test(mesh_adaptor.cc)
setup_test(mesh_configuration_test.cpp)
setup_test(eom_finding.cc)
setup_test(extractor_point.cc)
setup_test(solution_sample.cc)
setup_test(affine_constraint_metadata.cc)
