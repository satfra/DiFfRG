# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(solver)

# ##############################################################################
# Test timesteppers
# ##############################################################################

setup_test(sundials_ida.cc)
setup_mpi_test(sundials_ida_boost.cc)
setup_test(scaled_gmres.cc)
setup_test(implicit_euler.cc)
setup_test(explicit_euler.cc)
setup_test(trbdf2.cc)
setup_test(snapshot_restart.cc)
setup_test(local_tolerances.cc)
