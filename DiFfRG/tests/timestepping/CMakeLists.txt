# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(solver)

# ##############################################################################
# Test timesteppers
# ##############################################################################

setup_test(sundials_ida.cc)
setup_test(ida_warm_restart.cc)
setup_test(ida_jacobian_reuse.cc)
setup_mpi_test(sundials_ida_boost.cc)
setup_test(scaled_gmres.cc)
setup_test(implicit_euler.cc)
setup_test(explicit_euler.cc)
setup_test(abm_error_control.cc)
setup_test(trbdf2.cc)
setup_test(snapshot_restart.cc)
setup_test(local_tolerances.cc)
setup_test(variables_steppers.cc)
setup_test(unsupported_features.cc)
setup_test(output_grid.cc)
setup_test(hybrid_adaptation.cc)
