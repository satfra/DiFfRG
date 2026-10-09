# ##############################################################################
# Tests
# ##############################################################################

setup_test(KurganovTadmor_tests.cc)
setup_test(assembler_jacobian_fd_tests.cc)
setup_test(flux_jacobian_hessian_tests.cc)
setup_test(flowing_variables_tests.cc)
setup_test(limiter_tests.cc)
setup_test(reconstructor_tests.cc)
setup_test(wave_speed_tests.cc)
setup_test(kt_batch_equivalence.cc)

add_subdirectory(regression)
