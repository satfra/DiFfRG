# ##############################################################################
# Tests for basic integrators
# ##############################################################################

setup_test(quadrature_integrator_fT.cc)
setup_test(map_points_fT.cc)

# ##############################################################################
# Tests for derived integrators
# ##############################################################################

setup_test(integrator_fT.cc)
setup_test(integrator_fT_p2.cc)
setup_test(integrator_fT_p2_1ang.cc)
setup_test(integrator_fT_p2_4d_2ang.cc)

setup_test(matsubara_stress.cc)
setup_test(matsubara_split.cc)
setup_test(matsubara_continuity.cc)

# Cost table, not a pass/fail test: run the binary by hand before and after a change to the
# frequency-rule selection and diff the two.
setup_benchmark(matsubara_cost.cc)
