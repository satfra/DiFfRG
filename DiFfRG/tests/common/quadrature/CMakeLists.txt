# ##############################################################################
# Tests
# ##############################################################################

setup_test(diagonalization.cc)
setup_test(quadrature.cc)
setup_test(matsubara.cc)
setup_test(quadrature_provider.cc)

# Convergence study, not a pass/fail test: run the binary by hand to get the tables that say
# how many Matsubara nodes a given tail behaviour actually needs.
setup_benchmark(matsubara_convergence.cc)
