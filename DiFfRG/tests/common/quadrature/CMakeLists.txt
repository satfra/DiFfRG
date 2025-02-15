# ##############################################################################
# Tests
# ##############################################################################

add_executable(test_diagonalization test_diagonalization.cc)
setup_test(test_diagonalization)

add_executable(test_quadrature test_quadrature.cc)
setup_test(test_quadrature)

add_executable(test_matsubara test_matsubara.cc)
setup_test(test_matsubara)
