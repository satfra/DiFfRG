# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(vacuum)
add_subdirectory(finiteT)
add_subdirectory(lattice)

add_subdirectory(algorithm_benchmarking)

# ##############################################################################
# Tests for basic integrators
# ##############################################################################

setup_test(optimize.cc)
setup_test(quadrature_integrator.cc)
setup_test(map_points.cc)
setup_test(map_scheduler.cc)
setup_mpi_test(distributed_quadrature_integrator.cc)

# ##############################################################################
# Benchmarks
# ##############################################################################

setup_benchmark(benchmark_integrator_1d.cc)
