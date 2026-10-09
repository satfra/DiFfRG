# ##############################################################################
# Tests
# ##############################################################################

setup_test(cg.cc)
setup_test(cg_consistency.cc)
setup_test(dg.cc)
setup_test(dg_consistency.cc)
setup_test(dg_batch_equivalence.cc)
setup_test(ldg.cc)
setup_test(ldg_consistency.cc)
setup_test(ldg_batch_equivalence.cc)
setup_test(raw_potential_readouts.cc)

setup_benchmark(benchmark_cg.cc)
setup_benchmark(benchmark_dg.cc)
setup_benchmark(benchmark_ldg.cc)
