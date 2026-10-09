# ##############################################################################
# Add subdirectories
# ##############################################################################

add_subdirectory(quadrature)

# ##############################################################################
# Test
# ##############################################################################

setup_test(csv.cc)
setup_test(config_tree.cc)
setup_test(configuration_helper.cc)
setup_test(init_threads.cc)
setup_test(autodiff.cc)
setup_test(complex_autodiff.cc)
setup_test(tbb.cc)
setup_test(root_finding.cc)
setup_test(stream_filter.cc)
