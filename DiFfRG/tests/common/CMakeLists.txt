# ##############################################################################
# Test
# ##############################################################################

add_subdirectory(quadrature)

add_executable(test_configuration_helper test_configuration_helper.cc)
setup_test(test_configuration_helper)

add_executable(test_cpu_complex_autodiff test_cpu_complex_autodiff.cc)
setup_test(test_cpu_complex_autodiff)

if(USE_CUDA)
  add_executable(test_gpu_autodiff test_gpu_autodiff.cu)
  setup_test(test_gpu_autodiff)

  add_executable(test_gpu_complex test_gpu_complex.cu)
  setup_test(test_gpu_complex)

  add_executable(test_gpu_complex_autodiff test_gpu_complex_autodiff.cu)
  setup_test(test_gpu_complex_autodiff)
endif()
