# ##############################################################################
# Tests on CPU
# ##############################################################################

add_executable(test_tex_linear_interpolation_1D_cpu
               tex_linear_interpolation_1D_cpu.cc)
setup_test(test_tex_linear_interpolation_1D_cpu)

add_executable(test_tex_linear_interpolation_3D_cpu
               tex_linear_interpolation_3D_cpu.cc)
setup_test(test_tex_linear_interpolation_3D_cpu)

add_executable(test_tex_linear_interpolation_1D_stack_cpu
               tex_linear_interpolation_1D_stack_cpu.cc)
setup_test(test_tex_linear_interpolation_1D_stack_cpu)

add_executable(test_linear_interpolation_1D_cpu linear_interpolation_1D_cpu.cc)
setup_test(test_linear_interpolation_1D_cpu)

add_executable(test_linear_interpolation_2D_cpu linear_interpolation_2D_cpu.cc)
setup_test(test_linear_interpolation_2D_cpu)

add_executable(test_linear_interpolation_3D_cpu linear_interpolation_3D_cpu.cc)
setup_test(test_linear_interpolation_3D_cpu)

# ##############################################################################
# Tests on GPU
# ##############################################################################

if(USE_CUDA)
  add_executable(test_tex_linear_interpolation_1D_gpu
                 tex_linear_interpolation_1D_gpu.cu)
  setup_test(test_tex_linear_interpolation_1D_gpu)

  add_executable(test_tex_linear_interpolation_3D_gpu
                 tex_linear_interpolation_3D_gpu.cu)
  setup_test(test_tex_linear_interpolation_3D_gpu)

  add_executable(test_tex_linear_interpolation_1D_stack_gpu
                 tex_linear_interpolation_1D_stack_gpu.cu)
  setup_test(test_tex_linear_interpolation_1D_stack_gpu)

  add_executable(test_linear_interpolation_1D_gpu
                 linear_interpolation_1D_gpu.cu)
  setup_test(test_linear_interpolation_1D_gpu)

  add_executable(test_linear_interpolation_2D_gpu
                 linear_interpolation_2D_gpu.cu)
  setup_test(test_linear_interpolation_2D_gpu)

  add_executable(test_linear_interpolation_3D_gpu
                 linear_interpolation_3D_gpu.cu)
  setup_test(test_linear_interpolation_3D_gpu)
endif()

# ##############################################################################
# Benchmarks
# ##############################################################################

if(USE_CUDA)
  add_executable(benchmark_tex_linear_interpolation_1D
                 benchmark_tex_linear_interpolation_1D.cu)
  setup_benchmark(benchmark_tex_linear_interpolation_1D)

  add_executable(benchmark_tex_linear_interpolation_3D
                 benchmark_tex_linear_interpolation_3D.cu)
  setup_benchmark(benchmark_tex_linear_interpolation_3D)
endif()
