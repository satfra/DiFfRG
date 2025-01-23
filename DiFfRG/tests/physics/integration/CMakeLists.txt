# ##############################################################################
# Tests on CPU
# ##############################################################################

add_executable(test_integrator_constant integrator_constant.cc)
setup_test(test_integrator_constant)

add_executable(test_integrator_cpu integrator_cpu.cc)
setup_test(test_integrator_cpu)

add_executable(test_integrator_qmc integrator_qmc.cc)
setup_test(test_integrator_qmc)

add_executable(test_integrator_angle_cpu integrator_angle_cpu.cc)
setup_test(test_integrator_angle_cpu)

add_executable(test_integrator_angle_qmc integrator_angle_qmc.cc)
setup_test(test_integrator_angle_qmc)

add_executable(test_integrator_2Dpx0_cpu integrator_2Dpx0_cpu.cc)
setup_test(test_integrator_2Dpx0_cpu)

add_executable(test_integrator_2Dpq0_cpu integrator_2Dpq0_cpu.cc)
setup_test(test_integrator_2Dpq0_cpu)

add_executable(test_integrator_3D_cpu integrator_3D_cpu.cc)
setup_test(test_integrator_3D_cpu)

add_executable(test_integrator_3Dpx0_cpu integrator_3Dpx0_cpu.cc)
setup_test(test_integrator_3Dpx0_cpu)

add_executable(test_integrator_3Dpq0_cpu integrator_3Dpq0_cpu.cc)
setup_test(test_integrator_3Dpq0_cpu)

add_executable(test_integrator_4D_cpu integrator_4D_cpu.cc)
setup_test(test_integrator_4D_cpu)

add_executable(test_integrator_4D_2ang_cpu integrator_4D_2ang_cpu.cc)
setup_test(test_integrator_4D_2ang_cpu)

add_executable(test_integrator_4D_qmc integrator_4D_qmc.cc)
setup_test(test_integrator_4D_qmc)

add_executable(test_integrator_1D_cartesian_cpu integrator_1D_cartesian_cpu.cc)
setup_test(test_integrator_1D_cartesian_cpu)

add_executable(test_integrator_2D_cartesian_cpu integrator_2D_cartesian_cpu.cc)
setup_test(test_integrator_2D_cartesian_cpu)

add_executable(test_integrator_3D_cartesian_cpu integrator_3D_cartesian_cpu.cc)
setup_test(test_integrator_3D_cartesian_cpu)

# ##############################################################################
# Tests on GPU
# ##############################################################################

if(USE_CUDA)
  add_executable(test_integrator_gpu integrator_gpu.cu)
  setup_test(test_integrator_gpu)

  add_executable(test_integrator_qmc_gpu integrator_qmc.cu)
  setup_test(test_integrator_qmc_gpu)

  add_executable(test_integrator_angle_gpu integrator_angle_gpu.cu)
  setup_test(test_integrator_angle_gpu)

  add_executable(test_integrator_angle_qmc_gpu integrator_angle_qmc.cu)
  setup_test(test_integrator_angle_qmc_gpu)

  add_executable(test_integrator_2Dpx0_gpu integrator_2Dpx0_gpu.cu)
  setup_test(test_integrator_2Dpx0_gpu)

  add_executable(test_integrator_2Dpq0_gpu integrator_2Dpq0_gpu.cu)
  setup_test(test_integrator_2Dpq0_gpu)

  add_executable(test_integrator_3D_gpu integrator_3D_gpu.cu)
  setup_test(test_integrator_3D_gpu)

  add_executable(test_integrator_3Dpx0_gpu integrator_3Dpx0_gpu.cu)
  setup_test(test_integrator_3Dpx0_gpu)

  add_executable(test_integrator_3Dpq0_gpu integrator_3Dpq0_gpu.cu)
  setup_test(test_integrator_3Dpq0_gpu)

  add_executable(test_integrator_4D_gpu integrator_4D_gpu.cu)
  setup_test(test_integrator_4D_gpu)

  add_executable(test_integrator_4D_2ang_gpu integrator_4D_2ang_gpu.cu)
  setup_test(test_integrator_4D_2ang_gpu)

  add_executable(test_integrator_4D_qmc_gpu integrator_4D_qmc.cu)
  setup_test(test_integrator_4D_qmc_gpu)

  add_executable(test_integrator_1D_cartesian_gpu
                 integrator_1D_cartesian_gpu.cu)
  setup_test(test_integrator_1D_cartesian_gpu)

  add_executable(test_integrator_2D_cartesian_gpu
                 integrator_2D_cartesian_gpu.cu)
  setup_test(test_integrator_2D_cartesian_gpu)

  add_executable(test_integrator_3D_cartesian_gpu
                 integrator_3D_cartesian_gpu.cu)
  setup_test(test_integrator_3D_cartesian_gpu)
endif()

# ##############################################################################
# Benchmarks
# ##############################################################################

if(USE_CUDA)
  add_executable(benchmark_integrator benchmark_integrator.cu)
  setup_benchmark(benchmark_integrator)

  add_executable(benchmark_integrator_angle benchmark_integrator_angle.cu)
  setup_benchmark(benchmark_integrator_angle)

  add_executable(benchmark_integrator_3D benchmark_integrator_3D.cu)
  setup_benchmark(benchmark_integrator_3D)

  add_executable(benchmark_integrator_4D benchmark_integrator_4D.cu)
  setup_benchmark(benchmark_integrator_4D)
endif()
