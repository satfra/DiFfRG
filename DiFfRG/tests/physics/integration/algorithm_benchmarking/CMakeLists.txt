setup_benchmark(bench.cc)
target_sources(
  bench
  PRIVATE method1.cc
          method2.cc
          method3.cc
          method4.cc
          method5.cc
          method6.cc
          method7.cc
          method8.cc)
target_compile_options(bench PRIVATE -Wno-unused-parameter -Wno-unused-variable)

setup_benchmark(benchmark_gpu_vs_tbb.cc)
target_compile_options(benchmark_gpu_vs_tbb PRIVATE -Wno-unused-parameter -Wno-unused-variable)
