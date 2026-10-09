#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>

using namespace DiFfRG;

void method1(Catch::Benchmark::Chronometer &);
void method2(Catch::Benchmark::Chronometer &);
void method3(Catch::Benchmark::Chronometer &);
void method4(Catch::Benchmark::Chronometer &);
void method5(Catch::Benchmark::Chronometer &);
void method6(Catch::Benchmark::Chronometer &);
void method7(Catch::Benchmark::Chronometer &);
void method8(Catch::Benchmark::Chronometer &);

TEST_CASE("Benchmark different ND quadrature integrals", "[integration][quadrature]")
{
  DiFfRG::Init();
  BENCHMARK_ADVANCED("method1 GPU")(Catch::Benchmark::Chronometer meter) { method1(meter); };
  BENCHMARK_ADVANCED("method2 GPU")(Catch::Benchmark::Chronometer meter) { method2(meter); };
  BENCHMARK_ADVANCED("method3 GPU")(Catch::Benchmark::Chronometer meter) { method3(meter); };
  BENCHMARK_ADVANCED("method4 GPU")(Catch::Benchmark::Chronometer meter) { method4(meter); };
  BENCHMARK_ADVANCED("method5 GPU")(Catch::Benchmark::Chronometer meter) { method5(meter); };
  BENCHMARK_ADVANCED("method6 GPU")(Catch::Benchmark::Chronometer meter) { method6(meter); };
  BENCHMARK_ADVANCED("method7 GPU")(Catch::Benchmark::Chronometer meter) { method7(meter); };
  BENCHMARK_ADVANCED("method8 GPU")(Catch::Benchmark::Chronometer meter) { method8(meter); };
}
