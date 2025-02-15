#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/integrator_3D_cpu.hh>
#include <DiFfRG/physics/integration/integrator_3D_gpu.hh>
#include <DiFfRG/physics/integration/integrator_4D_2ang_gpu.hh>
#include <DiFfRG/physics/integration/integrator_4D_2ang_cpu.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/utils.hh>

using namespace DiFfRG;

#include "../boilerplate/2_angle/flows.hh"  

//--------------------------------------------
// Quadrature integration

TEST_CASE("Benchmark 4D momentum integrals with 2 angles", "[4D integration][quadrature integration]")
{
  JSONValue json = json::value({{"physical", {{"Lambda", 1.}}},
                                {"integration",
                                 {{"x_quadrature_order", 32},
                                  {"angle_quadrature_order", 8},
                                  {"x0_quadrature_order", 16},
                                  {"x0_summands", 8},
                                  {"q0_quadrature_order", 16},
                                  {"q0_summands", 8},
                                  {"x_extent_tolerance", 1e-3},
                                  {"x0_extent_tolerance", 1e-3},
                                  {"q0_extent_tolerance", 1e-3},
                                  {"jacobian_quadrature_factor", 0.5},

                                  {"cudathreadsperblock", 32},
                                  {"cudablocks", 64},

                                  {"rel_tol", 1e-3},
                                  {"abs_tol", 1e-12},
                                  {"max_eval", 10000},
                                  {"minn", 512},
                                  {"minm", 16}}},
                                {"output", {{"verbosity", 0}}}});

  const uint p_grid_size = 32;
  const double p_min = 1e-3;
  const double p_max = 40.0;
  const double p_bias = 6.0;

  using Coordinates1D = LogarithmicCoordinates1D<float>;
  Coordinates1D coordinates1D(p_grid_size, p_min, p_max, p_bias);
  auto grid1D = make_grid(coordinates1D);

  const double k = 20.0;
  const double m2A = 1.0;
  const double alphaA3 = 0.21;
  const double alphaAcbc = 0.21;
  const double alphaA4 = 0.21;
  TexLinearInterpolator1D<double, Coordinates1D> dtZc(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> dtZA(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> ZA(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> Zc(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> ZA4(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> ZAcbc(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> ZA3(coordinates1D);

  for (uint i = 0; i < p_grid_size; ++i) {
    // add some random noise to the data
    ZA3[i] = std::sqrt(4. * M_PI * alphaA3) + 0.01 * (rand() / (double)RAND_MAX);
    ZAcbc[i] = std::sqrt(4. * M_PI * alphaAcbc) + 0.01 * (rand() / (double)RAND_MAX);
    ZA4[i] = 4. * M_PI * alphaA4 + 0.01 * (rand() / (double)RAND_MAX);
    ZA[i] = (powr<2>(grid1D[i]) + m2A) / powr<2>(grid1D[i]) + 0.01 * (rand() / (double)RAND_MAX);
    Zc[i] = 1. + 0.01 * (rand() / (double)RAND_MAX);
    dtZc[i] = 0.01 * (rand() / (double)RAND_MAX);
    dtZA[i] = 0.01 * (rand() / (double)RAND_MAX);
  }

  ZA3.update();
  ZAcbc.update();
  ZA4.update();
  ZA.update();
  Zc.update();
  dtZc.update();
  dtZA.update();

  const auto arguments = std::tie(ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA, m2A);

  YangMillsFlowEquations<Integrator4D2AngTBB> flows_TBB(json);
  flows_TBB.set_k(k);
  YangMillsFlowEquations<Integrator4D2AngGPU> flows_GPU(json);
  flows_GPU.set_k(k);

  std::vector<double> dummy_data(p_grid_size, 0.);

  BENCHMARK_ADVANCED("GPU")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      auto futures_ZA3 = request_data<double>(flows_GPU.ZA3_integrator, grid1D, k, arguments);
      update_data(futures_ZA3, &(dummy_data[0]));
    });
  };

  BENCHMARK_ADVANCED("CPU")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      auto futures_ZA3 = request_data<double>(flows_TBB.ZA3_integrator, grid1D, k, arguments);
      update_data(futures_ZA3, &(dummy_data[0]));
    });
  };
}

TEST_CASE("Benchmark 3D momentum integrals with 2 angles", "[3D integration][quadrature integration]")
{
  JSONValue json = json::value({{"physical", {{"Lambda", 1.}}},
                                {"integration",
                                 {{"x_quadrature_order", 32},
                                  {"angle_quadrature_order", 8},
                                  {"x0_quadrature_order", 16},
                                  {"x0_summands", 8},
                                  {"q0_quadrature_order", 16},
                                  {"q0_summands", 8},
                                  {"x_extent_tolerance", 1e-3},
                                  {"x0_extent_tolerance", 1e-3},
                                  {"q0_extent_tolerance", 1e-3},
                                  {"jacobian_quadrature_factor", 0.5},

                                  {"cudathreadsperblock", 32},
                                  {"cudablocks", 64},

                                  {"rel_tol", 1e-3},
                                  {"abs_tol", 1e-12},
                                  {"max_eval", 10000},
                                  {"minn", 512},
                                  {"minm", 16}}},
                                {"output", {{"verbosity", 0}}}});

  const uint p_grid_size = 32;
  const double p_min = 1e-3;
  const double p_max = 40.0;
  const double p_bias = 6.0;

  using Coordinates1D = LogarithmicCoordinates1D<float>;
  Coordinates1D coordinates1D(p_grid_size, p_min, p_max, p_bias);
  auto grid1D = make_grid(coordinates1D);

  const double k = 20.0;
  const double m2A = 1.0;
  const double alphaA3 = 0.21;
  const double alphaAcbc = 0.21;
  const double alphaA4 = 0.21;
  TexLinearInterpolator1D<double, Coordinates1D> dtZc(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> dtZA(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> ZA(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> Zc(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> ZA4(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> ZAcbc(coordinates1D);
  TexLinearInterpolator1D<double, Coordinates1D> ZA3(coordinates1D);

  for (uint i = 0; i < p_grid_size; ++i) {
    // add some random noise to the data
    ZA3[i] = std::sqrt(4. * M_PI * alphaA3) + 0.01 * (rand() / (double)RAND_MAX);
    ZAcbc[i] = std::sqrt(4. * M_PI * alphaAcbc) + 0.01 * (rand() / (double)RAND_MAX);
    ZA4[i] = 4. * M_PI * alphaA4 + 0.01 * (rand() / (double)RAND_MAX);
    ZA[i] = (powr<2>(grid1D[i]) + m2A) / powr<2>(grid1D[i]) + 0.01 * (rand() / (double)RAND_MAX);
    Zc[i] = 1. + 0.01 * (rand() / (double)RAND_MAX);
    dtZc[i] = 0.01 * (rand() / (double)RAND_MAX);
    dtZA[i] = 0.01 * (rand() / (double)RAND_MAX);
  }

  ZA3.update();
  ZAcbc.update();
  ZA4.update();
  ZA.update();
  Zc.update();
  dtZc.update();
  dtZA.update();

  const auto arguments = std::tie(ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA, m2A);

  YangMillsFlowEquations<Integrator3DTBB> flows_TBB(json);
  flows_TBB.set_k(k);
  YangMillsFlowEquations<Integrator3DGPU> flows_GPU(json);
  flows_GPU.set_k(k);

  std::vector<double> dummy_data(p_grid_size, 0.);

  BENCHMARK_ADVANCED("GPU")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      auto futures_ZA3 = request_data<double>(flows_GPU.ZA3_integrator, grid1D, k, arguments);
      update_data(futures_ZA3, &(dummy_data[0]));
    });
  };

  BENCHMARK_ADVANCED("CPU")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      auto futures_ZA3 = request_data<double>(flows_TBB.ZA3_integrator, grid1D, k, arguments);
      update_data(futures_ZA3, &(dummy_data[0]));
    });
  };
}