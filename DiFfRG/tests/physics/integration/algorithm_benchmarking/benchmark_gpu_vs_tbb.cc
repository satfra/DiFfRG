#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration.hh>
#include <DiFfRG/physics/interpolation.hh>
#include <DiFfRG/physics/regulators.hh>

// #include "./ZA4/kernel.hh"
#include "./flows/ZA4/kernel.hh"

using namespace DiFfRG;

TEST_CASE("Benchmark ZA4 GPU vs TBB", "[integration][benchmark]")
{
  DiFfRG::Init();

  using Regulator = DiFfRG::PolynomialExpRegulator<>;
  using Kernel = ZA4_kernel<Regulator>;
  const LogCoordinates coordinates(64, 1e-3, 40., 10.);
  QuadratureProvider quadrature_provider;
  const std::array<size_t, 4> grid_size = {32, 8, 8, 8};
  const double k = 10.;
  std::vector<double> dummy_data(coordinates.size(), 1.0);

  // ONE set of interpolators, driven through both backends. They are valid on whichever side the
  // kernel happens to execute on, so nothing here is per-backend any more -- which is the point of
  // the space-agnostic design.
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> ZA3(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> ZAcbc(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> ZA4(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> dtZc(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> Zc(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> dtZA(coordinates);
  SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> ZA(coordinates);

  ZA3.update(dummy_data.data());
  ZAcbc.update(dummy_data.data());
  ZA4.update(dummy_data.data());
  dtZc.update(dummy_data.data());
  Zc.update(dummy_data.data());
  dtZA.update(dummy_data.data());
  ZA.update(dummy_data.data());

  BENCHMARK_ADVANCED("GPU")(Catch::Benchmark::Chronometer meter)
  {
    Integrator_p2_4D_3ang<4, double, Kernel, GPU_exec> integrator(quadrature_provider, grid_size);
    std::vector<double> result(coordinates.size());

    meter.measure([&] { integrator.map(result.data(), coordinates, k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA).fence(); });

    REQUIRE(is_close(result[0] - 2.9958e-05, 0., 1e-9));
  };

  BENCHMARK_ADVANCED("TBB")(Catch::Benchmark::Chronometer meter)
  {
    Integrator_p2_4D_3ang<4, double, Kernel, TBB_exec> integrator(quadrature_provider, grid_size);
    std::vector<double> result(coordinates.size());

    meter.measure([&] { integrator.map(result.data(), coordinates, k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA).fence(); });

    REQUIRE(is_close(result[0] - 2.9958e-05, 0., 1e-9));
  };
}
