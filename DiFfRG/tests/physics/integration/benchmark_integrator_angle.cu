#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/integrator_angle_cpu.hh>
#include <DiFfRG/physics/integration/integrator_angle_gpu.hh>
#include <DiFfRG/physics/integration/quadrature_provider.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto kernel(const double q, const double cos, const double /*k*/,
                                                         const double /*c*/, const double x0, const double x1,
                                                         const double x2, const double x3, const double x4,
                                                         const double x5, const double cos_x0, const double cos_x1,
                                                         const double cos_x2, const double cos_x3)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q) + x4 * powr<4>(q) + x5 * powr<5>(q)) *
           (cos_x0 + cos_x1 * powr<1>(cos) + cos_x2 * powr<2>(cos) + cos_x3 * powr<3>(cos));
  }

  static __forceinline__ __host__ __device__ auto
  constant(const double /*k*/, const double c, const double /*x0*/, const double /*x1*/, const double /*x2*/,
           const double /*x3*/, const double /*x4*/, const double /*x5*/, const double /*cos_x0*/,
           const double /*cos_x1*/, const double /*cos_x2*/, const double /*cos_x3*/)
  {
    return c;
  }
};

TEMPLATE_TEST_CASE_SIG("Benchmark cpu momentum integrals", "[integration][quadrature integration]", ((int dim), dim),
                       (2), (3), (4))
{
  const double x_extent = GENERATE(take(1, random(1., 2.)));
  QuadratureProvider quadrature_provider;

  constexpr uint take_n = 1;
  const auto poly = Polynomial({
      dim == 1 ? 0. : GENERATE(take(take_n, random(-1., 1.))), // x0
      GENERATE(take(take_n, random(-1., 1.))),                 // x1
      GENERATE(take(take_n, random(-1., 1.))),                 // x2
      GENERATE(take(1, random(-1., 1.))),                      // x3
      GENERATE(take(1, random(-1., 1.))),                      // x4
      GENERATE(take(1, random(-1., 1.))),                      // x5
  });
  const auto cos_poly = Polynomial({
      GENERATE(take(take_n, random(-1., 1.))), // x0
      GENERATE(take(take_n, random(-1., 1.))), // x1
      GENERATE(take(1, random(-1., 1.))),      // x2
      GENERATE(take(1, random(-1., 1.)))       // x3
  });

  const double k = GENERATE(take(take_n, random(0., 1.)));
  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  const double constant = GENERATE(take(take_n, random(-1., 1.)));

  {
    IntegratorAngleGPU<dim, double, PolyIntegrand> integrator(quadrature_provider, {{64, 16}}, x_extent);
    BENCHMARK_ADVANCED("GPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator
            .request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                     cos_poly[2], cos_poly[3])
            .get();
      });
    };
    BENCHMARK_ADVANCED("get GPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                       cos_poly[2], cos_poly[3]);
      });
    };
    BENCHMARK_ADVANCED("GPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        std::vector<std::future<double>> futures;
        for (int i = 0; i < 128; ++i)
          futures.emplace_back(
              std::move(integrator.request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5],
                                           cos_poly[0], cos_poly[1], cos_poly[2], cos_poly[3])));
        for (auto &f : futures)
          f.get();
      });
    };
    BENCHMARK_ADVANCED("get GPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        for (int i = 0; i < 128; ++i)
          integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                         cos_poly[2], cos_poly[3]);
      });
    };
  }
  {
    IntegratorAngleTBB<dim, double, PolyIntegrand> integrator(quadrature_provider, {{64, 16}}, x_extent);
    BENCHMARK_ADVANCED("CPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator
            .request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                     cos_poly[2], cos_poly[3])
            .get();
      });
    };
    BENCHMARK_ADVANCED("get CPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                       cos_poly[2], cos_poly[3]);
      });
    };
    BENCHMARK_ADVANCED("CPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        std::vector<std::future<double>> futures;
        for (int i = 0; i < 128; ++i)
          futures.emplace_back(
              std::move(integrator.request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5],
                                           cos_poly[0], cos_poly[1], cos_poly[2], cos_poly[3])));
        for (auto &f : futures)
          f.get();
      });
    };
    BENCHMARK_ADVANCED("get CPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        for (int i = 0; i < 128; ++i)
          integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                         cos_poly[2], cos_poly[3]);
      });
    };
  }
}