#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/integration_finiteT/integrator_4D_finiteTx0_cpu.hh>
#include <DiFfRG/physics/integration_finiteT/integrator_4D_finiteTx0_gpu.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto
  kernel(const double q, const double cos, const double phi, const double q0, const double /*k*/, const double /*c*/,
         const double x0, const double x1, const double x2, const double x3, const double x4, const double x5,
         const double cos_x0, const double cos_x1, const double cos_x2, const double cos_x3, const double phi_x0,
         const double phi_x1, const double phi_x2, const double phi_x3, const double q0_x0, const double q0_x1,
         const double q0_x2, const double q0_x3)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q) + x4 * powr<4>(q) + x5 * powr<5>(q)) *
           (cos_x0 + cos_x1 * powr<1>(cos) + cos_x2 * powr<2>(cos) + cos_x3 * powr<3>(cos)) *
           (phi_x0 + phi_x1 * powr<1>(phi) + phi_x2 * powr<2>(phi) + phi_x3 * powr<3>(phi)) *
           (q0_x0 + q0_x1 * powr<1>(q0) + q0_x2 * powr<2>(q0) + q0_x3 * powr<3>(q0));
  }

  static __forceinline__ __host__ __device__ auto
  constant(const double /*k*/, const double c, const double /*x0*/, const double /*x1*/, const double /*x2*/,
           const double /*x3*/, const double /*x4*/, const double /*x5*/, const double /*cos_x0*/,
           const double /*cos_x1*/, const double /*cos_x2*/, const double /*cos_x3*/, const double /*phi_x0*/,
           const double /*phi_x1*/, const double /*phi_x2*/, const double /*phi_x3*/, const double /*q0_x0*/,
           const double /*q0_x1*/, const double /*q0_x2*/, const double /*q0_x3*/)
  {
    return c;
  }
};

TEST_CASE("Benchmark 4D momentum integrals", "[4D integration][quadrature integration]")
{
  constexpr int dim = 4;

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
      GENERATE(take(1, random(-1., 1.))),      // x1
      GENERATE(take(1, random(-1., 1.))),      // x2
      GENERATE(take(1, random(-1., 1.)))       // x3
  });
  const auto phi_poly = Polynomial({
      GENERATE(take(take_n, random(-1., 1.))), // x0
      GENERATE(take(1, random(-1., 1.))),      // x1
      GENERATE(take(1, random(-1., 1.))),      // x2
      GENERATE(take(1, random(-1., 1.)))       // x3
  });
  const auto x0_poly = Polynomial({
      GENERATE(take(take_n, random(-1., 1.))), // x0
      GENERATE(take(1, random(-1., 1.))),      // x1
      GENERATE(take(1, random(-1., 1.))),      // x2
      GENERATE(take(1, random(-1., 1.)))       // x3
  });

  const double k = GENERATE(take(take_n, random(0., 1.)));
  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  const double constant = GENERATE(take(take_n, random(-1., 1.)));
  const double x0_summands = 8;
  const double T = GENERATE(take(take_n, random(0.5, 1.)));
  const double x0_extent = x0_summands * 10 * 2. * M_PI * T / k * GENERATE(take(1, random(1., 2.)));

  auto int_poly = poly;
  std::vector<double> coeff_integrand(dim, 0.);
  coeff_integrand[dim - 1] = 1.;
  int_poly *= Polynomial(coeff_integrand);
  {
    Integrator4DFiniteTx0GPU<double, PolyIntegrand> integrator(quadrature_provider, {{32, 8, 8, 8}}, x_extent,
                                                               x0_extent, x0_summands, T);
    BENCHMARK_ADVANCED("GPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        const double integral =
            integrator
                .request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                         cos_poly[2], cos_poly[3], phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3], x0_poly[0],
                         x0_poly[1], x0_poly[2], x0_poly[3])
                .get();
      });
    };
    BENCHMARK_ADVANCED("GPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        std::vector<std::future<double>> futures;
        for (int i = 0; i < 128; ++i)
          futures.emplace_back(
              std::move(integrator.request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5],
                                           cos_poly[0], cos_poly[1], cos_poly[2], cos_poly[3], phi_poly[0], phi_poly[1],
                                           phi_poly[2], phi_poly[3], x0_poly[0], x0_poly[1], x0_poly[2], x0_poly[3])));
        for (auto &f : futures)
          f.get();
      });
    };
    BENCHMARK_ADVANCED("get GPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        for (int i = 0; i < 128; ++i)
          integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                         cos_poly[2], cos_poly[3], phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3], x0_poly[0],
                         x0_poly[1], x0_poly[2], x0_poly[3]);
      });
    };
  }
  {
    Integrator4DFiniteTx0TBB<double, PolyIntegrand> integrator(quadrature_provider, {{32, 8, 8, 8}}, x_extent,
                                                               x0_extent, x0_summands, T);
    BENCHMARK_ADVANCED("CPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        const double integral =
            integrator
                .request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                         cos_poly[2], cos_poly[3], phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3], x0_poly[0],
                         x0_poly[1], x0_poly[2], x0_poly[3])
                .get();
      });
    };
    BENCHMARK_ADVANCED("get CPU")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        const double integral =
            integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                           cos_poly[2], cos_poly[3], phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3], x0_poly[0],
                           x0_poly[1], x0_poly[2], x0_poly[3]);
      });
    };
    BENCHMARK_ADVANCED("CPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        std::vector<std::future<double>> futures;
        for (int i = 0; i < 128; ++i)
          futures.emplace_back(
              std::move(integrator.request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5],
                                           cos_poly[0], cos_poly[1], cos_poly[2], cos_poly[3], phi_poly[0], phi_poly[1],
                                           phi_poly[2], phi_poly[3], x0_poly[0], x0_poly[1], x0_poly[2], x0_poly[3])));
        for (auto &f : futures)
          f.get();
      });
    };
    BENCHMARK_ADVANCED("get CPU 128x")(Catch::Benchmark::Chronometer meter)
    {
      meter.measure([&] {
        for (int i = 0; i < 128; ++i)
          integrator.get(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                         cos_poly[2], cos_poly[3], phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3], x0_poly[0],
                         x0_poly[1], x0_poly[2], x0_poly[3]);
      });
    };
  }
}