#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DiFfRG/discretization/coordinates/coordinates.hh"
#include "boilerplate/poly_integrand.hh"
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>

using namespace DiFfRG;

TEST_CASE("Benchmark cpu momentum integrals", "[integration][quadrature integration]")
{
  DiFfRG::Init();

  const double x_min = GENERATE(take(1, random(-2., -1.)));
  const double x_max = GENERATE(take(1, random(1., 2.)));

  QuadratureProvider quadrature_provider;

  const double constant = GENERATE(take(1, random(-1., 1.)));
  const auto poly = Polynomial({
      GENERATE(take(1, random(-1., 1.))), // x0
      GENERATE(take(1, random(-1., 1.))), // x1
      GENERATE(take(1, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.)))  // x3
  });

  // The "kokkos-host" rows use KokkosHost_exec, which is Kokkos::Serial unless the build sets
  // KOKKOS_THREADS=ON -- so they are a *serial* CPU baseline, not DiFfRG's CPU path. The threaded
  // CPU path is TBB_exec; benchmark_gpu_vs_tbb.cc is the one that contrasts it with the device.
  constexpr_for<5, 9, 1>([&](auto j) {
    constexpr_for<5, 9, 1>([&](auto i) {
      constexpr uint rsize = powr<j>(2);
      constexpr uint isize = powr<i>(2);
      {
        QuadratureIntegrator<1, double, PolyIntegrand<1, double>, KokkosHost_exec> integrator(
            quadrature_provider, {isize}, {x_min}, {x_max}, {QuadratureType::legendre});
        Kokkos::View<double *, KokkosHost_memory> integral_view("cpu_integral_results", rsize);

        BENCHMARK_ADVANCED("kokkos-host isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure([&] {
            for (uint k = 0; k < rsize; ++k) {
              // make subview
              auto subview = Kokkos::subview(integral_view, k);
              integrator.get(subview, constant, poly[0], poly[1], poly[2], poly[3]);
            }
            Kokkos::fence();
          });
        };
      }
      {
        QuadratureIntegrator<1, double, PolyIntegrand<1, double>, GPU_exec> integrator(
            quadrature_provider, {isize}, {x_min}, {x_max}, {QuadratureType::legendre});
        Kokkos::View<double *, GPU_memory> integral_view("cpu_integral_results", rsize);

        BENCHMARK_ADVANCED("device isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure([&] {
            for (uint k = 0; k < rsize; ++k) {
              // make subview
              auto subview = Kokkos::subview(integral_view, k);
              integrator.get(subview, constant, poly[0], poly[1], poly[2], poly[3]);
            }
            Kokkos::fence();
          });
        };
      }
      {
        QuadratureIntegrator<1, double, PolyIntegrand<1, double>, KokkosHost_exec> integrator(
            quadrature_provider, {isize}, {x_min}, {x_max}, {QuadratureType::legendre});
        std::vector<double> integral_view(rsize);
        LinearCoordinates1D<double> coordinates(rsize, 0., 1.);

        BENCHMARK_ADVANCED("kokkos-host nested isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure(
              [&] { integrator.map(integral_view.data(), coordinates, poly[0], poly[1], poly[2], poly[3]).fence(); });
        };

        // Check the results
        for (uint k = 0; k < rsize; ++k) {
          const double result = integral_view[k];
          const double reference_integral = coordinates.forward(k) + poly.integral(x_min, x_max);
          if (!is_close(reference_integral, result, 1e-6)) {
            std::cerr << "reference: " << reference_integral << "| integral: " << result
                      << "| relative error: " << abs(reference_integral - result) / abs(reference_integral)
                      << std::endl;
          }
          CHECK(is_close(reference_integral, result, 1e-6));
        }
      }
      {
        QuadratureIntegrator<1, double, PolyIntegrand<1, double>, GPU_exec> integrator(
            quadrature_provider, {isize}, {x_min}, {x_max}, {QuadratureType::legendre});
        std::vector<double> integral_view(rsize);
        LinearCoordinates1D<double> coordinates(rsize, 0., 1.);

        BENCHMARK_ADVANCED("device nested isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure(
              [&] { integrator.map(integral_view.data(), coordinates, poly[0], poly[1], poly[2], poly[3]).fence(); });
        };

        // Check the results
        for (uint k = 0; k < rsize; ++k) {
          const double result = integral_view[k];
          const double reference_integral = coordinates.forward(k) + poly.integral(x_min, x_max);
          if (!is_close(reference_integral, result, 1e-6)) {
            std::cerr << "reference: " << reference_integral << "| integral: " << result
                      << "| relative error: " << abs(reference_integral - result) / abs(reference_integral)
                      << std::endl;
          }
          CHECK(is_close(reference_integral, result, 1e-6));
        }
      }
    });
  });
}