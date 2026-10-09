#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "../boilerplate/poly_integrand.hh"
#include "DiFfRG/discretization/coordinates/coordinates.hh"
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat.hh>

using namespace DiFfRG;

TEST_CASE("Benchmark 4D lattice cpu momentum integrals", "[integration][lattice integration]")
{
  DiFfRG::Init();

  using ctype = double;
  using T = double;

  const ctype a0 = GENERATE(take(1, random(0.01, 1.)));
  const ctype a1 = GENERATE(take(1, random(0.01, 1.)));

  const double constant = GENERATE(take(1, random(-1., 1.)));
  const auto poly = Polynomial({
      GENERATE(take(1, random(-1., 1.))), // x0
      GENERATE(take(1, random(-1., 1.))), // x1
      GENERATE(take(1, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.)))  // x3
  });

  constexpr_for<5, 8, 1>([&](auto j) {
    constexpr_for<5, 8, 1>([&](auto i) {
      constexpr uint rsize = powr<j>(2);
      constexpr uint isize = powr<i>(2);

      {
        IntegratorLat4D<T, PolyIntegrand<4, T>, KokkosHost_exec> integrator({{isize, isize}}, {{a0, a1}}, true);
        Kokkos::View<double *, CPU_memory> integral_view("cpu_integral_results", rsize);

        BENCHMARK_ADVANCED("OPT host isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure([&] {
            for (uint k = 0; k < rsize; ++k) {
              // make subview
              auto subview = Kokkos::subview(integral_view, k);
              integrator.reduce(KokkosHost_exec(), subview, constant, poly[0], poly[1], poly[2], poly[3], 1., 0., 0., 0., 1., 0., 0., 0., 1.,
                             0., 0., 0.);
            }
            Kokkos::fence();
          });
        };
      }
    });
  });
}

TEST_CASE("Benchmark 3D lattice cpu momentum integrals", "[integration][lattice integration][3D]")
{
  DiFfRG::Init();

  using ctype = double;
  using T = double;

  const ctype a0 = GENERATE(take(1, random(0.01, 1.)));
  const ctype a1 = GENERATE(take(1, random(0.01, 1.)));

  const double constant = GENERATE(take(1, random(-1., 1.)));
  const auto poly = Polynomial({
      GENERATE(take(1, random(-1., 1.))), // x0
      GENERATE(take(1, random(-1., 1.))), // x1
      GENERATE(take(1, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.)))  // x3
  });

  constexpr_for<5, 9, 1>([&](auto j) {
    constexpr_for<5, 8, 1>([&](auto i) {
      constexpr uint rsize = powr<j>(2);
      constexpr uint isize = powr<i>(2);

      {
        IntegratorLat3D<T, PolyIntegrand<3, T>, KokkosHost_exec> integrator({{isize, isize}}, {{a0, a1}}, true);
        Kokkos::View<double *, CPU_memory> integral_view("cpu_integral_results", rsize);

        BENCHMARK_ADVANCED("OPT host isize=" + std::to_string(isize) +
                           " rsize=" + std::to_string(rsize))(Catch::Benchmark::Chronometer meter)
        {
          meter.measure([&] {
            for (uint k = 0; k < rsize; ++k) {
              // make subview
              auto subview = Kokkos::subview(integral_view, k);
              integrator.reduce(KokkosHost_exec(), subview, constant, poly[0], poly[1], poly[2], poly[3], 1., 0., 0., 0., 1., 0., 0., 0.);
            }
            Kokkos::fence();
          });
        };
      }
    });
  });
}