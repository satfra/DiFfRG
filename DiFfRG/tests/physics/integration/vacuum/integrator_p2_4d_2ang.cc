#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_4d_2ang.hh>

#include "../boilerplate/poly_integrand.hh"

using namespace DiFfRG;

TEST_CASE("Test 4D momentum + 2 angle integrals", "[integration][quadrature]")
{
  DiFfRG::Init();

  auto check = [](auto execution_space, auto type) {
    using NT = std::decay_t<decltype(type)>;
    using ctype = typename get_type::ctype<NT>;
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;

    using Kokkos::abs;
    auto t_abs = [](const auto val) {
      using type = std::decay_t<decltype(val)>;
      using Kokkos::abs;
      if constexpr (std::is_same_v<type, autodiff::real>)
        return abs(autodiff::val(val)) + abs(autodiff::grad(val));
      else if constexpr (std::is_same_v<type, cxreal>)
        return abs(autodiff::val(val)) + abs(autodiff::grad(val));
      else
        return abs(val);
    };

    constexpr int dim = 4;

    const ctype x_extent = GENERATE(take(1, random(1., 2.)));
    QuadratureProvider quadrature_provider;
    Integrator_p2_4D_2ang<4, NT, PolyIntegrand<3, NT>, ExecutionSpace> integrator(quadrature_provider, {32, 8, 8},
                                                                                  x_extent);

    SECTION("Volume integral")
    {
      const ctype k = GENERATE(take(1, random(0., 1.)));
      const ctype q_extent = std::sqrt(x_extent * powr<2>(k));
      const NT reference_integral = V_d(dim, q_extent) / powr<dim>(2. * M_PI);

      integrator.set_k(k);

      NT integral{};
      integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.);

      constexpr ctype expected_precision = 1e-14;
      const ctype rel_err = t_abs((integral - reference_integral) / reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "reference: " << std::scientific << std::setw(10) << reference_integral
                  << " | integral: " << std::setw(10) << integral << " | relative error: " << std::setw(12) << rel_err
                  << " | expected precision: " << std::setw(12) << expected_precision << std::endl;
      }
      CHECK(rel_err < expected_precision);
    }

    SECTION("Random polynomials")
    {
      constexpr uint take_n = 2;
      const auto poly = Polynomial({
          GENERATE(take(take_n, random(-1., 1.))), // x0
          GENERATE(take(take_n, random(-1., 1.))), // x1
          GENERATE(take(take_n, random(-1., 1.))), // x2
          GENERATE(take(1, random(-1., 1.)))       // x3
      });
      const auto cos1_poly = Polynomial({
          GENERATE(take(take_n, random(-1., 1.))), // x0
          GENERATE(take(1, random(-1., 1.))),      // x1
          GENERATE(take(1, random(-1., 1.))),      // x2
          GENERATE(take(1, random(-1., 1.)))       // x3
      });
      const auto cos2_poly = Polynomial({
          GENERATE(take(take_n, random(-1., 1.))), // x0
          GENERATE(take(1, random(-1., 1.))),      // x1
          GENERATE(take(1, random(-1., 1.))),      // x2
          GENERATE(take(1, random(-1., 1.)))       // x3
      });

      const ctype k = GENERATE(take(take_n, random(0., 1.)));
      const ctype q_extent = std::sqrt(x_extent * powr<2>(k));
      const NT constant = GENERATE(take(take_n, random(-1., 1.)));

      auto int_poly = poly;
      std::vector<ctype> coeff_integrand(dim, 0.);
      coeff_integrand[dim - 1] = 1.;
      int_poly *= Polynomial(coeff_integrand);
      const NT reference_integral = constant + int_poly.integral(0., q_extent) *
                                                   ((4 * cos1_poly[0] + cos1_poly[2]) * M_PI / 8.) *
                                                   cos2_poly.integral(-1., 1.) * 2. * M_PI / powr<dim>(2. * M_PI);

      integrator.set_k(k);

      NT integral{};
      integrator.get(integral, constant, poly[0], poly[1], poly[2], poly[3], cos1_poly[0], cos1_poly[1], cos1_poly[2],
                     cos1_poly[3], cos2_poly[0], cos2_poly[1], cos2_poly[2], cos2_poly[3]);

      constexpr ctype expected_precision = 1e-12;
      // Relative to the size of the summed parts: constant and integral are random and can nearly
      // cancel, which would turn a roundoff-level absolute error into a large relative one.
      const ctype rel_err =
          t_abs(integral - reference_integral) / (t_abs(constant) + t_abs(reference_integral - constant));
      if (rel_err >= expected_precision) {
        std::cerr << "reference: " << std::scientific << std::setw(10) << reference_integral
                  << " | integral: " << std::setw(10) << integral << " | relative error: " << std::setw(12) << rel_err
                  << " | expected precision: " << std::setw(12) << expected_precision << std::endl;
      }
      CHECK(rel_err < expected_precision);
    }
  };

  // Check on TBB
  SECTION("TBB") { check(TBB_exec(), (double)0); }
  // Check on Threads
  SECTION("Threads") { check(KokkosHost_exec(), (double)0); }
  // Check on GPU
  SECTION("GPU") { check(GPU_exec(), (double)0); }
}