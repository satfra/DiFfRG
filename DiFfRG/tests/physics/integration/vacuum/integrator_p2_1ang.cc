#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2_1ang.hh>

using namespace DiFfRG;

#include "../boilerplate/poly_integrand.hh"

//--------------------------------------------
// Quadrature integration

TEMPLATE_TEST_CASE_SIG("Test momentum + 1 angle integrals", "[integration][quadrature]", ((int dim), dim), (2), (3),
                       (4), (5))
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

    const ctype x_extent = GENERATE(take(1, random(1., 2.)));
    const uint size = GENERATE(32, 64, 128);

    QuadratureProvider quadrature_provider;
    Integrator_p2_1ang<dim, NT, PolyIntegrand<2, NT>, ExecutionSpace> integrator(quadrature_provider, {size, 8},
                                                                                 x_extent);

    SECTION("Volume integral")
    {
      const ctype k = GENERATE(take(1, random(0., 1.)));
      const ctype q_extent = std::sqrt(x_extent * powr<2>(k));
      const NT reference_integral = V_d(dim, q_extent) / powr<dim>(2. * M_PI);

      integrator.set_k(k);

      NT integral{};
      integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0.);

      constexpr ctype expected_precision = 1e-14;
      const ctype rel_err = t_abs((integral - reference_integral) / reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "reference: " << std::scientific << std::setw(10) << reference_integral
                  << " | integral: " << std::setw(10) << integral << " | relative error: " << std::setw(12) << rel_err
                  << " | expected precision: " << std::setw(12) << expected_precision << " | order: " << std::setw(5)
                  << size << " | dim: " << std::setw(3) << dim << std::endl;
      }
      CHECK(rel_err < expected_precision);
    }
    SECTION("Volume map")
    {
      const ctype k = GENERATE(take(1, random(0., 1.)));
      const ctype q_extent = std::sqrt(x_extent * powr<2>(k));
      const NT reference_integral = V_d(dim, q_extent) / powr<dim>(2. * M_PI);

      integrator.set_k(k);

      std::array<NT, 4 * 2> coeffs{};
      for (uint d = 0; d < 2; ++d)
        coeffs[4 * d] = 1;

      const uint rsize = GENERATE(32, 64);
      std::vector<NT> integral_view(rsize);
      LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

      std::apply([&](auto... coeffs) { integrator.map(integral_view.data(), coordinates, coeffs...).fence(); }, coeffs);

      constexpr ctype expected_precision = 1e-14;
      for (uint i = 0; i < rsize; ++i) {
        const ctype rel_err = t_abs(coordinates.forward(i) + reference_integral - integral_view[i]) /
                              t_abs(coordinates.forward(i) + reference_integral);
        if (rel_err >= expected_precision) {
          std::cout << "reference: " << coordinates.forward(i) + reference_integral
                    << "| integral: " << integral_view[i] << "| relative error: "
                    << abs(coordinates.forward(i) + reference_integral - integral_view[i]) /
                           abs(coordinates.forward(i) + reference_integral)
                    << "| type: " << typeid(type).name() << "| i: " << i << std::endl;
        }
        CHECK(rel_err < expected_precision);
      }
    };
    SECTION("Random polynomials")
    {
      const auto x_poly = Polynomial({
          dim == 1 ? 0. : GENERATE(take(1, random(-1., 1.))), // x0
          GENERATE(take(2, random(-1., 1.))),                 // x1
          GENERATE(take(2, random(-1., 1.))),                 // x2
          GENERATE(take(2, random(-1., 1.)))                  // x3
      });
      const auto cos_poly = Polynomial({
          GENERATE(take(2, random(-1., 1.))), // x0
          GENERATE(take(2, random(-1., 1.))), // x1
          GENERATE(take(1, random(-1., 1.))), // x2
          GENERATE(take(1, random(-1., 1.)))  // x3
      });

      const ctype k = GENERATE(take(2, random(0., 1.)));
      const ctype q_extent = std::sqrt(x_extent * powr<2>(k));
      const NT constant = GENERATE(take(2, random(-1., 1.)));

      auto int_poly = x_poly;
      std::vector<ctype> coeff_integrand(dim, 0.);
      coeff_integrand[dim - 1] = 1.;
      int_poly *= Polynomial(coeff_integrand);

      // The polar angle carries the zonal measure (1-c^2)^{(dim-3)/2} dc (supplied by the Gauss-Jacobi
      // quadrature), so the angular reference uses the zonal moments mu_k = \int_{-1}^1 (1-c^2)^{(dim-3)/2} c^k dc,
      // not the plain Legendre integral. mu_k = 0 for odd k, and mu_{2m} = B(m+1/2,(dim-1)/2).
      using std::tgamma;
      auto zonal_moment = [&](int kk) -> double {
        if (kk % 2 == 1) return 0.;
        const int m = kk / 2;
        return tgamma(m + 0.5) * tgamma((dim - 1) / 2.) / tgamma(m + dim / 2.);
      };
      double angular = 0.;
      for (int kk = 0; kk < 4; ++kk)
        angular += (double)cos_poly[kk] * zonal_moment(kk);
      const NT reference_integral =
          constant + S_d_prec<ctype>(dim - 1) * int_poly.integral(0., q_extent) / powr<dim>(2. * M_PI) * angular;

      integrator.set_k(k);

      double integral{};
      integrator.get(integral, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3], cos_poly[0], cos_poly[1],
                     cos_poly[2], cos_poly[3]);

      // 1e-11: GPU summation reorders the reduction, so the higher-degree random-polynomial
      // integrand drifts above tighter gates (observed up to ~1.4e-12 on CI at order 32).
      constexpr ctype expected_precision = 1e-11;
      // Relative to the size of the summed parts: constant and integral are random and can nearly
      // cancel, which would turn a roundoff-level absolute error into a large relative one.
      const ctype rel_err =
          t_abs(integral - reference_integral) / (t_abs(constant) + t_abs(reference_integral - constant));
      if (rel_err >= expected_precision) {
        std::cerr << "reference: " << std::scientific << std::setw(10) << reference_integral
                  << " | integral: " << std::setw(10) << integral << " | relative error: " << std::setw(12) << rel_err
                  << " | expected precision: " << std::setw(12) << expected_precision << " | order: " << std::setw(5)
                  << size << " | dim: " << std::setw(3) << dim << std::endl;
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