#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_1ang.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

#include "../boilerplate/poly_integrand.hh"

//--------------------------------------------
// Quadrature integration

TEMPLATE_TEST_CASE_SIG("Test finite T momentum integrals with 1 angle", "[integration][quadrature]", ((int dim), dim),
                       (3), (4))
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

    const ctype T = GENERATE(take(1, random(0.01, 1.)));
    const ctype x_extent = GENERATE(take(1, random(1., 2.)));
    const uint size = GENERATE(64, 128, 256);

    QuadratureProvider quadrature_provider;
    Integrator_fT_p2_1ang<dim, NT, PolyIntegrand<3, NT, -1>, ExecutionSpace> integrator(quadrature_provider, {size, 16},
                                                                                        x_extent);
    integrator.set_T(T);

    const ctype k = GENERATE(take(1, random(0., 1.)));
    const ctype q_extent = std::sqrt(x_extent * powr<2>(k));

    integrator.set_k(k);

    SECTION("Volume integral")
    {
      const ctype val = GENERATE(take(1, random(0., 1.)));
      integrator.set_typical_E(val);

      const NT reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI) // spatial part
                                    / (std::tanh(val / (2. * T)) * 2. * val);         // sum

      NT integral{};
      integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0., powr<2>(val), 0., 1., 0.);

      constexpr ctype expected_precision = 1e-8;
      const ctype rel_err = t_abs(reference_integral - integral) / t_abs(reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "reference: " << reference_integral << "| integral: " << integral
                  << "| relative error: " << abs(reference_integral - integral) / abs(reference_integral) << std::endl;
      }
      CHECK(rel_err < expected_precision);
    }
    SECTION("Volume map")
    {
      const ctype val = GENERATE(take(1, random(0., 1.)));
      integrator.set_typical_E(val);

      const NT reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI) // spatial part
                                    / (std::tanh(val / (2. * T)) * 2. * val);         // sum

      const uint rsize = GENERATE(32, 64);
      std::vector<NT> integral_view(rsize);
      LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

      integrator.map(integral_view.data(), coordinates, 1., 0., 0., 0., 1., 0., 0., 0., powr<2>(val), 0., 1., 0.)
          .fence();

      constexpr ctype expected_precision = 1e-8;
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
      // sdim spatial dimensions: the polar angle carries the zonal measure (1-c^2)^{(sdim-3)/2} dc,
      // supplied by the Gauss-Jacobi angular quadrature. This section exercises a non-trivial angular
      // integrand and therefore validates the zonal weighting (the volume sections above are angle-flat).
      constexpr int sdim = dim - 1;

      const ctype val = GENERATE(take(1, random(0.1, 1.)));
      integrator.set_typical_E(val);

      const auto x_poly = Polynomial({
          GENERATE(take(1, random(-1., 1.))), // q^0
          GENERATE(take(1, random(-1., 1.))), // q^1
          GENERATE(take(1, random(-1., 1.))), // q^2
          GENERATE(take(1, random(-1., 1.)))  // q^3
      });
      const auto cos_poly = Polynomial({
          GENERATE(take(1, random(1., 2.))),  // c^0 (kept away from zero so the angular reference stays well-scaled)
          GENERATE(take(1, random(-1., 1.))), // c^1
          GENERATE(take(1, random(-1., 1.))), // c^2
          GENERATE(take(1, random(-1., 1.)))  // c^3
      });

      // zonal moments mu_k = \int_{-1}^1 (1-c^2)^{(sdim-3)/2} c^k dc; 0 for odd k, B(m+1/2,(sdim-1)/2) for k=2m.
      using std::tgamma;
      auto zonal_moment = [&](int kk) -> double {
        if (kk % 2 == 1) return 0.;
        const int m = kk / 2;
        return tgamma(m + 0.5) * tgamma((sdim - 1) / 2.) / tgamma(m + sdim / 2.);
      };
      double angular = 0.;
      for (int kk = 0; kk < 4; ++kk)
        angular += (double)cos_poly[kk] * zonal_moment(kk);

      auto int_poly = x_poly;
      std::vector<ctype> coeff_integrand(sdim, 0.);
      coeff_integrand[sdim - 1] = 1.;
      int_poly *= Polynomial(coeff_integrand);

      const NT reference_integral = S_d_prec<ctype>(sdim - 1) * int_poly.integral(0., q_extent) /
                                    powr<sdim>(2. * M_PI) * angular // spatial part with zonal angular measure
                                    / (std::tanh(val / (2. * T)) * 2. * val); // Matsubara sum

      NT integral{};
      integrator.get(integral, 0., x_poly[0], x_poly[1], x_poly[2], x_poly[3], cos_poly[0], cos_poly[1], cos_poly[2],
                     cos_poly[3], powr<2>(val), 0., 1., 0.);

      constexpr ctype expected_precision = 1e-8;
      const ctype rel_err = t_abs(reference_integral - integral) / t_abs(reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "reference: " << reference_integral << "| integral: " << integral
                  << "| relative error: " << abs(reference_integral - integral) / abs(reference_integral)
                  << "| dim: " << dim << std::endl;
      }
      CHECK(rel_err < expected_precision);
    };
  };

  // Check on TBB
  SECTION("TBB") { check(TBB_exec(), (double)0); }
  // Check on Threads
  SECTION("Threads") { check(KokkosHost_exec(), (double)0); }
  // Check on GPU
  SECTION("GPU") { check(GPU_exec(), (double)0); }
}