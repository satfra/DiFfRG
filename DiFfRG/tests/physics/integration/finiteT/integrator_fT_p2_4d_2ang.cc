#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_4d_2ang.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

#include "../boilerplate/poly_integrand.hh"

//--------------------------------------------
// Quadrature integration

TEST_CASE("Test finite T 4D momentum + 2 angle integrals", "[integration][quadrature]")
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
    Integrator_fT_p2_4D_2ang<4, NT, PolyIntegrand<4, NT, -1>, ExecutionSpace> integrator(quadrature_provider,
                                                                                         {size, 8, 8}, x_extent);
    integrator.set_T(T);

    const ctype k = GENERATE(take(1, random(0., 1.)));
    const ctype q_extent = std::sqrt(x_extent * powr<2>(k));

    integrator.set_k(k);

    SECTION("Volume integral")
    {
      const ctype val = GENERATE(take(1, random(0., 1.)));
      integrator.set_typical_E(val);

      const NT reference_integral = V_d(3, q_extent) / powr<3>(2. * M_PI)     // spatial part
                                    / (std::tanh(val / (2. * T)) * 2. * val); // sum

      NT integral{};
      integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., powr<2>(val), 0., 1., 0.);

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

      const NT reference_integral = V_d(3, q_extent) / powr<3>(2. * M_PI)     // spatial part
                                    / (std::tanh(val / (2. * T)) * 2. * val); // sum

      const uint rsize = GENERATE(32, 64);
      std::vector<NT> integral_view(rsize);
      LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

      integrator
          .map(integral_view.data(), coordinates, 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., powr<2>(val), 0., 1.,
               0.)
          .fence();

      constexpr ctype expected_precision = 1e-8;
      for (uint i = 0; i < rsize; ++i) {
        std::cout << "i: " << i << "| coordinates: " << coordinates.forward(i)
                  << "| reference: " << coordinates.forward(i) + reference_integral
                  << "| integral: " << integral_view[i] << std::endl;
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
  };

  // Check on TBB
  SECTION("TBB") { check(TBB_exec(), (double)0); }
  // Check on Threads
  SECTION("Threads") { check(KokkosHost_exec(), (double)0); }
  // Check on GPU
  SECTION("GPU") { check(GPU_exec(), (double)0); }
}