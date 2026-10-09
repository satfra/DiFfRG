#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2.hh>
#include <DiFfRG/physics/integration/vacuum/integrator_p2.hh>
#include <DiFfRG/physics/interpolation.hh>
#include <DiFfRG/physics/regulators.hh>

using namespace DiFfRG;

#include "../boilerplate/poly_integrand.hh"

//--------------------------------------------
// Quadrature integration

TEMPLATE_TEST_CASE_SIG("Test finite T momentum integrals", "[integration][quadrature]", ((int dim), dim), (2), (3), (4))
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

    const ctype T = GENERATE(1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1., 5., 10.);
    const ctype x_extent = GENERATE(take(1, random(1., 2.)));
    const uint size = GENERATE(16, 32);

    QuadratureProvider quadrature_provider;
    Integrator_fT_p2<dim, NT, PolyIntegrand<2, NT, -1>, ExecutionSpace> integrator(quadrature_provider, {size},
                                                                                   x_extent);

    const ctype k = GENERATE(take(1, random(0., 1.)));
    const ctype q_extent = std::sqrt(x_extent * powr<2>(k));

    SECTION("Volume integral (bosonic)")
    {
      const ctype val = GENERATE(1e-4, 1e-3, 1e-2, 1e-1, 1., 10.);
      integrator.set_T(T);
      integrator.set_k(k);
      integrator.set_typical_E(val);

      const NT reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI) // spatial part
                                    / (std::tanh(val / (2. * T)) * 2. * val);         // sum

      NT integral{};
      integrator.get(integral, 0., 1., 0., 0., 0., powr<2>(val), 0., 1., 0.);

      constexpr ctype expected_precision = 1e-8;
      const ctype rel_err = t_abs(reference_integral - integral) / t_abs(reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "Failure for T = " << T << ", k = " << k << ", val = " << val << "\n"
                  << "reference: " << reference_integral << "| integral: " << integral
                  << "| relative error: " << rel_err << std::endl;
      }
      CHECK(rel_err < expected_precision);
    }
    SECTION("Volume integral (fermionic)")
    {
      const ctype val = GENERATE(1e-4, 1e-3, 1e-2, 1e-1, 1., 10.);
      integrator.set_T(T);
      integrator.set_k(k);
      integrator.set_typical_E(val);

      const NT reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI) // spatial part
                                    * std::tanh(val / (2. * T)) / (2. * val);          // sum

      NT integral{};
      integrator.get(integral, 0., 1., 0., 0., 0., powr<2>(val) + powr<2>(M_PI * T), 2 * M_PI * T, 1., 0.);

      constexpr ctype expected_precision = 1e-8;
      const ctype rel_err = t_abs(reference_integral - integral) / t_abs(reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "Failure for T = " << T << ", k = " << k << ", val = " << val << "\n"
                  << "reference: " << reference_integral << "| integral: " << integral
                  << "| relative error: " << rel_err << std::endl;
      }
      CHECK(rel_err < expected_precision);
    }
    SECTION("Volume map")
    {
      const ctype val = GENERATE(1e-3, 1e-1, 10.);
      integrator.set_T(T);
      integrator.set_k(k);
      integrator.set_typical_E(val);

      const NT reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI) // spatial part
                                    / (std::tanh(val / (2. * T)) * 2. * val);         // sum

      const uint rsize = GENERATE(32, 64);
      std::vector<NT> integral_view(rsize);
      LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

      integrator.map(integral_view.data(), coordinates, 1., 0., 0., 0., powr<2>(val), 0., 1., 0.).fence();

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
  };

  // Check on TBB
  SECTION("TBB") { check(TBB_exec(), (double)0); }
  // Check on Threads
  SECTION("Threads") { check(KokkosHost_exec(), (double)0); }
  // Check on GPU
  SECTION("GPU") { check(GPU_exec(), (double)0); }
}

TEST_CASE("Integrator_fT_p2 in single precision writes double results", "[integration][quadrature][float]")
{
  DiFfRG::Init();

  // The float integrator must agree with the double one to float accuracy, whether it hands its
  // results back through get() or map() into double destinations.
  auto check = [](auto execution_space) {
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;
    constexpr int dim = 4;
    const float m2 = 0.5f;

    QuadratureProvider quadrature_provider;
    Integrator_fT_p2<dim, float, PolyIntegrand<2, float, -1>, ExecutionSpace> integrator_f(quadrature_provider,
                                                                                           {32}, 2.f);
    Integrator_fT_p2<dim, double, PolyIntegrand<2, double, -1>, ExecutionSpace> integrator_d(quadrature_provider,
                                                                                             {32}, 2.);
    integrator_f.set_T(0.1f);
    integrator_d.set_T(0.1);
    integrator_f.set_k(0.8f);
    integrator_d.set_k(0.8);
    integrator_f.set_typical_E(1.f);
    integrator_d.set_typical_E(1.);

    double integral_f = 0., integral_d = 0.;
    integrator_f.get(integral_f, 0.25f, 1.f, 0.f, 0.f, 0.f, m2, 0.f, 1.f, 0.f);
    integrator_d.get(integral_d, 0.25, 1., 0., 0., 0., m2, 0., 1., 0.);
    CHECK(integral_f == Catch::Approx(integral_d).epsilon(1e-5));

    // map() passes the grid position as the kernel's constant term.
    const LinearCoordinates1D<float> coordinates_f(16, 0.f, 1.f);
    const LinearCoordinates1D<double> coordinates_d(16, 0., 1.);
    std::vector<double> mapped_f(coordinates_f.size()), mapped_d(coordinates_d.size());
    integrator_f.map(mapped_f.data(), coordinates_f, 1.f, 0.f, 0.f, 0.f, m2, 0.f, 1.f, 0.f);
    integrator_d.map(mapped_d.data(), coordinates_d, 1., 0., 0., 0., m2, 0., 1., 0.);
    for (size_t i = 0; i < mapped_f.size(); ++i)
      CHECK(mapped_f[i] == Catch::Approx(mapped_d[i]).epsilon(1e-5));
  };

  SECTION("TBB") { check(TBB_exec()); }
  SECTION("Threads") { check(KokkosHost_exec()); }
  SECTION("GPU") { check(GPU_exec()); }
}

TEST_CASE("Test integrator_fT_p2 bug", "[integration][quadrature]")
{
  using NT = double;
  using ctype = typename get_type::ctype<NT>;
  using ExecutionSpace = ExecutionSpaces::TBB_exec_space;
  using Regulator = DiFfRG::PolynomialExpRegulator<>;
  const int dim_fT = 4;
  const int dim = 3;

  DiFfRG::Init();
  const NT T = 1e-2;
  const NT k = 0.65;
  // const double x_extent = GENERATE(take(1, random(1., 2.)));
  const NT x_extent = 2.0;
  // const uint size = GENERATE(64, 128, 256);
  const size_t size = 256;

  QuadratureProvider quadrature_provider;
  Integrator_fT_p2<dim_fT, NT, quark_kernel<Regulator>, ExecutionSpace> integrator_fT(quadrature_provider, {size},
                                                                                      x_extent);
  Integrator_p2<dim, NT, quarkIntegrated_kernel<Regulator>, ExecutionSpace> integrator(quadrature_provider, {size},
                                                                                       x_extent);
  integrator_fT.set_T(T);

  // const double q_extent = std::sqrt(x_extent * powr<2>(k));
  // const double mq2 = GENERATE(0.0, 0.5, 1.0);
  const NT h = 6.2;
  const NT sigma = 0.01;
  const double mq2 = powr<2>(h * sigma);
  integrator_fT.set_typical_E(k);

  NT integral_fT{};
  integrator_fT.get(integral_fT, k, T, mq2);
  NT integralIntegrated{};
  integrator.get(integralIntegrated, k, T, mq2);

  constexpr ctype expected_precision = 1e-8;
  const ctype rel_err = abs(integral_fT - integralIntegrated) / abs(integralIntegrated);
  if (rel_err >= expected_precision) {
    std::cerr << "integral analytic matsubara sum: " << integralIntegrated
              << "| integral numeric matsubara sum: " << integral_fT << "| relative error: " << rel_err << std::endl;
  }
  CHECK(rel_err < expected_precision);

  integrator_fT.set_k(k);
}