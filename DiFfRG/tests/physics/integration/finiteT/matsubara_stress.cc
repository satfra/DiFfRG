#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT.hh>
#include <DiFfRG/physics/interpolation.hh>

using namespace DiFfRG;

#include "../boilerplate/poly_integrand.hh"

//--------------------------------------------
// Quadrature integration

TEST_CASE("Stress test p0 sums/integrals", "[integration][quadrature]")
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

    const ctype T = GENERATE(0., 1e-4, 1e-3, 1e-2, 1e-1, 1.);

    QuadratureProvider quadrature_provider;
    Integrator_fT<1, NT, PolyIntegrand<1, NT, -1>, ExecutionSpace> integrator(quadrature_provider, {}, {});

    const ctype k = GENERATE(take(1, random(0., 1.)));

    SECTION("Volume integral (bosonic)")
    {
      const ctype val = GENERATE(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.);
      integrator.set_T(T);
      integrator.set_k(k);
      integrator.set_typical_E(val);

      const NT reference_integral = T == 0. ? 1. / (2. * val) : 1. / (std::tanh(val / (2. * T)) * 2. * val); // sum

      NT integral{};
      integrator.get(integral, 0., powr<2>(val), 0., 1., 0.);

      // This is actually the precision we can expect at low T
      constexpr ctype expected_precision = 1e-9;
      const ctype rel_err = t_abs(reference_integral - integral) / t_abs(reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "Failure for T = " << T << ", k = " << k << ", val = " << val << "\n"
                  << "reference: " << reference_integral << "| integral: " << integral
                  << "| relative error: " << rel_err << std::endl;
      }
      CHECK(rel_err < expected_precision);
      std::cout << "BOS T = " << T << ", val = " << val << ", rel_err = " << rel_err
                << ", matsubara_size = " << integrator.get_matsubara_size() << std::endl;
    }

    SECTION("Volume integral (fermionic)")
    {
      const ctype val = GENERATE(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.);
      integrator.set_T(T);
      integrator.set_k(k);
      integrator.set_typical_E(val);

      const NT reference_integral = T == 0. ? 1. / (2. * val) : std::tanh(val / (2. * T)) / (2. * val); // sum

      NT integral{};
      integrator.get(integral, 0., powr<2>(val) + powr<2>(M_PI * T), 2 * M_PI * T, 1., 0.);

      // This is actually the precision we can expect at low T
      constexpr ctype expected_precision = 1e-9;
      const ctype rel_err = t_abs(reference_integral - integral) / t_abs(reference_integral);
      if (rel_err >= expected_precision) {
        std::cerr << "Failure for T = " << T << ", k = " << k << ", val = " << val << "\n"
                  << "reference: " << reference_integral << "| integral: " << integral
                  << "| relative error: " << rel_err << std::endl;
      }
      CHECK(rel_err < expected_precision);
      std::cout << "FERM T = " << T << ", val = " << val << ", rel_err = " << rel_err
                << ", matsubara_size = " << integrator.get_matsubara_size() << std::endl;
    }
  };

  // Check on TBB
  SECTION("TBB") { check(TBB_exec(), (double)0); }
  // Check on Threads
  SECTION("Threads") { check(KokkosHost_exec(), (double)0); }
  // Check on GPU
  SECTION("GPU") { check(GPU_exec(), (double)0); }
}
