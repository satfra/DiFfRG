#include <Kokkos_Core.hpp>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "../boilerplate/lattice_sum_tolerance.hh"
#include "../boilerplate/poly_integrand.hh"
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat.hh>

using namespace DiFfRG;

TEMPLATE_TEST_CASE("Test 2D lattice integrals on host", "[lattice][double][float][complex][autodiff]", double, float,
                   complex<double>, complex<float>, autodiff::real)
{
  using T = TestType;

  DiFfRG::Init();

  using ctype = typename get_type::ctype<T>;

  const ctype a0 = GENERATE(take(2, random(0.01, 1.)));
  const ctype a1 = GENERATE(take(2, random(0.01, 1.)));
  const uint size0 = GENERATE(16, 32, 64, 128);
  const uint size1 = GENERATE(16, 32, 64, 128);
  const bool q0_symmetric = GENERATE(false, true);

  IntegratorLat2D<T, PolyIntegrand<2, T>, KokkosHost_exec> integrator({{size0, size1}}, {{a0, a1}}, q0_symmetric);

  SECTION("Volume integral")
  {
    const ctype reference_integral = 1. / a0 / a1;

    const std::size_t n_terms = lattice_sum_terms(2, size0, size1, q0_symmetric);
    const ctype expected_precision = lattice_sum_tolerance<ctype>(n_terms);
    if (expected_precision > lattice_sum_meaningful_tolerance)
      SKIP("summing " << n_terms << " weights in this precision leaves a round-off bound of "
                      << expected_precision << ", too coarse to test anything");

    T integral{};
    integrator.get(integral, 0., 1., 0., 0., 0., 1., 0., 0., 0.);

    if (!is_close(reference_integral, integral, expected_precision)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << "| tolerance: " << expected_precision << "| terms: " << n_terms
                << "| sizes: " << size0 << "," << size1 << "| a: " << a0 << "," << a1
                << "| symm: " << q0_symmetric << std::endl;
    }
    CHECK(is_close(reference_integral, integral, expected_precision));
  }
}