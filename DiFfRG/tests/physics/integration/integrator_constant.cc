#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/integration/integrator_constant.hh>
#include <DiFfRG/physics/loop_integrals.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto kernel(const double /*k*/, const double /*c*/, const double x0)
  {
    return x0;
  }

  static __forceinline__ __host__ __device__ auto constant(const double /*k*/, const double c, const double /*x0*/)
  {
    return c;
  }
};

TEMPLATE_TEST_CASE_SIG("Test cpu momentum integrals", "[integration][quadrature integration]", ((int dim), dim), (1),
                       (2), (3), (4))
{
  QuadratureProvider quadrature_provider;
  IntegratorConstant<dim, double, PolyIntegrand> integrator(quadrature_provider, 64, 1.);

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double reference_integral = S_d(dim) / powr<dim>(2. * M_PI);

    const double integral = integrator.request(k, 0., 1.).get();

    if (!is_close(reference_integral, integral, dim == 1 ? 1e-2 : 1e-6)) {
      std::cerr << "dim: " << dim << "| reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }

  SECTION("Random polynomials")
  {
    const double k = GENERATE(take(2, random(0., 1.)));
    const double constant = GENERATE(take(2, random(-1., 1.)));
    const double x0 = GENERATE(take(2, random(-1., 1.)));

    const double reference_integral = constant + S_d(dim) * x0 / powr<dim>(2. * M_PI);

    const double integral = integrator.request(k, constant, x0).get();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "dim: " << dim << "| reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}