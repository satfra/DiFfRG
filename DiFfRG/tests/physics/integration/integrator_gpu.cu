#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/integration/integrator_gpu.hh>
#include <DiFfRG/physics/loop_integrals.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto kernel(const double q, const double /*k*/, const double /*c*/,
                                                         const double x0, const double x1, const double x2,
                                                         const double x3, const double x4, const double x5)
  {
    return x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q) + x4 * powr<4>(q) + x5 * powr<5>(q);
  }

  static __forceinline__ __host__ __device__ auto constant(const double /*k*/, const double c, const double /*x0*/,
                                                           const double /*x1*/, const double /*x2*/,
                                                           const double /*x3*/, const double /*x4*/,
                                                           const double /*x5*/)
  {
    return c;
  }
};

TEMPLATE_TEST_CASE_SIG("Test gpu momentum integrals", "[GPU][integration][quadrature integration]", ((int dim), dim),
                       (1), (2), (3), (4))
{
  const double x_extent = GENERATE(take(1, random(1., 2.)));
  QuadratureProvider quadrature_provider;
  IntegratorGPU<dim, double, PolyIntegrand> integrator(quadrature_provider, 64, x_extent);

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double reference_integral = V_d(dim, q_extent) / powr<dim>(2. * M_PI);

    const double integral = integrator.request(k, 0., 1., 0., 0., 0., 0., 0.).get();

    if (!is_close(reference_integral, integral, dim == 1 ? 1e-2 : 1e-6)) {
      std::cerr << "dim: " << dim << "| reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, dim == 1 ? 1e-2 : 1e-6));
  }

  SECTION("Random polynomials")
  {
    const auto poly = Polynomial({
        dim == 1 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
        GENERATE(take(2, random(-1., 1.))),                 // x1
        GENERATE(take(2, random(-1., 1.))),                 // x2
        GENERATE(take(2, random(-1., 1.))),                 // x3
        GENERATE(take(2, random(-1., 1.))),                 // x4
        GENERATE(take(2, random(-1., 1.))),                 // x5
    });

    const double k = GENERATE(take(2, random(0., 1.)));
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double constant = GENERATE(take(2, random(-1., 1.)));

    auto int_poly = poly;
    std::vector<double> coeff_integrand(dim, 0.);
    coeff_integrand[dim - 1] = 1.;
    int_poly *= Polynomial(coeff_integrand);
    const double reference_integral = constant + S_d(dim) * int_poly.integral(0., q_extent) / powr<dim>(2. * M_PI);

    const double integral = integrator.request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5]).get();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "dim: " << dim << "| reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}