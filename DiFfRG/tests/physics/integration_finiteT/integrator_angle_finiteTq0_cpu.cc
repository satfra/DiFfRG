#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/integration_finiteT/integrator_angle_finiteTq0_cpu.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto
  kernel(const double q, const double cos, const double q0, const double /*k*/, const double /*c*/, const double x0,
         const double x1, const double x2, const double x3, const double x4, const double x5, const double cos_x0,
         const double cos_x1, const double cos_x2, const double cos_x3, const double q0_x0, const double q0_x1,
         const double q0_x2, const double q0_x3)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q) + x4 * powr<4>(q) + x5 * powr<5>(q)) *
           (cos_x0 + cos_x1 * powr<1>(cos) + cos_x2 * powr<2>(cos) + cos_x3 * powr<3>(cos)) /
           (q0_x0 + q0_x1 * powr<1>(q0) + q0_x2 * powr<2>(q0) + q0_x3 * powr<3>(q0));
  }

  static __forceinline__ __host__ __device__ auto
  constant(const double /*k*/, const double c, const double /*x0*/, const double /*x1*/, const double /*x2*/,
           const double /*x3*/, const double /*x4*/, const double /*x5*/, const double /*cos_x0*/,
           const double /*cos_x1*/, const double /*cos_x2*/, const double /*cos_x3*/, const double /*q0_x0*/,
           const double /*q0_x1*/, const double /*q0_x2*/, const double /*q0_x3*/)
  {
    return c;
  }
};

TEMPLATE_TEST_CASE_SIG("Test cpu momentum integrals with angle finite T (q0)", "[integration][quadrature integration]",
                       ((int dim), dim), (3), (4))
{
  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const double T = GENERATE(take(5, random(0.01, 1.)));
  const double k = GENERATE(take(2, random(0., 1.)));
  const double val = GENERATE(take(4, random(0.8, 1.2)));

  QuadratureProvider quadrature_provider;
  IntegratorAngleFiniteTq0TBB<dim, double, PolyIntegrand> integrator(quadrature_provider, {{64, 16}}, x_extent, T);
  SECTION("Volume integral")
  {
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI) // spatial part
                                      / T / std::tanh(0.5 / val) / 2. / val;            // sum

    const double integral =
        integrator.get(k, 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., powr<2>(T), 0., powr<2>(val), 0.);

    if (!is_close(reference_integral, integral, dim == 2 ? 1e-2 : 5e-5)) {
      std::cerr << "dim: " << dim << "| T: " << T << "| reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(isfinite(integral));
    CHECK(is_close(reference_integral, integral, dim == 2 ? 1e-2 : 5e-5));
  }
}