#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/quadrature_provider.hh>
#include <DiFfRG/physics/integration_finiteT/integrator_4D_finiteTq0_cpu.hh>
#include <DiFfRG/physics/integration_finiteT/integrator_4D_finiteTx0_cpu.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto
  kernel(const double q, const double cos, const double phi, const double q0, const double /*k*/, const double /*c*/,
         const double x0, const double x1, const double x2, const double x3, const double x4, const double x5,
         const double cos_x0, const double cos_x1, const double cos_x2, const double cos_x3, const double phi_x0,
         const double phi_x1, const double phi_x2, const double phi_x3, const double q0_x0, const double q0_x1,
         const double q0_x2, const double q0_x3)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q) + x4 * powr<4>(q) + x5 * powr<5>(q)) *
           (cos_x0 + cos_x1 * powr<1>(cos) + cos_x2 * powr<2>(cos) + cos_x3 * powr<3>(cos)) *
           (phi_x0 + phi_x1 * powr<1>(phi) + phi_x2 * powr<2>(phi) + phi_x3 * powr<3>(phi)) /
           (q0_x0 + q0_x1 * powr<1>(q0) + q0_x2 * powr<2>(q0) + q0_x3 * powr<3>(q0));
  }

  static __forceinline__ __host__ __device__ auto
  constant(const double /*k*/, const double c, const double /*x0*/, const double /*x1*/, const double /*x2*/,
           const double /*x3*/, const double /*x4*/, const double /*x5*/, const double /*cos_x0*/,
           const double /*cos_x1*/, const double /*cos_x2*/, const double /*cos_x3*/, const double /*phi_x0*/,
           const double /*phi_x1*/, const double /*phi_x2*/, const double /*phi_x3*/, const double /*q0_x0*/,
           const double /*q0_x1*/, const double /*q0_x2*/, const double /*q0_x3*/)
  {
    return c;
  }
};

TEST_CASE("Test 4D cpu momentum integrals with finite T (x0)", "[4D integration][quadrature integration]")
{
  constexpr int dim = 4;

  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const uint x0_summands = 32;
  const uint x0_int_order = 64;
  const double T = GENERATE(0., take(3, random(0.01, 1.)));
  const double k = GENERATE(take(2, random(0., 1.)));
  const double x0_extent = x0_summands * 10 * 2. * M_PI * T / k * GENERATE(take(1, random(1., 2.))) + 1000. / k;
  QuadratureProvider quadrature_provider;
  Integrator4DFiniteTx0TBB<double, PolyIntegrand> integrator(quadrature_provider, {{64, 12, 12, x0_int_order}},
                                                             x_extent, x0_extent, x0_summands, T);

  SECTION("Volume integral")
  {
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double q0_extent = x0_extent * k;
    const double reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI)                 // spatial part
                                      * ((2 * x0_summands - 1) * T                                      // summands
                                         + (q0_extent - 2. * M_PI * T * x0_summands) * 2. / (2. * M_PI) // integral
                                        );

    const double integral =
        integrator.request(k, 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.).get();

    if (!is_close(reference_integral, integral, dim == 2 ? 1e-2 : 5e-5)) {
      std::cerr << "dim: " << dim << "| reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(isfinite(integral));
    CHECK(is_close(reference_integral, integral, dim == 2 ? 1e-2 : 5e-5));
  }

  SECTION("Compare against q0 integrator")
  {
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double q0_extent = x0_extent * k;
    const double val = GENERATE(take(4, random(0.8, 1.2)));
    const double reference_integral = V_d(dim - 1, q_extent) / powr<dim - 1>(2. * M_PI) // spatial part
                                      / T / std::tanh(0.5 / val) / 2. / val;            // sum

    Integrator4DFiniteTq0TBB<double, PolyIntegrand> integrator_cpu(quadrature_provider, {{64, 12, 12, x0_int_order}},
                                                                   x_extent, q0_extent, x0_summands, T);

    const double int_gpu =
        integrator
            .request(k, 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., powr<2>(T), 0., powr<2>(val), 0.)
            .get();
    const double int_cpu =
        integrator_cpu
            .request(k, 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., powr<2>(T), 0., powr<2>(val), 0.)
            .get();

    if (!is_close(reference_integral, int_cpu, dim == 2 ? 1e-2 : 5e-5)) {
      std::cerr << "dim: " << dim << "| reference: " << reference_integral << "| integral: " << int_cpu
                << "| relative error: " << std::abs(reference_integral - int_cpu) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(isfinite(int_cpu));
    CHECK(isfinite(int_gpu));
    CHECK(is_close(reference_integral, int_cpu, dim == 2 ? 1e-2 : 5e-5));
    CHECK(is_close(reference_integral, int_gpu, dim == 2 ? 1e-2 : 5e-5));
  }
}