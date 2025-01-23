#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/integrator_3D_cartesian_cpu.hh>
#include <DiFfRG/physics/integration/quadrature_provider.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto
  kernel(const double qx, const double qy, const double qz, const double /*k*/, const double /*c*/, const double x0,
         const double x1, const double x2, const double x3, const double y0, const double y1, const double y2,
         const double y3, const double z0, const double z1, const double z2, const double z3)
  {
    return (x0 + x1 * powr<1>(qx) + x2 * powr<2>(qx) + x3 * powr<3>(qx)) *
           (y0 + y1 * powr<1>(qy) + y2 * powr<2>(qy) + y3 * powr<3>(qy)) *
           (z0 + z1 * powr<1>(qz) + z2 * powr<2>(qz) + z3 * powr<3>(qz));
  }

  static __forceinline__ __host__ __device__ auto
  constant(const double /*k*/, const double c, const double /*x0*/, const double /*x1*/, const double /*x2*/,
           const double /*x3*/, const double /*y0*/, const double /*y1*/, const double /*y2*/, const double /*y3*/,
           const double /*z0*/, const double /*z1*/, const double /*z2*/, const double /*z3*/)
  {
    return c;
  }
};

TEST_CASE("Test 2d cartesian cpu momentum integrals", "[integration][quadrature integration]")
{
  const double qx_min = GENERATE(take(2, random(-2., -1.)));
  const double qy_min = GENERATE(take(2, random(-2., -1.)));
  const double qz_min = GENERATE(take(2, random(-2., -1.)));
  const double qx_max = GENERATE(take(2, random(1., 2.)));
  const double qy_max = GENERATE(take(2, random(1., 2.)));
  const double qz_max = GENERATE(take(2, random(1., 2.)));

  QuadratureProvider quadrature_provider;
  Integrator3DCartesianTBB<double, PolyIntegrand> integrator(quadrature_provider, {{32, 32, 32}}, 0., 0, qx_min, qy_min,
                                                             qz_min, qx_max, qy_max, qz_max);

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double reference_integral = (qx_max - qx_min) * (qy_max - qy_min) * (qz_max - qz_min) / powr<3>(2. * M_PI);

    const double integral = integrator.request(k, 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.).get();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }

  SECTION("Random polynomials")
  {
    constexpr uint take_n = 2;
    const auto x_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))), // x1
        GENERATE(take(1, random(-1., 1.))),      // x2
        GENERATE(take(1, random(-1., 1.))),      // x3
    });
    const auto y_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(take_n, random(-1., 1.))), // y1
        GENERATE(take(1, random(-1., 1.))),      // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });
    const auto z_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // y0
        GENERATE(take(take_n, random(-1., 1.))), // y1
        GENERATE(take(1, random(-1., 1.))),      // y2
        GENERATE(take(1, random(-1., 1.)))       // y3
    });

    const double k = GENERATE(take(take_n, random(0., 1.)));
    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(qx_min, qx_max) * y_poly.integral(qy_min, qy_max) *
                                                     z_poly.integral(qz_min, qz_max) / powr<3>(2. * M_PI);

    const double integral = integrator.get(k, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3], y_poly[0],
                                           y_poly[1], y_poly[2], y_poly[3], z_poly[0], z_poly[1], z_poly[2], z_poly[3]);

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}