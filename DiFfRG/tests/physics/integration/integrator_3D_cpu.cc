#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/physics/integration/integrator_3D_cpu.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto
  kernel(const double q, const double cos, const double phi, const double /*k*/, const double /*c*/, const double x0,
         const double x1, const double x2, const double x3, const double x4, const double x5, const double cos_x0,
         const double cos_x1, const double cos_x2, const double cos_x3, const double phi_x0, const double phi_x1,
         const double phi_x2, const double phi_x3)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q) + x4 * powr<4>(q) + x5 * powr<5>(q)) *
           (cos_x0 + cos_x1 * powr<1>(cos) + cos_x2 * powr<2>(cos) + cos_x3 * powr<3>(cos)) *
           (phi_x0 + phi_x1 * powr<1>(phi) + phi_x2 * powr<2>(phi) + phi_x3 * powr<3>(phi));
  }

  static __forceinline__ __host__ __device__ auto
  constant(const double /*k*/, const double c, const double /*x0*/, const double /*x1*/, const double /*x2*/,
           const double /*x3*/, const double /*x4*/, const double /*x5*/, const double /*cos_x0*/,
           const double /*cos_x1*/, const double /*cos_x2*/, const double /*cos_x3*/, const double /*phi_x0*/,
           const double /*phi_x1*/, const double /*phi_x2*/, const double /*phi_x3*/)
  {
    return c;
  }
};

TEST_CASE("Test 3D cpu momentum integrals", "[3D integration][quadrature integration]")
{
  constexpr int dim = 3;

  const double x_extent = GENERATE(take(1, random(1., 2.)));
  QuadratureProvider quadrature_provider;
  Integrator3DTBB<double, PolyIntegrand> integrator(quadrature_provider, {{64, 16, 16}}, x_extent);

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double reference_integral = V_d(dim, q_extent) / powr<dim>(2. * M_PI);

    const double integral = integrator.request(k, 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.).get();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "dim: " << dim << "| reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }

  SECTION("Random polynomials")
  {
    constexpr uint take_n = 2;

    const auto poly = Polynomial({
        dim == 1 ? 0. : GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))),                 // x1
        GENERATE(take(take_n, random(-1., 1.))),                 // x2
        GENERATE(take(1, random(-1., 1.))),                      // x3
        GENERATE(take(1, random(-1., 1.))),                      // x4
        GENERATE(take(1, random(-1., 1.))),                      // x5
    });
    const auto cos_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))), // x1
        GENERATE(take(1, random(-1., 1.))),      // x2
        GENERATE(take(1, random(-1., 1.)))       // x3
    });
    const auto phi_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))), // x1
        GENERATE(take(1, random(-1., 1.))),      // x2
        GENERATE(take(1, random(-1., 1.)))       // x3
    });

    const double k = GENERATE(take(take_n, random(0., 1.)));
    const double q_extent = std::sqrt(x_extent * powr<2>(k));
    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    auto int_poly = poly;
    std::vector<double> coeff_integrand(dim, 0.);
    coeff_integrand[dim - 1] = 1.;
    int_poly *= Polynomial(coeff_integrand);
    const double reference_integral = constant + int_poly.integral(0., q_extent) * cos_poly.integral(-1., 1.) *
                                                     phi_poly.integral(0., 2. * M_PI) / powr<dim>(2. * M_PI);

    const double integral =
        integrator
            .request(k, constant, poly[0], poly[1], poly[2], poly[3], poly[4], poly[5], cos_poly[0], cos_poly[1],
                     cos_poly[2], cos_poly[3], phi_poly[0], phi_poly[1], phi_poly[2], phi_poly[3])
            .get();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "dim: " << dim << "| reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}