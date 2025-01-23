#include "autodiff/forward/real/real.hpp"
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/integrator_1D_cartesian_gpu.hh>
#include <DiFfRG/physics/integration/quadrature_provider.hh>

using namespace DiFfRG;

//--------------------------------------------
// Quadrature integration

class PolyIntegrand
{
public:
  static __forceinline__ __host__ __device__ auto kernel(const double q, const double /*k*/, const double /*c*/,
                                                         const double x0, const double x1, const double x2,
                                                         const double x3)
  {
    return (x0 + x1 * powr<1>(q) + x2 * powr<2>(q) + x3 * powr<3>(q));
  }

  static __forceinline__ __host__ __device__ auto constant(const double /*k*/, const double c, const double /*x0*/,
                                                           const double /*x1*/, const double /*x2*/,
                                                           const double /*x3*/)
  {
    return c;
  }
};

TEST_CASE("Test 2d cartesian gpu momentum integrals", "[double][gpu][integration][quadrature]")
{
  const double qx_min = GENERATE(take(2, random(-2., -1.)));
  const double qx_max = GENERATE(take(2, random(1., 2.)));

  QuadratureProvider quadrature_provider;
  Integrator1DCartesianGPU<double, PolyIntegrand> integrator(quadrature_provider, {{32}}, 0., 256, qx_min, qx_max);

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double reference_integral = (qx_max - qx_min) / powr<1>(2. * M_PI);

    const double integral = integrator.request(k, 0., 1., 0., 0., 0.).get();

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
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
        GENERATE(take(take_n, random(-1., 1.))), // x2
        GENERATE(take(take_n, random(-1., 1.))), // x3
    });

    const double k = GENERATE(take(take_n, random(0., 1.)));
    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(qx_min, qx_max) / powr<1>(2. * M_PI);

    const double integral = integrator.get(k, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3]);

    if (!is_close(reference_integral, integral, 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, integral, 1e-6));
  }
}

TEST_CASE("Test 2d cartesian gpu momentum integrals (complex)", "[complex][gpu][integration][quadrature]")
{
  const double qx_min = GENERATE(take(2, random(-2., -1.)));
  const double qx_max = GENERATE(take(2, random(1., 2.)));

  QuadratureProvider quadrature_provider;
  Integrator1DCartesianGPU<complex<double>, PolyIntegrand> integrator(quadrature_provider, {{32}}, 0., 256, qx_min,
                                                                      qx_max);

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double reference_integral = (qx_max - qx_min) / powr<1>(2. * M_PI);

    const auto integral = integrator.request(k, 0., 1., 0., 0., 0.).get();

    if (!is_close(reference_integral, real(integral), 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / abs(reference_integral) << std::endl;
    }
    CHECK(is_close(reference_integral, real(integral), 1e-6));
    CHECK(is_close(0., imag(integral), 1e-6));
  }

  SECTION("Random polynomials")
  {
    constexpr uint take_n = 2;
    const auto x_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))), // x1
        GENERATE(take(take_n, random(-1., 1.))), // x2
        GENERATE(take(take_n, random(-1., 1.))), // x3
    });

    const double k = GENERATE(take(take_n, random(0., 1.)));
    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(qx_min, qx_max) / powr<1>(2. * M_PI);

    const auto integral = integrator.get(k, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3]);

    if (!is_close(reference_integral, real(integral), 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, real(integral), 1e-6));
    CHECK(is_close(0., imag(integral), 1e-6));
  }
}

TEST_CASE("Test 2d cartesian gpu momentum integrals (autodiff)", "[autodiff][gpu][integration][quadrature]")
{
  const double qx_min = GENERATE(take(2, random(-2., -1.)));
  const double qx_max = GENERATE(take(2, random(1., 2.)));

  QuadratureProvider quadrature_provider;
  Integrator1DCartesianGPU<autodiff::real, PolyIntegrand> integrator(quadrature_provider, {{32}}, 0., 256, qx_min,
                                                                     qx_max);

  using autodiff::val, autodiff::derivative;

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double reference_integral = (qx_max - qx_min) / powr<1>(2. * M_PI);

    const auto integral = integrator.request(k, 0., 1., 0., 0., 0.).get();

    if (!is_close(reference_integral, val(integral), 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, val(integral), 1e-6));
    CHECK(is_close(0., derivative(integral), 1e-6));
  }

  SECTION("Random polynomials")
  {
    constexpr uint take_n = 2;
    const auto x_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))), // x1
        GENERATE(take(take_n, random(-1., 1.))), // x2
        GENERATE(take(take_n, random(-1., 1.))), // x3
    });

    const double k = GENERATE(take(take_n, random(0., 1.)));
    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(qx_min, qx_max) / powr<1>(2. * M_PI);

    const auto integral = integrator.get(k, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3]);

    if (!is_close(reference_integral, val(integral), 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / abs(reference_integral) << std::endl;
    }
    CHECK(is_close(reference_integral, val(integral), 1e-6));
    CHECK(is_close(0., derivative(integral), 1e-6));
  }
}

TEST_CASE("Test 2d cartesian gpu momentum integrals (complex autodiff)",
          "[complex][autodiff][gpu][integration][quadrature]")
{
  const double qx_min = GENERATE(take(2, random(-2., -1.)));
  const double qx_max = GENERATE(take(2, random(1., 2.)));

  QuadratureProvider quadrature_provider;
  Integrator1DCartesianGPU<cxReal, PolyIntegrand> integrator(quadrature_provider, {{32}}, 0., 256, qx_min, qx_max);

  SECTION("Volume integral")
  {
    const double k = GENERATE(take(1, random(0., 1.)));
    const double reference_integral = (qx_max - qx_min) / powr<1>(2. * M_PI);

    const complex<double> integral = autodiff::val(integrator.request(k, 0., 1., 0., 0., 0.).get());

    if (!is_close(reference_integral, real(integral), 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / std::abs(reference_integral)
                << std::endl;
    }
    CHECK(is_close(reference_integral, real(integral), 1e-6));
    CHECK(is_close(0., imag(integral), 1e-6));
  }

  SECTION("Random polynomials")
  {
    constexpr uint take_n = 2;
    const auto x_poly = Polynomial({
        GENERATE(take(take_n, random(-1., 1.))), // x0
        GENERATE(take(take_n, random(-1., 1.))), // x1
        GENERATE(take(take_n, random(-1., 1.))), // x2
        GENERATE(take(take_n, random(-1., 1.))), // x3
    });

    const double k = GENERATE(take(take_n, random(0., 1.)));
    const double constant = GENERATE(take(take_n, random(-1., 1.)));

    const double reference_integral = constant + x_poly.integral(qx_min, qx_max) / powr<1>(2. * M_PI);

    const complex<double> integral =
        autodiff::val(integrator.get(k, constant, x_poly[0], x_poly[1], x_poly[2], x_poly[3]));

    if (!is_close(reference_integral, real(integral), 1e-6)) {
      std::cerr << "reference: " << reference_integral << "| integral: " << integral
                << "| relative error: " << abs(reference_integral - integral) / abs(reference_integral) << std::endl;
    }
    CHECK(is_close(reference_integral, real(integral), 2e-6));
    CHECK(is_close(0., imag(integral), 1e-6));
  }
}