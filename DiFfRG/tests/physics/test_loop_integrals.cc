#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/loop_integrals.hh>
#include <boost/math/special_functions/zeta.hpp>

#include <cmath>

using namespace DiFfRG;

constexpr double S_ds(int d) { return 2. * std::pow(M_PI, d / 2.) / std::tgamma(d / 2.); }

//--------------------------------------------
// Quadrature integration

TEMPLATE_TEST_CASE_SIG("Test momentum integrals", "[integration][quadrature integration]", ((int dim), dim), (1), (2),
                       (3), (4))
{
  const auto poly = Polynomial({
      dim == 1 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(dim, 0.);
  coeff_integrand[dim - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral = S_ds(dim) * square_poly.integral(0., q_extent) / powr<dim>(2. * M_PI);

  const QGauss<1> x_quadrature(64);
  double integral =
      LoopIntegrals::integrate<double, dim>([&](const auto &x) { return poly(x); }, x_quadrature, x_extent, k);
  if (!is_close(reference_integral, integral, 1e-6))
    std::cout << "reference: " << reference_integral << ", integral: " << integral
              << " relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-6));
}

TEMPLATE_TEST_CASE_SIG("Test momentum integrals with angle", "[integration][quadrature integration]", ((int dim), dim),
                       (2), (3), (4))
{
  const auto poly = Polynomial({
      dim == 1 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });
  const auto cos_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(2, random(-1., 1.))), // x2
      GENERATE(take(2, random(-1., 1.)))  // x3
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(dim, 0.);
  coeff_integrand[dim - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral =
      S_ds(dim) / 2. * square_poly.integral(0., q_extent) / powr<dim>(2. * M_PI) * cos_poly.integral(-1., 1.);

  const QGauss<1> x_quadrature(64);
  const QGauss<1> cos_quadrature(5);
  double integral = LoopIntegrals::angle_integrate<double, dim>(
      [&](const auto &x, const auto &cos) { return poly(x) * cos_poly(cos); }, x_quadrature, x_extent, k,
      cos_quadrature);
  if (!is_close(reference_integral, integral, 1e-6))
    std::cout << "reference: " << reference_integral << ", integral: " << integral
              << " relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-6));
}

TEMPLATE_TEST_CASE_SIG("Test momentum integrals with q0 integral",
                       "[integration][q0 integration][quadrature integration]", ((int dim), dim), (2), (3), (4))
{
  const auto poly = Polynomial({
      dim == 2 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });
  const auto q0_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(2, random(-1., 1.))), // x2
      GENERATE(take(2, random(-1., 1.)))  // x3
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const double q0_extent = GENERATE(take(2, random(1e-2, 1.)));

  constexpr int ds = dim - 1;

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(ds, 0.);
  coeff_integrand[ds - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral = S_ds(ds) * square_poly.integral(0., q_extent) / powr<ds>(2. * M_PI) *
                                    q0_poly.integral(-q0_extent, q0_extent) / powr<1>(2. * M_PI);

  const QGauss<1> x_quadrature(64);
  const QGauss<1> q0_quadrature(7);
  double integral = LoopIntegrals::spatial_integrate_and_integrate<double, dim>(
      [&](const auto &x, const auto &x0) { return poly(x) * q0_poly(x0); }, x_quadrature, x_extent, k, q0_quadrature,
      q0_extent);
  if (!is_close(reference_integral, integral, 1e-6))
    std::cout << "reference: " << reference_integral << ", integral: " << integral
              << " relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-6));
}

TEMPLATE_TEST_CASE_SIG("Test momentum integrals with q0 integral and angle",
                       "[integration][q0 angle integration][quadrature integration]", ((int dim), dim), (2), (3), (4))
{
  const auto poly = Polynomial({
      dim == 2 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });
  const auto cos_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(2, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.)))  // x3
  });
  const auto q0_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(2, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.)))  // x3
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const double q0_extent = GENERATE(take(2, random(1e-2, 1.)));

  constexpr int ds = dim - 1;

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(ds, 0.);
  coeff_integrand[ds - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral = S_ds(ds) / 2. * square_poly.integral(0., q_extent) / powr<ds>(2. * M_PI) *
                                    cos_poly.integral(-1., 1.) * q0_poly.integral(-q0_extent, q0_extent) /
                                    powr<1>(2. * M_PI);

  const QGauss<1> x_quadrature(64);
  const QGauss<1> q0_quadrature(7);
  const QGauss<1> cos_quadrature(5);
  double integral = LoopIntegrals::spatial_angle_integrate_and_integrate<double, dim>(
      [&](const auto &x, const auto &cos, const auto &x0) { return poly(x) * q0_poly(x0) * cos_poly(cos); },
      x_quadrature, x_extent, k, cos_quadrature, q0_quadrature, q0_extent);
  if (!is_close(reference_integral, integral, 1e-6))
    std::cout << "reference: " << reference_integral << ", integral: " << integral
              << " relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-6));
}

TEMPLATE_TEST_CASE_SIG("Test momentum integrals with matsubara sum",
                       "[integration][matsubara sum integration][quadrature integration]", ((int dim), dim), (2), (3),
                       (4))
{
  const auto poly = Polynomial({
      dim == 2 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const uint q0_order = GENERATE(take(2, random(30, 50)));
  const double q0_extent = 2. * M_PI * (q0_order + GENERATE(take(2, random(500., 700.))));
  const double zeta_val = GENERATE(take(4, random(3., 6.)));
  const double T_val = GENERATE(take(2, random(1e-2, 1.)));

  constexpr int ds = dim - 1;

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(ds, 0.);
  coeff_integrand[ds - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral = S_ds(ds) * square_poly.integral(0., q_extent) / powr<ds>(2. * M_PI) *
                                    (1. + 2. * boost::math::zeta(zeta_val)) * T_val;

  const QGauss<1> x_quadrature(64);
  const QGauss<1> q0_quadrature(32);
  double integral = LoopIntegrals::spatial_integrate_and_sum<double, dim>(
      [&](const auto &x, const auto &x0) {
        return is_close(x0, 0.) ? poly(x) : poly(x) / std::pow(std::abs(x0) / (2. * M_PI * T_val), zeta_val);
      },
      x_quadrature, x_extent, k, q0_order, q0_quadrature, q0_extent, T_val);
  if (!is_close(reference_integral, integral, 1e-5))
    std::cout << "reference: " << reference_integral << ", integral: " << integral
              << " relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-5));
}

TEMPLATE_TEST_CASE_SIG("Test momentum integrals with matsubara sum and angle",
                       "[integration][matsubara sum angle integration][quadrature integration]", ((int dim), dim), (2),
                       (3), (4))
{
  const auto poly = Polynomial({
      dim == 2 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });
  const auto cos_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(1, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.)))  // x3
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const uint q0_order = GENERATE(take(2, random(30, 50)));
  const double q0_extent = 2. * M_PI * (q0_order + GENERATE(take(1, random(500., 700.))));
  const double zeta_val = GENERATE(take(3, random(3., 6.)));
  const double T_val = GENERATE(take(2, random(1e-2, 1.)));

  constexpr int ds = dim - 1;

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(ds, 0.);
  coeff_integrand[ds - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral = S_ds(ds) / 2. * square_poly.integral(0., q_extent) / powr<ds>(2. * M_PI) *
                                    cos_poly.integral(-1., 1.) * (1. + 2. * boost::math::zeta(zeta_val)) * T_val;

  const QGauss<1> x_quadrature(64);
  const QGauss<1> q0_quadrature(32);
  const QGauss<1> cos_quadrature(5);
  double integral = LoopIntegrals::spatial_angle_integrate_and_sum<double, dim>(
      [&](const auto &x, const auto &cos, const auto &x0) {
        return poly(x) * cos_poly(cos) *
               (is_close(x0, 0.) ? 1. : 1. / std::pow(std::abs(x0) / (2. * M_PI * T_val), zeta_val));
      },
      x_quadrature, x_extent, k, cos_quadrature, q0_order, q0_quadrature, q0_extent, T_val);
  if (!is_close(reference_integral, integral, 1e-5))
    std::cout << "reference: " << reference_integral << ", integral: " << integral
              << " relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-5));
}

//-----------------------------------------
// adaptive
/*
TEMPLATE_TEST_CASE_SIG("Test adaptive momentum integrals", "[integration][adaptive integration]", ((int dim), dim), (1),
(2), (3), (4))
{
  const auto poly = Polynomial({
      dim == 1 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(2, random(-1., 1.))),                 // x5
      GENERATE(take(2, random(-1., 1.))),                 // x6
      GENERATE(take(2, random(-1., 1.)))                  // x6
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(dim, 0.);
  coeff_integrand[dim - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral = S_ds(dim) * square_poly.integral(0., q_extent) / powr<dim>(2. * M_PI);

  double integral = AdaptiveLoopIntegrals::integrate<double, dim>([&](const auto &x) { return poly(x); }, x_extent, k);
  if (!is_close(reference_integral, integral, 1e-6))
    std::cout << "reference: " << reference_integral << ", integral: " << integral << " relative error: " <<
std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-6));
}

TEMPLATE_TEST_CASE_SIG("Test adaptive momentum integrals with angle", "[integration][adaptive integration]", ((int dim),
dim), (2), (3), (4))
{
  const auto poly = Polynomial({
      dim == 1 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });
  const auto cos_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(2, random(-1., 1.))), // x2
      GENERATE(take(2, random(-1., 1.)))  // x3
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(dim, 0.);
  coeff_integrand[dim - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral = S_ds(dim) / 2. * square_poly.integral(0., q_extent) / powr<dim>(2. * M_PI) *
cos_poly.integral(-1., 1.);

  double integral = AdaptiveLoopIntegrals::angle_integrate<double, dim>([&](const auto &x, const auto &cos) { return
poly(x) * cos_poly(cos); }, x_extent, k); if (!is_close(reference_integral, integral, 1e-6)) std::cout << "reference: "
<< reference_integral << ", integral: " << integral << " relative error: " << std::abs(reference_integral - integral) /
std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-6));
}

TEMPLATE_TEST_CASE_SIG("Test adaptive momentum integrals with q0 integral", "[integration][q0 integration][adaptive
integration]", ((int dim), dim), (2), (3), (4))
{
  const auto poly = Polynomial({
      dim == 2 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });
  const auto q0_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(2, random(-1., 1.))), // x2
      GENERATE(take(2, random(-1., 1.)))  // x3
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const double q0_extent = GENERATE(take(2, random(1e-2, 1.)));

  constexpr int ds = dim - 1;

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(ds, 0.);
  coeff_integrand[ds - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral = S_ds(ds) * square_poly.integral(0., q_extent) / powr<ds>(2. * M_PI) *
q0_poly.integral(-q0_extent, q0_extent) / powr<1>(2. * M_PI);

  double integral =
      AdaptiveLoopIntegrals::spatial_integrate_and_integrate<double, dim>([&](const auto &x, const auto &x0) { return
poly(x) * q0_poly(x0); }, x_extent, k, q0_extent); if (!is_close(reference_integral, integral, 1e-6)) std::cout <<
"reference: " << reference_integral << ", integral: " << integral << " relative error: " << std::abs(reference_integral
- integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-6));
}

TEMPLATE_TEST_CASE_SIG("Test adaptive momentum integrals with q0 integral and angle", "[integration][q0 angle
integration][adaptive integration]", ((int dim), dim), (2), (3), (4))
{
  const auto poly = Polynomial({
      dim == 2 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });
  const auto cos_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(2, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.)))  // x3
  });
  const auto q0_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(2, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.)))  // x3
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const double q0_extent = GENERATE(take(2, random(1e-2, 1.)));

  constexpr int ds = dim - 1;

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(ds, 0.);
  coeff_integrand[ds - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral =
      S_ds(ds) / 2. * square_poly.integral(0., q_extent) / powr<ds>(2. * M_PI) * cos_poly.integral(-1., 1.) *
q0_poly.integral(-q0_extent, q0_extent) / powr<1>(2. * M_PI);

  double integral = AdaptiveLoopIntegrals::spatial_angle_integrate_and_integrate<double, dim>(
      [&](const auto &x, const auto &cos, const auto &x0) { return poly(x) * q0_poly(x0) * cos_poly(cos); }, x_extent,
k, q0_extent); if (!is_close(reference_integral, integral, 1e-6)) std::cout << "reference: " << reference_integral << ",
integral: " << integral << " relative error: " << std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-6));
}

TEMPLATE_TEST_CASE_SIG("Test adaptive momentum integrals with matsubara sum", "[integration][matsubara sum
integration][adaptive integration]", ((int dim), dim), (2), (3), (4))
{
  const auto poly = Polynomial({
      dim == 2 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const uint q0_order = GENERATE(take(2, random(30, 50)));
  const double q0_extent = 2. * M_PI * (q0_order + GENERATE(take(2, random(500., 700.))));
  const double zeta_val = GENERATE(take(4, random(3., 6.)));
  const double T_val = GENERATE(take(2, random(1e-2, 1.)));

  constexpr int ds = dim - 1;

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(ds, 0.);
  coeff_integrand[ds - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral = S_ds(ds) * square_poly.integral(0., q_extent) / powr<ds>(2. * M_PI) * (1. + 2. *
std::riemann_zeta(zeta_val)) * T_val;

  double integral = AdaptiveLoopIntegrals::spatial_integrate_and_sum<double, dim>(
      [&](const auto &x, const auto &x0) { return is_close(x0, 0.) ? poly(x) : poly(x) / std::pow(std::abs(x0) / (2. *
M_PI * T_val), zeta_val); }, x_extent, k, q0_order, q0_extent, T_val); if (!is_close(reference_integral, integral,
1e-5)) std::cout << "reference: " << reference_integral << ", integral: " << integral << " relative error: " <<
std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-5));
}
TEMPLATE_TEST_CASE_SIG("Test adaptive momentum integrals with matsubara sum and angle", "[integration][matsubara sum
angle integration][adaptive integration]", ((int dim), dim), (2), (3), (4))
{
  const auto poly = Polynomial({
      dim == 2 ? 0. : GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))),                 // x1
      GENERATE(take(2, random(-1., 1.))),                 // x2
      GENERATE(take(2, random(-1., 1.))),                 // x3
      GENERATE(take(2, random(-1., 1.))),                 // x4
      GENERATE(take(1, random(-1., 1.))),                 // x5
      GENERATE(take(1, random(-1., 1.))),                 // x6
      GENERATE(take(1, random(-1., 1.)))                  // x6
  });
  const auto cos_poly = Polynomial({
      GENERATE(take(2, random(-1., 1.))), // x0
      GENERATE(take(2, random(-1., 1.))), // x1
      GENERATE(take(1, random(-1., 1.))), // x2
      GENERATE(take(1, random(-1., 1.)))  // x3
  });

  const double k = GENERATE(take(2, random(0., 1.)));
  const double x_extent = GENERATE(take(2, random(1., 2.)));
  const uint q0_order = GENERATE(take(2, random(30, 50)));
  const double q0_extent = 2. * M_PI * (q0_order + GENERATE(take(1, random(500., 700.))));
  const double zeta_val = GENERATE(take(3, random(3., 6.)));
  const double T_val = GENERATE(take(2, random(1e-2, 1.)));

  constexpr int ds = dim - 1;

  const double q_extent = std::sqrt(x_extent * powr<2>(k));
  auto square_poly = poly.square_argument();
  std::vector<double> coeff_integrand(ds, 0.);
  coeff_integrand[ds - 1] = 1.;
  square_poly *= Polynomial(coeff_integrand);
  const double reference_integral =
      S_ds(ds) / 2. * square_poly.integral(0., q_extent) / powr<ds>(2. * M_PI) * cos_poly.integral(-1., 1.) * (1. + 2. *
std::riemann_zeta(zeta_val)) * T_val;

  double integral = AdaptiveLoopIntegrals::spatial_angle_integrate_and_sum<double, dim>(
      [&](const auto &x, const auto &cos, const auto &x0) {
        return poly(x) * cos_poly(cos) * (is_close(x0, 0.) ? 1. : 1. / std::pow(std::abs(x0) / (2. * M_PI * T_val),
zeta_val));
      },
      x_extent, k, q0_order, q0_extent, T_val);
  if (!is_close(reference_integral, integral, 1e-5))
    std::cout << "reference: " << reference_integral << ", integral: " << integral << " relative error: " <<
std::abs(reference_integral - integral) / std::abs(reference_integral)
              << std::endl;
  CHECK(is_close(reference_integral, integral, 1e-5));
}*/