#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/quadrature/matsubara.hh>

using namespace DiFfRG;

TEST_CASE("Test matsubara quadrature rule", "[double][quadrature][matsubara]")
{
  SECTION("Simple test")
  {
    const double T = GENERATE(take(5, random(0.01, 1.0)));
    MatsubaraQuadrature<double> mq(T, 1.);

    const auto f = [&](const double x) { return 1. / (1. + powr<2>(x)); };

    const double reference = 0.5 * 1. / std::tanh(0.5 / T);
    const double sum = mq.sum(f);

    CHECK(sum == Catch::Approx(reference));
  }
  SECTION("Test with different parameters")
  {
    const double T = GENERATE(take(5, random(0.001, 0.1)));
    const double a = GENERATE(take(5, random(0.1, 1.0)));
    const double b = GENERATE(take(5, random(0.01, 0.1)));

    MatsubaraQuadrature<double> mq(T, a);

    const auto f = [&](const double x) { return 1. / (powr<2>(a) + b * x + powr<2>(x)); };

    const double reference =
        -(std::sinh(std::sqrt(4 * std::pow(a, 2) - std::pow(b, 2)) / (2. * T)) /
          (std::sqrt(4 * std::pow(a, 2) - std::pow(b, 2)) *
           (std::cos(b / (2. * T)) - std::cosh(std::sqrt(4 * std::pow(a, 2) - std::pow(b, 2)) / (2. * T)))));

    const double result = mq.sum(f);
    CHECK(result == Catch::Approx(reference));
  }
  SECTION("Test at T=0")
  {
    const double T = 0.;
    const double a = GENERATE(take(20, random(0.01, 10.0)));

    MatsubaraQuadrature<double> mq(T, a);

    const auto f = [&](const double x) { return 1. / (powr<2>(a) + powr<2>(x)); };

    const double reference = 1 / (2. * a);

    const double result = mq.sum(f);
    if (!(result == Catch::Approx(reference).margin(5e-6)))
      std::cout << "result: " << result << "| reference: " << reference << std::endl;
    CHECK(result == Catch::Approx(reference).margin(5e-6));
  }
}