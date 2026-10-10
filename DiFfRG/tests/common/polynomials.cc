#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/polynomials.hh>

using namespace DiFfRG;

TEST_CASE("Polynomial arithmetic", "[common][polynomial]")
{
  // p(x) = 1 + 2x + 3x^2
  Polynomial p({1., 2., 3.});
  REQUIRE(p(2.) == Catch::Approx(17.));
  REQUIRE(p.derivative()(2.) == Catch::Approx(14.));
  REQUIRE(p.integral(0., 1.) == Catch::Approx(3.));
  REQUIRE(p.square_argument()(2.) == Catch::Approx(1. + 8. + 48.));

  Polynomial q({0., 1.});
  q *= p;
  REQUIRE(q(2.) == Catch::Approx(34.));
}

TEST_CASE("The empty polynomial is the zero polynomial", "[common][polynomial]")
{
  const Polynomial zero(std::vector<double>{});
  REQUIRE(zero(3.) == 0.);
  REQUIRE(zero.derivative()(3.) == 0.);
  REQUIRE(zero.square_argument()(3.) == 0.);
  REQUIRE(Polynomial({5.}).derivative()(3.) == 0.);

  Polynomial p({1., 2.});
  p *= zero;
  REQUIRE(p(3.) == 0.);
}
