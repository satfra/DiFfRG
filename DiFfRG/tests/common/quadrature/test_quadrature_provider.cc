#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/quadrature/quadrature_provider.hh>

using namespace DiFfRG;

TEST_CASE("Test quadrature provider", "[double][quadrature]")
{
  const double T = GENERATE(take(5, random(0.1, 1.0)));
  const double start_k = GENERATE(take(5, random(1.0, 100.0)));
  const double speed_k = GENERATE(take(5, random(1e-5, 1e-2)));
  const int steps = GENERATE(take(5, random(1e+1, 1e+4)));

  const size_t order = GENERATE(take(2, random(8, 64)));

  DiFfRG::QuadratureProvider quadrature_provider;

  SECTION("Bare test")
  {
    double point = 0;
    double weight = 0;
    for (int j = 0; j < steps; ++j) {
      const double k = std::exp((long double)(-j * speed_k)) * start_k;

      const auto &points = quadrature_provider.get_points<double>(order);
      const auto &matsubara_points = quadrature_provider.get_matsubara_points<double>(T, k);
      const auto &weights = quadrature_provider.get_weights<double>(order);
      const auto &matsubara_weights = quadrature_provider.get_matsubara_weights<double>(T, k);

      for (uint i = 0; i < order; ++i) {
        point += points[i] / (double)steps;
        weight += weights[i] / (double)steps;
        if (!isfinite(points[i]) || !isfinite(weights[i]))
          throw std::runtime_error("Not finite: Quadrature points or weights");
      }
      for (uint i = 0; i < matsubara_points.size(); ++i) {
        point += matsubara_points[i] / (double)steps;
        weight += matsubara_weights[i] / (double)steps;

        if (!isfinite(matsubara_points[i]) || !isfinite(matsubara_weights[i])) {
          std::string error =
              "Not finite: Matsubara points or weights for k = " + std::to_string(k) + " and T = " + std::to_string(T);
          error += "\nMatsubara size is " + std::to_string(matsubara_points.size());
          throw std::runtime_error(error);
        }
      }
    }

    REQUIRE(point != Catch::Approx(0.));
    REQUIRE(weight != Catch::Approx(0.));
  }

  SECTION("Test with integration") {}
}
