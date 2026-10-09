#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/physics/integration/optimize.hh>
#include <DiFfRG/physics/regulators.hh>

#include <DiFfRG/physics/integration.hh>

#include <cmath>

using namespace DiFfRG;

TEST_CASE("Test optimization of x_extent", "[integration][quadrature integration][optimization]")
{
  json::value jv = {{"integration", {{"x_order", 32}, {"x_extent_tolerance", 1e-4}}}, {"output", {{"verbosity", 0}}}};
  using std::abs;

  SECTION("Litim")
  {
    const auto x_extent = DiFfRG::optimize_x_extent<DiFfRG::LitimRegulator<>>(jv);
    REQUIRE(abs(x_extent - 1.) / 1. < 1e-10);
  }

  SECTION("PolyExp")
  {
    const auto x_extent = DiFfRG::optimize_x_extent<DiFfRG::PolynomialExpRegulator<>>(jv);
    REQUIRE(abs(x_extent - 1.52) / 1.52 < 1e-2);
  }

  struct sth {
    using Regulator = DiFfRG::LitimRegulator<>;
  };

  DiFfRG::Init();

  QuadratureProvider quad;
  DiFfRG::Integrator_p2<4, double, sth, GPU_exec> integrator(quad, jv);
}
