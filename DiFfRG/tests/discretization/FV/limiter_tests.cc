#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>

#include <autodiff/forward/real.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using DiFfRG::def::MinModLimiter;
using DiFfRG::def::limiter_utils::sgn;

// ──────────────────────────────────────────────────────────────────────
// Concept checks
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("MinModLimiter satisfies HasSlopeLimiter concept", "[FV][limiter]")
{
  static_assert(DiFfRG::def::HasSlopeLimiter<MinModLimiter>);
}

// ──────────────────────────────────────────────────────────────────────
// Basic MinMod slope_limit behaviour
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("MinMod slope_limit - same-sign slopes", "[FV][limiter]")
{
  SECTION("Both positive - returns smaller")
  {
    CHECK(MinModLimiter::slope_limit(2.0, 5.0) == Catch::Approx(2.0));
    CHECK(MinModLimiter::slope_limit(5.0, 2.0) == Catch::Approx(2.0));
  }

  SECTION("Both negative - returns smaller magnitude (more negative)")
  {
    CHECK(MinModLimiter::slope_limit(-3.0, -1.0) == Catch::Approx(-1.0));
    CHECK(MinModLimiter::slope_limit(-1.0, -3.0) == Catch::Approx(-1.0));
  }

  SECTION("Equal slopes")
  {
    CHECK(MinModLimiter::slope_limit(4.0, 4.0) == Catch::Approx(4.0));
    CHECK(MinModLimiter::slope_limit(-2.5, -2.5) == Catch::Approx(-2.5));
  }
}

TEST_CASE("MinMod slope_limit - opposite-sign and zero slopes", "[FV][limiter]")
{
  SECTION("Opposite signs - returns zero")
  {
    CHECK(MinModLimiter::slope_limit(1.0, -1.0) == Catch::Approx(0.0));
    CHECK(MinModLimiter::slope_limit(-2.0, 3.0) == Catch::Approx(0.0));
  }

  SECTION("One slope zero - returns zero")
  {
    CHECK(MinModLimiter::slope_limit(0.0, 5.0) == Catch::Approx(0.0));
    CHECK(MinModLimiter::slope_limit(5.0, 0.0) == Catch::Approx(0.0));
    CHECK(MinModLimiter::slope_limit(-3.0, 0.0) == Catch::Approx(0.0));
    CHECK(MinModLimiter::slope_limit(0.0, -3.0) == Catch::Approx(0.0));
  }

  SECTION("Both zero - returns zero") { CHECK(MinModLimiter::slope_limit(0.0, 0.0) == Catch::Approx(0.0)); }
}

TEST_CASE("MinMod slope_limit - symmetry", "[FV][limiter]")
{
  // slope_limit(a,b) == slope_limit(b,a) when both have the same sign
  const double a = 1.7, b = 4.3;
  CHECK(MinModLimiter::slope_limit(a, b) == Catch::Approx(MinModLimiter::slope_limit(b, a)));

  const double c = -0.9, d = -6.1;
  CHECK(MinModLimiter::slope_limit(c, d) == Catch::Approx(MinModLimiter::slope_limit(d, c)));
}

// ──────────────────────────────────────────────────────────────────────
// Signum helper
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("limiter_utils::sgn", "[FV][limiter]")
{
  CHECK(sgn(3.0) == 1);
  CHECK(sgn(-2.5) == -1);
  CHECK(sgn(0.0) == 0);
  CHECK(sgn(1e-15) == 1);
  CHECK(sgn(-1e-15) == -1);
}

// ──────────────────────────────────────────────────────────────────────
// Autodiff derivative propagation through minmod
// (moved and adapted from KurganovTadmor_tests.cc)
// ──────────────────────────────────────────────────────────────────────

TEST_CASE("Minmod autodiff - derivative propagation", "[FV][limiter][autodiff]")
{
  using AD = autodiff::real;

  // minmod(a, b) := 0.5 * [sgn(a) + sgn(b)] * min(|a|, |b|)
  auto minmod = [](AD a, AD b) -> AD {
    using std::abs;
    using std::min;
    return 0.5 * (sgn(a) + sgn(b)) * min(abs(a), abs(b));
  };

  // 3-point stencil (x_{j-1}, x_j, x_{j+1}) with cell values
  // (u_{j-1}, u_j, u_{j+1}).  Minmod-limited slope at cell j:
  //
  //   slope_j = minmod( (u_j - u_{j-1}) / dx_L ,
  //                     (u_{j+1} - u_j) / dx_R )

  const double x_jm1 = 0.0;
  const double x_j = 1.0;
  const double x_jp1 = 2.0;
  const double dx_L = x_j - x_jm1; // = 1
  const double dx_R = x_jp1 - x_j; // = 1

  SECTION("Derivative w.r.t. u_{j+1}: right slope is smaller -> d/d(u_{j+1}) = 1/dx_R")
  {
    AD u_jm1 = 0.0;
    AD u_j_val = 2.0;
    AD u_jp1 = 2.5;
    autodiff::seed<1>(u_jp1, 1.0);

    AD slope_L = (u_j_val - u_jm1) / dx_L;
    AD slope_R = (u_jp1 - u_j_val) / dx_R;
    AD result = minmod(slope_L, slope_R);

    CHECK(result.val() == Catch::Approx(0.5));
    CHECK(result[1] == Catch::Approx(1.0 / dx_R));
  }

  SECTION("Derivative w.r.t. u_{j+1}: right slope is larger -> d/d(u_{j+1}) = 0")
  {
    AD u_jm1 = 1.0;
    AD u_j_val = 2.0;
    AD u_jp1 = 5.0;
    autodiff::seed<1>(u_jp1, 1.0);

    AD slope_L = (u_j_val - u_jm1) / dx_L;
    AD slope_R = (u_jp1 - u_j_val) / dx_R;
    AD result = minmod(slope_L, slope_R);

    CHECK(result.val() == Catch::Approx(1.0));
    CHECK(result[1] == Catch::Approx(0.0));
  }

  SECTION("Derivative w.r.t. u_j: both slopes depend on u_j, smaller is selected")
  {
    AD u_jm1 = 0.0;
    AD u_j_val = 2.0;
    autodiff::seed<1>(u_j_val, 1.0);
    AD u_jp1 = 6.0;

    AD slope_L = (u_j_val - u_jm1) / dx_L;
    AD slope_R = (u_jp1 - u_j_val) / dx_R;
    AD result = minmod(slope_L, slope_R);

    CHECK(result.val() == Catch::Approx(2.0));
    CHECK(result[1] == Catch::Approx(1.0 / dx_L));
  }

  SECTION("Derivative w.r.t. u_{j-1}: left slope is smaller -> d/d(u_{j-1}) = -1/dx_L")
  {
    AD u_jm1 = 1.5;
    autodiff::seed<1>(u_jm1, 1.0);
    AD u_j_val = 2.0;
    AD u_jp1 = 5.0;

    AD slope_L = (u_j_val - u_jm1) / dx_L;
    AD slope_R = (u_jp1 - u_j_val) / dx_R;
    AD result = minmod(slope_L, slope_R);

    CHECK(result.val() == Catch::Approx(0.5));
    CHECK(result[1] == Catch::Approx(-1.0 / dx_L));
  }

  SECTION("Opposite-sign slopes -> minmod = 0, all derivatives vanish")
  {
    AD u_jm1 = 3.0;
    AD u_j_val = 2.0;
    AD u_jp1 = 3.0;
    autodiff::seed<1>(u_jp1, 1.0);

    AD slope_L = (u_j_val - u_jm1) / dx_L;
    AD slope_R = (u_jp1 - u_j_val) / dx_R;
    AD result = minmod(slope_L, slope_R);

    CHECK(result.val() == Catch::Approx(0.0));
    CHECK(result[1] == Catch::Approx(0.0));
  }

  SECTION("Both negative slopes, seed u_{j+1}")
  {
    AD u_jm1 = 5.0;
    AD u_j_val = 2.0;
    AD u_jp1 = 1.0;
    autodiff::seed<1>(u_jp1, 1.0);

    AD slope_L = (u_j_val - u_jm1) / dx_L;
    AD slope_R = (u_jp1 - u_j_val) / dx_R;
    AD result = minmod(slope_L, slope_R);

    CHECK(result.val() == Catch::Approx(-1.0));
    CHECK(result[1] == Catch::Approx(1.0 / dx_R));
  }
}
