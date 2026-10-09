#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

// deliberately included first and alone, so that this TU also checks that the header is
// self-contained
#include <DiFfRG/common/root_finding.hh>

#include <cmath>
#include <stdexcept>

using namespace DiFfRG;

namespace
{
  // The searches below bracket the root at 1 in [0, 3]. The first bisection point is 1.5, which
  // keeps successes and failures alternating; a bracket whose midpoint sits exactly on the root
  // would produce a single success and then fail forever.
  constexpr double root = 1.;
  constexpr double lower = 0.;
  constexpr double upper = 3.;
} // namespace

TEST_CASE("BisectionRootFinder converges the search variable", "[common][root_finding]")
{
  const double tol = 1e-6;

  BisectionRootFinder search([](const double x) -> bool { return x >= root; }, tol, 100);
  search.set_bounds(lower, upper);

  const double x = search.search();

  CHECK(x >= root);
  CHECK(std::abs(x - root) < tol);
}

TEST_CASE("BisectionRootFinderTarget converges the target value", "[common][root_finding]")
{
  const double tol = 1e-6;

  uint calls = 0;
  BisectionRootFinderTarget search(
      [&](const double x, double &target) -> bool {
        // the counter must name the step currently being taken
        CHECK(search.get_iter() == calls);
        ++calls;

        if (x < root) return false; // failures leave the target untouched
        target = x * x;
        return true;
      },
      tol, 100);
  search.set_bounds(lower, upper);

  const double x = search.search();

  CHECK(search.converged());
  CHECK(x >= root);
  CHECK(search.get_target() == Catch::Approx(x * x));
  // |t_k - t_{k-1}| = |x_k^2 - x_{k-1}^2| ~ 2 |x_k - x_{k-1}|, so x is resolved to about tol/2
  CHECK(std::abs(x - root) < tol);
}

TEST_CASE("BisectionRootFinderTarget stops on a flat target", "[common][root_finding]")
{
  BisectionRootFinderTarget search(
      [](const double x, double &target) -> bool {
        target = 42.;
        return x >= root;
      },
      1e-6, 100);
  search.set_bounds(lower, upper);

  const double x = search.search();

  // an x-insensitive target is "converged" as soon as two successes have been seen
  CHECK(search.converged());
  CHECK(search.get_target() == Catch::Approx(42.));
  CHECK(x >= root);
  CHECK(search.get_iter() <= 4);
}

TEST_CASE("BisectionRootFinderTarget reports stagnation instead of throwing", "[common][root_finding]")
{
  // an unreachable tolerance: |t_k - t_{k-1}| < 0 is never true
  BisectionRootFinderTarget search(
      [](const double x, double &target) -> bool {
        target = x * x;
        return x >= root;
      },
      0., 200);
  search.set_bounds(lower, upper);

  double x = 0.;
  REQUIRE_NOTHROW(x = search.search());

  CHECK_FALSE(search.converged());
  CHECK(x >= root);
  CHECK(std::abs(x - root) < 1e-12); // the bracket collapsed all the way down
  CHECK(search.get_target() == Catch::Approx(x * x));
}

TEST_CASE("BisectionRootFinderTarget honours a custom collapse tolerance", "[common][root_finding]")
{
  BisectionRootFinderTarget search(
      [](const double x, double &target) -> bool {
        target = x * x;
        return x >= root;
      },
      0., 200);
  search.set_bounds(lower, upper);
  search.set_x_collapse_tol(1e-3);

  const double x = search.search();

  CHECK_FALSE(search.converged());
  CHECK(x >= root);
  CHECK(std::abs(x - root) < 1e-2);
  CHECK(search.get_iter() < 30);
}

TEST_CASE("BisectionRootFinderTarget throws without a successful evaluation", "[common][root_finding]")
{
  BisectionRootFinderTarget search([](const double, double &) -> bool { return false; }, 1e-6, 100);
  search.set_bounds(lower, upper);

  CHECK_THROWS_AS(search.search(), std::runtime_error);
}

TEST_CASE("BisectionRootFinderTarget throws without bounds", "[common][root_finding]")
{
  BisectionRootFinderTarget search(
      [](const double x, double &target) -> bool {
        target = x;
        return x >= root;
      },
      1e-6, 100);

  CHECK_THROWS_AS(search.search(), std::runtime_error);
}

// ============================================================================================
// ScalingRootFinder
// ============================================================================================

namespace
{
  // Analytic oracle mimicking the fRG separatrix this finder was built for: an observable with
  // a critical power law above x_c, and flows below x_c that run away at an RG time diverging
  // logarithmically as x_c is approached.
  constexpr double sc_x_c = 0.3;
  constexpr double sc_C = 2.7;
  constexpr double sc_beta = 0.588;
  constexpr double sc_theta = 2.0;
  constexpr double sc_target = 0.01;

  double sc_obs(const double x) { return sc_C * std::pow(x - sc_x_c, sc_beta); }
  double sc_residual(const double x) { return 3.0 - std::log(sc_x_c - x) / sc_theta; }
  double sc_x_star() { return sc_x_c + std::pow(sc_target / sc_C, 1. / sc_beta); }

  /// Full oracle: convergent above x_c, divergent below.
  bool sc_probe(const double x, double &obs, double &residual)
  {
    if (x <= sc_x_c) {
      residual = sc_residual(x);
      return false;
    }
    obs = sc_obs(x);
    return true;
  }
} // namespace

TEST_CASE("solve_scaling_shift recovers the shift exactly", "[common][root_finding]")
{
  const auto shift_of = [](const double s) {
    const double d2 = 1., d1 = 3., beta = 0.7; // y_i = (s + d_i)^beta with d3 = 0
    const double y1 = std::pow(s + d1, beta), y2 = std::pow(s + d2, beta), y3 = std::pow(s, beta);
    return DiFfRG::detail::solve_scaling_shift(std::log(y1 / y2), std::log(y2 / y3), d1, d2);
  };

  SECTION("exact wherever the shift is determinable")
  {
    // d2 is fixed at 1, so this sweeps s from 1e3 down to 1e-17: 21 decades, covering the whole
    // regime the finder actually works in (the closest probe nearer to x_c than the probe
    // spacing). This is also the range the log1p branch exists to protect -- the naive
    // log((s+d2)/s) loses the first term entirely once s >> d2.
    for (int e = -3; e <= 17; ++e) {
      const double s = std::pow(10., -e);
      const double got = shift_of(s);
      CAPTURE(e, s, got);
      REQUIRE(std::isfinite(got));
      CHECK(std::abs(got - s) <= 1e-6 * s);
    }
  }

  SECTION("rejected, not mis-fitted, once the points sit far from criticality")
  {
    // s >> d2 means all three points are clustered far from x_c, where the power law is locally
    // a straight line and the defining difference cancels to noise. The solver must say so.
    // The call stays inline inside the CHECK_FALSE. Hoisting it into a named double changes
    // codegen enough under -ffast-math to make e == 4 pass on its own, which would leave this
    // section green while the guard it exists to check was reverted.
    for (int e = 4; e <= 18; ++e) {
      CAPTURE(e);
      CHECK_FALSE(std::isfinite(shift_of(std::pow(10., e))));
    }
  }
}

TEST_CASE("fit_power_law_3 recovers the critical point", "[common][root_finding]")
{
  const std::array<double, 3> xs{{0.9, 0.5, 0.31}};
  std::array<double, 3> ys{};
  for (std::size_t i = 0; i < 3; ++i)
    ys[i] = sc_obs(xs[i]);

  const auto fit = DiFfRG::detail::fit_power_law_3(xs, ys);
  REQUIRE(fit.valid);
  const double x_c = 0.31 - fit.shift;
  CHECK(std::abs(x_c - sc_x_c) < 1e-10);
  CHECK(fit.exponent == Catch::Approx(sc_beta).epsilon(1e-8));
  CHECK(std::exp(fit.log_amplitude) == Catch::Approx(sc_C).epsilon(1e-8));
}

TEST_CASE("fit_power_law_3 rejects data that refutes the power law", "[common][root_finding]")
{
  SECTION("log-linear in x admits no positive shift")
  {
    // y = exp(-k x) is the exponential, not a power law: A*d2 == B*(d1-d2) exactly, so the
    // asymptotic sign test finds no root rather than returning a huge spurious shift.
    const std::array<double, 3> xs{{3., 2., 1.}};
    std::array<double, 3> ys{};
    for (std::size_t i = 0; i < 3; ++i)
      ys[i] = std::exp(-0.5 * xs[i]);
    CHECK_FALSE(DiFfRG::detail::fit_power_law_3(xs, ys).valid);
  }

  SECTION("convex the wrong way")
  {
    const std::array<double, 3> xs{{3., 2., 1.}};
    const std::array<double, 3> ys{{100., 2., 1.}};
    CHECK_FALSE(DiFfRG::detail::fit_power_law_3(xs, ys).valid);
  }

  SECTION("non-monotone")
  {
    const std::array<double, 3> xs{{3., 2., 1.}};
    const std::array<double, 3> ys{{2., 5., 1.}};
    CHECK_FALSE(DiFfRG::detail::fit_power_law_3(xs, ys).valid);
  }

  SECTION("ties and non-positive values")
  {
    CHECK_FALSE(DiFfRG::detail::fit_power_law_3({{3., 2., 1.}}, {{2., 2., 1.}}).valid);
    CHECK_FALSE(DiFfRG::detail::fit_power_law_3({{3., 2., 1.}}, {{2., 1., 0.}}).valid);
    CHECK_FALSE(DiFfRG::detail::fit_power_law_3({{3., 3., 1.}}, {{3., 2., 1.}}).valid);
  }
}

TEST_CASE("ScalingRootFinder converges the observable and never leaves the bracket", "[common][root_finding]")
{
  uint calls = 0;
  ScalingRootFinder search(
      [&](const double x, double &obs, double &residual) -> bool {
        ++calls;
        // Requirement: no proposal, model-based or not, may escape the live bracket.
        CHECK(x >= 0.);
        CHECK(x <= 3.);
        return sc_probe(x, obs, residual);
      },
      sc_target, 1e-3, 40);
  search.set_bounds(0., 3.);
  search.set_residual_window(0.);

  const double x = search.search();

  CHECK(search.converged());
  CHECK(search.get_obs() >= sc_target);
  CHECK(search.get_obs() <= sc_target * (1. + 1e-3));
  CHECK(std::abs(x - sc_x_star()) < 1e-6);
  // The whole point: a 4-decade bracket in (x - x_c) resolved in a handful of evaluations,
  // where bisection to the same observable tolerance needs ~30.
  CHECK(calls <= 12);
  // A critical point must have been established -- by either branch -- and be right.
  CHECK(std::isfinite(search.get_x_critical()));
  CHECK(std::abs(search.get_x_critical() - sc_x_c) < 1e-6);
}

TEST_CASE("ScalingRootFinder widens hypothesised bounds that do not straddle", "[common][root_finding]")
{
  SECTION("bracket entirely above the root")
  {
    ScalingRootFinder search(sc_probe, sc_target, 1e-3, 60);
    search.set_bounds(1.0, 3.0); // both convergent, both above target
    CHECK(search.search() == Catch::Approx(sc_x_star()).epsilon(1e-4));
  }

  SECTION("bracket entirely below the root")
  {
    ScalingRootFinder search(sc_probe, sc_target, 1e-3, 60);
    search.set_bounds(-30.0, -0.5); // both divergent
    CHECK(search.search() == Catch::Approx(sc_x_star()).epsilon(1e-4));
  }
}

TEST_CASE("ScalingRootFinder terminates when the model lies", "[common][root_finding]")
{
  // A "power law" whose exponent flips halfway: any fit extrapolates to the wrong place. The
  // Brent stagnation safeguard must still drive the bracket down.
  uint calls = 0;
  ScalingRootFinder search(
      [&](const double x, double &obs, double &residual) -> bool {
        ++calls;
        if (x <= sc_x_c) {
          residual = sc_residual(x);
          return false;
        }
        const double d = x - sc_x_c;
        obs = (d > 1e-3) ? sc_C * std::pow(d, 0.9) : sc_C * std::pow(1e-3, 0.9 - 0.2) * std::pow(d, 0.2);
        return true;
      },
      sc_target, 1e-3, 60);
  search.set_bounds(0., 3.);

  CHECK_NOTHROW(search.search());
  CHECK(calls <= 60);
}

TEST_CASE("ScalingRootFinder in deep-scaling mode converges the critical point", "[common][root_finding]")
{
  // target == 0: the target IS the singularity, so the observable tolerance is vacuous and the
  // fitted critical point takes over. The returned point must still be a convergent one.
  ScalingRootFinder search(sc_probe, 0.0, 1e-8, 60);
  search.set_bounds(0., 3.);
  search.set_obs_window(0.); // no window: every convergent probe is admissible evidence

  const double x = search.search();

  CHECK(x > sc_x_c); // never lands on the singularity itself
  CHECK(std::abs(search.get_x_critical() - sc_x_c) < 1e-6);
}

TEST_CASE("ScalingRootFinder tolerates quantised residuals", "[common][root_finding]")
{
  // Reading t_fail out of HDF5 quantises it at output_dt. That scatters the divergent fit but
  // must not bias it, because only residual differences enter.
  ScalingRootFinder search(
      [&](const double x, double &obs, double &residual) -> bool {
        if (x <= sc_x_c) {
          residual = std::floor(sc_residual(x) / 0.25) * 0.25;
          return false;
        }
        obs = sc_obs(x);
        return true;
      },
      sc_target, 1e-3, 60);
  search.set_bounds(0., 3.);

  CHECK_NOTHROW(search.search());
  CHECK(search.converged());
}

TEST_CASE("ScalingRootFinder throws without bounds or without any success", "[common][root_finding]")
{
  {
    ScalingRootFinder search(sc_probe, sc_target, 1e-3, 10);
    CHECK_THROWS_AS(search.search(), std::runtime_error);
  }
  {
    ScalingRootFinder search([](const double, double &, double &) -> bool { return false; }, sc_target, 1e-3, 10);
    search.set_bounds(0., 3.);
    CHECK_THROWS_AS(search.search(), std::runtime_error);
  }
}

TEST_CASE("scaling fits recover the critical point of a recorded Yang-Mills tuning run", "[common][root_finding]")
{
  // Real probes from vacuum/YM_NumTracer (Lambda = 100). The bisection that produced them
  // needed 28 flows; these fits reach the same answer from three points each. m2A_c lies in
  // (-106.1843912303, -106.1843907088) -- the tightest bracket that run established.
  constexpr double m2A_c = -106.18439123;

  SECTION("convergent branch, three probes inside the scaling window")
  {
    // (m2A, Zc(0)) for the three probes with the smallest Zc(0).
    const std::array<double, 3> xs{{-106.1843776703, -106.1843860149, -106.1843901873}};
    const std::array<double, 3> ys{{0.008303434404644766, 0.004734391912027866, 0.0018394165060647556}};

    const auto fit = DiFfRG::detail::fit_power_law_3(xs, ys);
    REQUIRE(fit.valid);
    const double x_c = *std::min_element(xs.begin(), xs.end()) - fit.shift;
    // Bisection had this bracketed only to ~7e-4 at that point.
    CHECK(std::abs(x_c - m2A_c) < 1e-6);
    CHECK(fit.exponent == Catch::Approx(0.588).margin(0.02));
  }

  SECTION("divergent branch, the three crudest failing probes")
  {
    // (m2A, t_fail) for the first three divergent probes of the run, from a bracket 70 wide.
    // final_time = 15, output_dt = 0.2, so t_fail is quantised at 0.2.
    const double x1 = -150.0, x2 = -115.0, x3 = -106.25;
    const double r1 = 2.6, r2 = 2.8, r3 = 5.0;

    const double s = DiFfRG::detail::solve_scaling_shift(r2 - r1, r3 - r2, x3 - x1, x3 - x2);
    REQUIRE(std::isfinite(s));
    const double x_c = x3 + s;
    // Three failed flows put us within 0.1 of the critical point out of a 44-wide gap -- worth
    // roughly nine bisections, and the whole reason failures are fitted rather than counted.
    CHECK(std::abs(x_c - m2A_c) < 0.1);
  }
}

TEST_CASE("ScalingRootFinder BelowTarget accepts anything under the target", "[common][root_finding]")
{
  uint calls = 0;
  ScalingRootFinder search(
      [&](const double x, double &obs, double &residual) -> bool {
        ++calls;
        return sc_probe(x, obs, residual);
      },
      sc_target, 1e-3, 40);
  search.set_bounds(0., 3.);
  search.set_acceptance(ScalingRootFinder::Acceptance::BelowTarget);

  const double x = search.search();

  CHECK(search.converged());
  // The answer must lie strictly between the critical point and the target crossing.
  CHECK(x > sc_x_c);
  CHECK(search.get_obs() > 0.);
  CHECK(search.get_obs() < sc_target);
  // Hitting an interval is easier than hitting a point, but not automatically cheaper: aiming
  // at the middle of (0, target) sits closer to the critical point, so an imperfect fit can
  // overshoot into the divergent region. On this oracle it costs slightly more than AtTarget
  // (11 vs 8); the win shows up when the admissible interval is the binding constraint.
  CHECK(calls <= 14);
}

TEST_CASE("ScalingRootFinder BelowTarget never returns a divergent point", "[common][root_finding]")
{
  // A model whose exponent lies about where the crossing is, so the aim overshoots repeatedly.
  ScalingRootFinder search(
      [&](const double x, double &obs, double &residual) -> bool {
        if (x <= sc_x_c) {
          residual = sc_residual(x);
          return false;
        }
        obs = sc_C * std::pow(x - sc_x_c, 0.15); // far flatter than the fit will infer
        return true;
      },
      sc_target, 1e-3, 60);
  search.set_bounds(0., 3.);
  search.set_acceptance(ScalingRootFinder::Acceptance::BelowTarget);

  const double x = search.search();
  CHECK(x > sc_x_c);
  double obs = NAN, residual = NAN;
  REQUIRE(sc_probe(x, obs, residual));
  CHECK(std::isfinite(obs));
}
