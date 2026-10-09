#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>

#include <array>
#include <cmath>
#include <string_view>

using namespace DiFfRG;

namespace
{
  void check_matsubara_nodes_and_weights(const MatsubaraQuadrature<double> &mq, const double T,
                                         const double typical_E, const std::string_view context)
  {
    const auto nodes = mq.nodes<CPU_memory>();
    const auto weights = mq.weights<CPU_memory>();

    REQUIRE(mq.size() > 0);
    for (std::size_t i = 0; i < mq.size(); ++i) {
      const double node = nodes[i];
      const double weight = weights[i];
      CAPTURE(context, T, typical_E, mq.size(), i, node, weight);
      REQUIRE(std::isfinite(node));
      REQUIRE(std::isfinite(weight));
      REQUIRE(node > 0.);
      REQUIRE(weight > 0.);
    }
  }
} // namespace

TEST_CASE("Test matsubara quadrature rule", "[double][quadrature][matsubara]")
{
  DiFfRG::Init();

  SECTION("Simple test")
  {
    const double T = GENERATE(take(64, random(0.01, 1.0)));
    MatsubaraQuadrature<double> mq(T, 1.);

    check_matsubara_nodes_and_weights(mq, T, 1., "simple generated quadrature");

    const auto f = [&](const double x) { return 1. / (1. + powr<2>(x)); };

    const double reference = 0.5 * 1. / std::tanh(0.5 / T);
    const double sum = mq.sum(f);

    CAPTURE(T, mq.size(), sum, reference);
    constexpr double expected_precision = 1e-9;
    CHECK(sum == Catch::Approx(reference).epsilon(expected_precision));
  }

  SECTION("Test with different parameters")
  {
    const double T = GENERATE(take(8, random(0.001, 0.1)));
    const double a = GENERATE(take(8, random(0.1, 1.0)));
    const double b = GENERATE(take(8, random(0.01, 0.1)));

    MatsubaraQuadrature<double> mq(T, a);

    check_matsubara_nodes_and_weights(mq, T, a, "generated parameter quadrature");

    const auto f = [&](const double x) { return 1. / (powr<2>(a) + b * x + powr<2>(x)); };

    // Analytically this is sinh(X) / (R * (cosh(X) - cos(y))) with R = sqrt(4a^2 - b^2),
    // X = R/(2T), y = b/(2T) -- but it must not be evaluated in that form. Over these parameter
    // ranges X reaches ~10^3, so sinh(X) and cosh(X) are astronomically large (they overflow
    // outright near the corner T=0.001, a=1) and the value is recovered only as their ratio.
    // Dividing through by cosh(X) first keeps every intermediate bounded: tanh -> 1 and
    // cos(y)/cosh(X) -> 0 (or exactly 0 once cosh overflows, which is the right limit).
    // Verified against a direct Matsubara sum; the unnormalised form disagreed with it by
    // factors of ~10^3 on a few percent of draws, which is what made this test flaky.
    const double R = std::sqrt(4 * std::pow(a, 2) - std::pow(b, 2));
    const double reference = std::tanh(R / (2. * T)) / (R * (1. - std::cos(b / (2. * T)) / std::cosh(R / (2. * T))));

    const double result = mq.sum(f);

    CAPTURE(T, a, b, mq.size(), result, reference);
    constexpr double expected_precision = 1e-6;
    CHECK(result == Catch::Approx(reference).epsilon(expected_precision));
  }

  SECTION("Test at T=0")
  {
    const double T = 0.;
    const double a = GENERATE(take(64, random(0.01, 10.0)));

    MatsubaraQuadrature<double> mq(T, a);

    check_matsubara_nodes_and_weights(mq, T, a, "vacuum generated quadrature");

    const auto f = [&](const double x) { return 1. / (powr<2>(a) + powr<2>(x)); };

    const double reference = 1 / (2. * a);

    const double result = mq.sum(f);

    CAPTURE(T, a, mq.size(), result, reference);
    // The tangent-map rule is essentially exact on this integrand: p0 = a tan(theta) sends
    // 1/(a^2+p0^2) dp0 to the constant d(theta)/a. The old two-piece linear+log rule needed
    // ~60 nodes for 1e-9 here.
    constexpr double expected_precision = 1e-12;
    CHECK(result == Catch::Approx(reference).epsilon(expected_precision));
  }

  SECTION("Test at T=0, pole away from the mapping scale")
  {
    // The rule is built around typical_E, but a propagator pole sits at its own energy. A
    // mismatch of a decade either way is routine (spatial momentum up to sqrt(x_extent)*k,
    // masses above or below k), so it must not cost accuracy at the shipped node count.
    const double T = 0.;
    const double typical_E = 1.;
    const double ratio = GENERATE(0.1, 0.3, 1.0, 3.0, 10.0);
    const double pole = ratio * typical_E;

    MatsubaraQuadrature<double> mq(T, typical_E, 2, 0, 256, 64);

    const auto f = [&](const double x) { return 1. / (powr<2>(pole) + powr<2>(x)); };
    const double reference = 1 / (2. * pole);
    const double result = mq.sum(f);

    CAPTURE(T, typical_E, pole, mq.size(), result, reference);
    CHECK(result == Catch::Approx(reference).epsilon(1e-11));
  }

  SECTION("Test at T=0, a 1/p0^2 tail has no error floor")
  {
    // This is the regression guard for the defect the tangent map fixes. The old rule
    // truncated the frequency integral at 1e11 * typical_E, which floored the relative error
    // of a 1/p0^2 tail at ~6.4e-12 -- going from 81 to 513 nodes did not improve it at all.
    // An unregulated temporal direction is exactly this case, so the floor has to stay gone.
    const double a = 1.;
    const auto f = [&](const double x) { return 1. / (powr<2>(a) + powr<2>(x)); };
    const double reference = 1 / (2. * a);

    for (const int size : {32, 64, 128, 256}) {
      MatsubaraQuadrature<double> mq;
      mq.reinit_with_size(size, 0., a);
      const double err = std::abs(mq.sum(f) - reference) / reference;
      CAPTURE(size, mq.size(), err);
      CHECK(err < 1e-13);
    }
  }
}

TEST_CASE("Matsubara quadrature nodes and weights are finite for representative parameters",
          "[double][quadrature][matsubara][diagnostic]")
{
  DiFfRG::Init();

  constexpr std::array<double, 8> temperatures = {0., 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 5.0e-1, 1.0};
  constexpr std::array<double, 8> typical_energies = {1.0e-2, 5.0e-2, 1.0e-1, 2.5e-1,
                                                      5.0e-1, 1.0,    2.0,    10.0};

  for (const double T : temperatures) {
    for (const double typical_E : typical_energies) {
      MatsubaraQuadrature<double> mq(T, typical_E);
      check_matsubara_nodes_and_weights(mq, T, typical_E, "representative quadrature");
    }
  }
}

TEST_CASE("Test exact Matsubara sum", "[double][quadrature][matsubara][exact-sum]")
{
  DiFfRG::Init();

  SECTION("Node count and placement")
  {
    const double T = GENERATE(0.01, 0.1, 0.5, 1.3);
    const double cutoff = GENERATE(0.5, 1., 7., 100.);

    MatsubaraQuadrature<double> mq;
    mq.reinit_exact_sum(T, cutoff);

    const int n = MatsubaraQuadrature<double>::modes_below(T, cutoff);
    CAPTURE(T, cutoff, n);
    REQUIRE((int)mq.size() == n);
    // The zero mode is a node of the summation list but not of size().
    REQUIRE(mq.sum_size() == mq.size() + 1);

    const auto nodes = mq.nodes<CPU_memory>();
    for (int i = 0; i < n; ++i) {
      // The true bosonic frequencies, not a Gaussian approximation to them.
      REQUIRE_THAT(nodes[i], Catch::Matchers::WithinRel(2. * M_PI * (i + 1) * T, 1e-14));
      REQUIRE(nodes[i] <= cutoff);
    }
    // ...and the first mode left out is genuinely outside the cutoff.
    REQUIRE(2. * M_PI * (n + 1) * T > cutoff);

    const auto sum_nodes = mq.sum_nodes<CPU_memory>();
    const auto sum_weights = mq.sum_weights<CPU_memory>();
    REQUIRE(sum_nodes[mq.size()] == 0.);
    // w * (f(+0) + f(-0)) must reproduce T * f(0).
    REQUIRE_THAT(sum_weights[mq.size()], Catch::Matchers::WithinRel(T / 2., 1e-15));
  }

  SECTION("Exact on a compactly supported summand")
  {
    // A summand that vanishes identically outside |p0| <= R is not approximated by this rule --
    // it IS the sum. Check that against the brute-force enumeration it claims to be.
    const double T = GENERATE(0.02, 0.11, 0.7);
    const double R = GENERATE(1., 3.7, 20.);

    auto f = [&](const double p0) {
      const double u = p0 * p0 / (R * R);
      return u >= 1. ? 0. : (1. - u) * (1. - u) * (1. - u) / (1. + p0 * p0);
    };

    MatsubaraQuadrature<double> mq;
    mq.reinit_exact_sum(T, R);

    double brute = T * f(0.);
    for (int n = 1; 2. * M_PI * n * T < 2. * R; ++n)
      brute += T * (f(2. * M_PI * n * T) + f(-2. * M_PI * n * T));

    CAPTURE(T, R, mq.size(), brute);
    REQUIRE_THAT(mq.sum(f), Catch::Matchers::WithinRel(brute, 1e-13));
  }

  SECTION("Agrees with the Monien rule where both are valid")
  {
    // Effectively compact (a super-exponential cut at ~R) but smooth and algebraically tailed
    // enough for the Gaussian rule, so the two constructions must land on the same number.
    const double T = GENERATE(0.05, 0.2);
    const double E = 1.;
    const double w = 2. * E; // width of the super-exponential cut
    const double R = 8. * E; // cutoff handed to the exact sum: (R/w)^8 = 65536, i.e. dead by any measure

    auto f = [&](const double p0) {
      const double u = p0 / w;
      return std::exp(-std::pow(u * u, 4)) / (p0 * p0 + E * E);
    };

    MatsubaraQuadrature<double> monien(T, E);
    MatsubaraQuadrature<double> exact;
    exact.reinit_exact_sum(T, R);

    CAPTURE(T, monien.size(), exact.size());
    REQUIRE_THAT(exact.sum(f), Catch::Matchers::WithinRel(monien.sum(f), 1e-9));
  }
}

TEST_CASE("Finite-interval rule vs the tangent map on compact summands",
          "[double][quadrature][matsubara][finite-interval]")
{
  DiFfRG::Init();

  const double E = 1., R = 8.;

  // TWO shapes, because which rule wins depends on where the integrand actually lives.
  //   "peaked"  concentrates near p0 = 0 and is dead well before R -- the tangent map's node
  //             clustering at small p0 is built for exactly this, and it wins.
  //   "spread"  is O(1) across the whole support and vanishes at the edge, which is the shape a
  //             regulator insertion produces: RBdot(k^2, p0^2+l1^2) dies gradually over [0, R].
  auto peaked = [&](const double p0) {
    const double u = p0 / 2.;
    return std::exp(-std::pow(u * u, 4)) / (p0 * p0 + E * E);
  };
  auto spread = [&](const double p0) {
    const double u = p0 / R;
    return u >= 1. ? 0. : powr<3>(1. - u * u) / (p0 * p0 + E * E);
  };

  Quadrature<double> gl512(512, QuadratureType::legendre);

  auto interval_err = [&](const auto &f, const int n, const double exact) {
    Quadrature<double> gl(n, QuadratureType::legendre);
    MatsubaraQuadrature<double> q;
    q.reinit_finite_interval(R, gl.nodes<CPU_memory>(), gl.weights<CPU_memory>());
    return std::abs(q.sum(f) - exact) / std::abs(exact);
  };
  auto tangent_err = [&](const auto &f, const int n, const double exact) {
    MatsubaraQuadrature<double> q;
    q.reinit_with_size(n, 0., E);
    return std::abs(q.sum(f) - exact) / std::abs(exact);
  };

  for (int shape = 0; shape < 2; ++shape) {
    MatsubaraQuadrature<double> ref;
    ref.reinit_finite_interval(R, gl512.nodes<CPU_memory>(), gl512.weights<CPU_memory>());
    const double exact = shape == 0 ? ref.sum(peaked) : ref.sum(spread);

    const double i32 = shape == 0 ? interval_err(peaked, 32, exact) : interval_err(spread, 32, exact);
    const double i64 = shape == 0 ? interval_err(peaked, 64, exact) : interval_err(spread, 64, exact);
    const double t64 = shape == 0 ? tangent_err(peaked, 64, exact) : tangent_err(spread, 64, exact);

    WARN("shape=" << (shape == 0 ? "peaked" : "spread") << "  interval32=" << i32 << "  interval64=" << i64
                  << "  tangent64=" << t64);

    // At EQUAL node count the finite-interval rule wins on both shapes -- that is the defensible
    // claim, and the only one measured. How much it wins by depends entirely on where the
    // integrand lives: on `peaked` it is ~100x (8e-10 vs 9e-8), on `spread` it is many orders
    // (1e-15 vs worse than 1e-6), because there the tangent map has to resolve the support EDGE,
    // which lands at theta = 0.57*(pi/2) where its nodes are sparsest.
    //
    // Note what this does NOT say: interval@32 does not beat tangent@64 on `peaked` (6.6e-6 vs
    // 9.3e-8). Halving the order is a real trade, not a free one, which is why it is a knob.
    REQUIRE(i64 < t64);
    REQUIRE(i64 < 1e-8);
    REQUIRE(i32 < i64 * 1e7); // the lower order still converges, just less far
  }

  SECTION("no zero mode, and the caller convention is the one sum() expects")
  {
    Quadrature<double> gl32(32, QuadratureType::legendre);
    MatsubaraQuadrature<double> interval;
    interval.reinit_finite_interval(R, gl32.nodes<CPU_memory>(), gl32.weights<CPU_memory>());
    // Integrates rather than sums, like the T=0 rule: sum_nodes() must not append a zero mode.
    REQUIRE(interval.get_T() == 0.);
    REQUIRE(interval.sum_size() == interval.size());

    // A flat integrand checks the normalisation outright: (1/2pi) int_{-R}^{R} 1 dp0 = R/pi.
    REQUIRE_THAT(interval.sum([](const double) { return 1.; }), Catch::Matchers::WithinRel(R / M_PI, 1e-13));
  }
}
