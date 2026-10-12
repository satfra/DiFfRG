#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/physics/regulators.hh>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

using namespace DiFfRG;

namespace
{
  template <int n> struct RationalExpOrder : RationalExpRegulatorOpts {
    static constexpr int order = n;
  };
  template <int n> struct PolynomialExpOrder : PolynomialExpRegulatorOpts {
    static constexpr int order = n;
  };

  // Every regulator function evaluated in float must stay float -- a double constant anywhere in
  // its body would promote the result -- and agree with the double evaluation to float accuracy.
  // The error is taken against the function's scale over the sample, not pointwise: several of
  // these functions pass through zero.
  template <typename Reg, typename F> void check_function(const char *name, F &&f)
  {
    static_assert(std::is_same_v<decltype(f(1.f, 1.f)), float>, "regulator function promotes float");
    std::vector<double> err, ref;
    const double k2 = 1.3;
    for (int i = 0; i <= 400; ++i) {
      const double q2 = k2 * (1e-3 + 4. * i / 400.);
      const double d = f(k2, q2);
      const float s = f(float(k2), float(q2));
      if (!std::isfinite(d)) continue;
      ref.push_back(std::abs(d));
      err.push_back(std::abs(double(s) - d));
    }
    REQUIRE(!ref.empty());
    const double scale = *std::max_element(ref.begin(), ref.end());
    const double max_err = *std::max_element(err.begin(), err.end());
    INFO(name << ": max|float - double| = " << max_err << ", scale = " << scale);
    CHECK(max_err <= 1e-4 * scale);
  }
} // namespace

TEMPLATE_TEST_CASE("Regulators evaluate in single precision", "[physics][regulators][float]", LitimRegulator<>,
                   BosonicRegulator<>, ExponentialRegulator<>, SmoothedLitimRegulator<>, RationalExpRegulator<>,
                   RationalExpRegulator<RationalExpOrder<2>>, RationalExpRegulator<RationalExpOrder<5>>,
                   RationalExpRegulator<RationalExpOrder<12>>, RationalExpRegulator<RationalExpOrder<16>>,
                   PolynomialExpRegulator<>, PolynomialExpRegulator<PolynomialExpOrder<3>>)
{
  using R = TestType;
  check_function<R>("RB", [](auto k2, auto q2) { return R::RB(k2, q2); });
  check_function<R>("RBdot", [](auto k2, auto q2) { return R::RBdot(k2, q2); });
  check_function<R>("RF", [](auto k2, auto q2) { return R::RF(k2, q2); });
  check_function<R>("RFdot", [](auto k2, auto q2) { return R::RFdot(k2, q2); });
  if constexpr (requires { R::dq2RB(1.f, 1.f); })
    check_function<R>("dq2RB", [](auto k2, auto q2) { return R::dq2RB(k2, q2); });
  if constexpr (requires { R::dq2RF(1.f, 1.f); })
    check_function<R>("dq2RF", [](auto k2, auto q2) { return R::dq2RF(k2, q2); });
}

namespace
{
  // Compares an analytic derivative with a central finite difference of f, over the same sample of
  // x = q^2/k^2 as above. Points near x = 1 are skipped: the Litim regulator has a kink there.
  template <typename D, typename F> void check_derivative(const char *name, D &&analytic, F &&finite_difference)
  {
    std::vector<double> err, ref;
    const double k2 = 1.3;
    for (int i = 0; i <= 400; ++i) {
      const double x = 1e-3 + 4. * i / 400.;
      if (std::abs(x - 1.) < 1e-2) continue;
      const double a = analytic(k2, k2 * x);
      const double n = finite_difference(k2, k2 * x);
      if (!std::isfinite(a) || !std::isfinite(n)) continue;
      ref.push_back(std::abs(n));
      err.push_back(std::abs(a - n));
    }
    REQUIRE(!ref.empty());
    const double scale = *std::max_element(ref.begin(), ref.end());
    const double max_err = *std::max_element(err.begin(), err.end());
    INFO(name << ": max|analytic - finite difference| = " << max_err << ", scale = " << scale);
    CHECK(max_err <= 1e-6 * scale);
  }

  // d/dq^2 and d/dt (t = ln k, so d/dt = 2 k^2 d/dk^2 at fixed q^2) by central differences.
  template <typename F> auto d_q2(F f)
  {
    return [f](double k2, double q2) {
      const double h = 1e-6 * k2;
      return (f(k2, q2 + h) - f(k2, q2 - h)) / (2. * h);
    };
  }
  template <typename F> auto d_t(F f)
  {
    return [f](double k2, double q2) {
      const double h = 1e-6 * k2;
      return 2. * k2 * (f(k2 + h, q2) - f(k2 - h, q2)) / (2. * h);
    };
  }
} // namespace

TEMPLATE_TEST_CASE("Regulator derivatives match finite differences", "[physics][regulators]", LitimRegulator<>,
                   BosonicRegulator<>, ExponentialRegulator<>, SmoothedLitimRegulator<>, RationalExpRegulator<>,
                   RationalExpRegulator<RationalExpOrder<2>>, RationalExpRegulator<RationalExpOrder<5>>,
                   PolynomialExpRegulator<>, PolynomialExpRegulator<PolynomialExpOrder<3>>)
{
  using R = TestType;
  const auto RB = [](double k2, double q2) { return R::RB(k2, q2); };
  const auto RF = [](double k2, double q2) { return R::RF(k2, q2); };
  check_derivative("RBdot", [](double k2, double q2) { return R::RBdot(k2, q2); }, d_t(RB));
  check_derivative("RFdot", [](double k2, double q2) { return R::RFdot(k2, q2); }, d_t(RF));
  if constexpr (requires { R::dq2RB(1., 1.); })
    check_derivative("dq2RB", [](double k2, double q2) { return R::dq2RB(k2, q2); }, d_q2(RB));
  if constexpr (requires { R::dq2RF(1., 1.); })
    check_derivative("dq2RF", [](double k2, double q2) { return R::dq2RF(k2, q2); }, d_q2(RF));
}
