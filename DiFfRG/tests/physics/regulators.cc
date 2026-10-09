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
