#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/timestepping/abm_error_control.hh>


#include <cmath>

using namespace DiFfRG;

namespace
{
  // y' = sech^2((t - t_c) / w) / w, i.e. y(t) = tanh((t - t_c) / w) + const: a step of width w << dt at t_c.
  constexpr double t_c = 0.5, width = 1e-2;
  double exact(const double t) { return std::tanh((t - t_c) / width) - std::tanh(-t_c / width); }

  struct Result {
    double t;
    double error;
    size_t steps;
    size_t rejected;
  };

  Result integrate(const bool enabled, const double t_stop)
  {
    const auto rhs = [](const Eigen::VectorXd & /*y*/, Eigen::VectorXd &dydt, const double t) {
      const double c = std::cosh((t - t_c) / width);
      dydt.resize(1);
      dydt[0] = 1. / (width * c * c);
    };
    const double base_dt = 0.05;
    internal::ABMWithErrorEstimate<8> abm;
    internal::ABMStepControl control({.enabled = enabled,
                                                       .aligned = true,
                                                       .t0 = 0.,
                                                       .base_dt = base_dt,
                                                       .minimal_dt = 1e-10,
                                                       .abs_tol = 1e-6,
                                                       .rel_tol = 1e-6});
    Eigen::VectorXd y = Eigen::VectorXd::Zero(1);
    double t = 0., dt = base_dt;
    size_t steps = 0;
    while (t < t_stop - 1e-3 * dt) {
      control.step(abm, rhs, y, t, dt, [](double, double, double) {});
      ++steps;
    }
    return {t, std::abs(y[0] - exact(t)), steps, control.rejected()};
  }
} // namespace

TEST_CASE("ABM error control resolves a feature far below the base step", "[timestepping][abm][error_control]")
{
  const auto fixed = integrate(false, 1.);
  const auto controlled = integrate(true, 1.);
  INFO("fixed: error " << fixed.error << ", " << fixed.steps << " steps");
  INFO("controlled: error " << controlled.error << ", " << controlled.steps << " steps, " << controlled.rejected
                            << " rejected");

  REQUIRE(fixed.rejected == 0);
  REQUIRE(fixed.steps == 20);
  REQUIRE(fixed.error > 1e-2);

  REQUIRE(controlled.rejected > 0);
  REQUIRE(controlled.error < 1e-4);
  // The step grows back after the feature: far fewer steps than resolving everything at the smallest size.
  REQUIRE(controlled.steps < 400);
  // Aligned growth keeps the steps on the base grid, so the run ends exactly on t_stop.
  REQUIRE_THAT(controlled.t, Catch::Matchers::WithinAbs(1., 1e-12));
}

TEST_CASE("ABM Milne factor", "[timestepping][abm][error_control]")
{
  // Steps = 2: AB2 error constant 5/12, trapezoidal rule -1/12, so the corrector error is
  // (1/12) / (5/12 + 1/12) = 1/6 of the predictor-corrector difference.
  REQUIRE_THAT((internal::ABMWithErrorEstimate<2>::milne_factor()),
               Catch::Matchers::WithinRel(1. / 6., 1e-12));
}
