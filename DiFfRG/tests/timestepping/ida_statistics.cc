#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/timestepping/ida_driver.hh>

#include <deal.II/lac/vector.h>

#include <cmath>

using namespace DiFfRG;

// IDADriver::get_statistics() counts over the whole solve: a restart recreates IDA, whose own counters start over,
// but the reported totals must not.

namespace
{
  /// Solve du/dt = -u on [0, 1], restarting IDA at every output if @p restart; counts the residual calls.
  long int residual_calls(const bool restart, const bool warm, IDADriver<dealii::Vector<double>>::Statistics &stats)
  {
    using VectorType = dealii::Vector<double>;
    IDAParameters prm;
    prm.output_period = 0.1;
    prm.maximal_step_size = 0.05;
    prm.absolute_tolerance = 1e-12;
    prm.relative_tolerance = 1e-9;
    prm.warm_restart = warm;
    IDADriver<VectorType> driver(prm);

    long int calls = 0;
    double alpha_last = 1.;
    driver.residual = [&](const double, const VectorType &y, const VectorType &y_dot, VectorType &res) {
      ++calls;
      res[0] = y_dot[0] + y[0];
    };
    driver.setup_jacobian = [&](const double, const VectorType &, const VectorType &, const double alpha) {
      alpha_last = alpha;
    };
    driver.solve_with_jacobian = [&](const VectorType &src, VectorType &dst, const double) {
      dst[0] = src[0] / (alpha_last + 1.);
    };
    double last_restart = -1.;
    driver.solver_should_restart = [&](const double t, VectorType &, VectorType &) {
      if (!restart || t == last_restart) return false;
      last_restart = t;
      return true;
    };

    VectorType u(1), u_dot(1);
    u[0] = 1.;
    u_dot[0] = -1.;
    driver.solve_dae(u, u_dot);
    REQUIRE(std::abs(u[0] - std::exp(-1.)) < 1e-6);
    stats = driver.get_statistics();
    return calls;
  }
} // namespace

TEST_CASE("IDA statistics are totals across restarts", "[timestepping][sundials_ida]")
{
  DiFfRG::Init();
  for (const bool warm : {true, false}) {
    CAPTURE(warm);
    IDADriver<dealii::Vector<double>>::Statistics plain, restarted;
    const long int plain_calls = residual_calls(false, warm, plain);
    const long int restarted_calls = residual_calls(true, warm, restarted);
    REQUIRE(plain.valid);
    REQUIRE(restarted.valid);
    // Every residual call goes through IDA's counter, except those of IDACalcIC at a cold (re)start, which IDA
    // counts separately: the totals must at least cover all calls after the last restart and at most all calls.
    CHECK(plain.num_residual_evaluations <= plain_calls);
    CHECK(restarted.num_residual_evaluations <= restarted_calls);
    // Nine restarts: the counts of the instances before the last one must be included.
    CHECK(restarted.num_residual_evaluations > restarted_calls / 2);
    CHECK(restarted.num_steps >= plain.num_steps);
    CHECK(restarted.num_linear_solver_setups >= 9);
  }
}
