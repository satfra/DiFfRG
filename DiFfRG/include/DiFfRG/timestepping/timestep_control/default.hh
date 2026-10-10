#pragma once

// standard library
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <DiFfRG/common/math.hh>

namespace DiFfRG
{
  /**
   * @brief Step-size control for the Newton-based implicit steppers (TimeStepperImplicitEuler, TimeStepperTRBDF2);
   * base class of TC_PI.
   *
   * Calls the step function, advances the time, writes output, keeps the step in [min_dt, max_dt] and ends exactly
   * at max_t. When the suggested step falls below min_dt, the step min_dt is taken and accepted even if Newton does
   * not converge. This base class keeps the step size constant and aborts on the first failed step; TC_PI adapts it.
   * Throws if the time does not advance for more than 10 steps.
   *
   * @tparam NEWT the solver; must provide set_ignore_nonconv(bool) and get_error().
   */
  template <typename NEWT> class TC_Default
  {
  public:
    /**
     * @param newton_ the solver, see NEWT.
     * @param alg_order_ order of the time discretization, used by TC_PI.
     * @param t_ start time.
     * @param max_t_ final time.
     * @param dt_ initial step.
     * @param min_dt_, max_dt_ bounds of the step.
     * @param output_dt_ output interval.
     */
    TC_Default(NEWT &newton_, unsigned int alg_order_, double t_, double max_t_, double dt_, double min_dt_,
               double max_dt_, double output_dt_)
        : newton(newton_), alg_order(alg_order_), t(t_), max_t(max_t_), sug_dt(dt_), min_dt(min_dt_), max_dt(max_dt_),
          output_dt(output_dt_), cur_dt(sug_dt), last_save(t), last_t(t_), t_start(t_), stuck(0), fin(false)
    {
    }

    /// The step to take next.
    double get_dt() const { return cur_dt; }
    /// The current time.
    double get_t() const { return t; }
    /// How many steps in a row did not advance the time.
    unsigned int get_stuck() const { return stuck; }
    /// Whether max_t has been reached.
    bool finished() const { return fin; }

    /**
     * @brief Attempt one step and plan the next.
     *
     * Writes a frame at the first time past each multiple of output_dt from the start, and at the end.
     *
     * @param f performs the step from t to t + dt, `void(double t, double dt)`; throws on failure.
     * @param of writes output at time t, `void(double t)`.
     */
    template <typename F, typename OF> void advance(F &f, OF &of)
    {
      // first, try to do the timestep
      try {
        f(t, cur_dt);
        // on a success, possibly amend dt
        this->step_success();
      } catch (const std::exception &e) {
        // on a fail, possibly amend dt
        this->step_fail(e);
      }

      // check if the stepper got stuck on some time t
      if (is_close(t, last_t))
        stuck++;
      else {
        stuck = 0;
        last_t = t;
      }

      // check if the stepper should output data
      if (t > last_save + output_dt) {
        of(t);
        // Anchored at the start, like the output grids of the other steppers.
        last_save = t_start + std::floor((t - t_start) / output_dt) * output_dt;
      }

      // should the suggested timestep be below the minimum timestep, tell the solver that it should
      // accept even timesteps which do not fulfill the convergence criterion (otherwise the evolution would just stop)
      newton.set_ignore_nonconv(false);
      if (sug_dt < min_dt) {
        cur_dt = min_dt;
        newton.set_ignore_nonconv(true);
      } else if (sug_dt > max_dt)
        cur_dt = max_dt;
      else
        cur_dt = sug_dt;

      // do not over-step the final time
      if (t + cur_dt > max_t) cur_dt = max_t - t;

      // if we are over max_t, the timestepping is finished
      if (t - 1e-4 * cur_dt >= max_t) {
        of(t);
        fin = true;
      }
      // Unrecoverably stuck: throw rather than finish, since callers rely on ending at max_t.
      if (stuck > 10) {
        of(t);
        throw std::runtime_error("Timestepping got stuck at t = " + std::to_string(t));
      }
    }

    /// Unused.
    void print_step_info()
    {
      std::ostringstream message;
      message << std::setw(16) << "| step from t = " << std::setw(8) << std::setprecision(5) << last_t << std::setw(8)
              << " to t = ";
    }

  protected:
    /// Called after a successful step: advance the time. The step size stays the suggested one.
    virtual void step_success()
    {
      t += cur_dt;
      if (t + sug_dt - 1e-4 * sug_dt > last_save + output_dt)
        cur_dt = last_save + output_dt - t;
      else
        cur_dt = sug_dt;
    }
    /// Called after a failed step; this default throws.
    virtual void step_fail(const std::exception &e)
    {
      throw std::runtime_error("Timestepping failed. Error: \n" + std::string(e.what()));
    }

    NEWT &newton;
    unsigned int alg_order;
    double t, max_t, sug_dt, min_dt, max_dt, output_dt, cur_dt, last_save, last_t;
    const double t_start;
    unsigned int stuck;

    bool fin;
  };
} // namespace DiFfRG
