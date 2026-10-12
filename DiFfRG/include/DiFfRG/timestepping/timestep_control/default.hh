#pragma once

// standard library
#include <cmath>
#include <stdexcept>
#include <string>

#include <DiFfRG/common/math.hh>

namespace DiFfRG
{
  /**
   * @brief Step-size control for the Newton-based implicit steppers (TimeStepperImplicitEuler, TimeStepperTRBDF2);
   * base class of TC_PI.
   *
   * Calls the step function, advances the time, keeps the step in [min_dt, max_dt] and ends exactly at max_t.
   * Output is the caller's business. When the suggested step falls below min_dt, the step min_dt is taken and accepted
   * even if Newton does not converge. This base class keeps the step size constant and aborts on the first failed step;
   * TC_PI adapts it. Throws if the time does not advance for more than 10 steps.
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
     */
    TC_Default(NEWT &newton_, unsigned int alg_order_, double t_, double max_t_, double dt_, double min_dt_,
               double max_dt_)
        : newton(newton_), alg_order(alg_order_), t(t_), max_t(max_t_), sug_dt(dt_), min_dt(min_dt_), max_dt(max_dt_),
          cur_dt(sug_dt), last_t(t_), stuck(0), fin(false)
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
     * @param f performs the step from t to t + dt, `void(double t, double dt)`; throws on failure.
     */
    template <typename F> void advance(F &f)
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
      if (t - 1e-4 * cur_dt >= max_t) fin = true;
      // Unrecoverably stuck: throw rather than finish, since callers rely on ending at max_t.
      if (stuck > 10) {
        throw std::runtime_error("Timestepping got stuck at t = " + std::to_string(t));
      }
    }

  protected:
    /// Called after a successful step: advance the time. The step size stays the suggested one.
    virtual void step_success()
    {
      t += cur_dt;
      cur_dt = sug_dt;
    }
    /// Called after a failed step; this default throws.
    virtual void step_fail(const std::exception &e)
    {
      throw std::runtime_error("Timestepping failed. Error: \n" + std::string(e.what()));
    }

    NEWT &newton;
    unsigned int alg_order;
    double t, max_t, sug_dt, min_dt, max_dt, cur_dt, last_t;
    unsigned int stuck;

    bool fin;
  };
} // namespace DiFfRG
