#pragma once

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/timestepping/timestep_control/default.hh>

namespace DiFfRG
{
  /**
   * @brief A simple PI controller which adjusts time steps in a smooth fashion depending on how well the solver
   * performs, taking into account the most recent time step, too.
   *
   * @tparam NEWT the used solver type which should at least explose the methods set_ignore_nonconv(bool)->void and
   * get_error()->double.
   */
  template <typename NEWT> class TC_PI : public TC_Default<NEWT>
  {
  public:
    using Base = TC_Default<NEWT>;
    TC_PI(NEWT &newton_, unsigned int alg_order_, double t_, double max_t_, double dt_, double min_dt_, double max_dt_,
          double output_dt_)
        : Base(newton_, alg_order_, t_, max_t_, dt_, min_dt_, max_dt_, output_dt_)
    {
      // these are magic numbers, inspired by the implementation of the PI controller in DifferentialEquations.jl,
      // although a tad more conservative (i.e. cautious when increasing dt)
      beta1 = 7. / (10. * alg_order);
      beta2 = 2. / (5. * alg_order);
      qmin = 0.2;
      qmax = 10.;
      qsteady_min = 0.99;
      qsteady_max = 1.2;
      qoldinit = 1e-4;
      qold = qoldinit;
      gamma = 0.9;
    }

  protected:
    using Base::t, Base::newton, Base::cur_dt, Base::sug_dt, Base::alg_order;

    /**
     * @brief This method just updates q for the calculation of the next sug_dt
     */
    void get_q()
    {
      const double EEst = newton.get_error();
      q11 = std::pow(EEst, beta1);
      q = q11 / std::pow(qold, beta2);
      q = std::max(1. / qmax, std::min(1. / qmin, q / gamma));
    }

    /**
     * @brief Increment t and suggest a dt based on current error and previous error
     */
    virtual void step_success() override
    {
      t += cur_dt;

      const double EEst = newton.get_error();
      get_q();

      if (q <= qsteady_max && q >= qsteady_min) q = 1.;
      qold = std::max(EEst, qoldinit);

      sug_dt = cur_dt / q;
    }

    /**
     * @brief On a fail, decrease the timestep by at least a factor of 2 relative to the failed step.
     *
     * The error estimate alone cannot guarantee that: a failure need not come with EEst > 1, and the failed step may
     * have been clipped below sug_dt (to hit max_t or an output time), so scaling sug_dt could retry the same step.
     */
    virtual void step_fail(const std::exception &) override
    {
      get_q();
      sug_dt = cur_dt / std::max(2., std::min(1. / qmin, q11 / gamma));
    }

    double beta1, beta2, qsteady_min, qsteady_max, qmin, qmax, qoldinit, qold, q11, q, gamma;
  };
} // namespace DiFfRG
