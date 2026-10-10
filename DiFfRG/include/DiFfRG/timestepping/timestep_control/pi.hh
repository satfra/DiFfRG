#pragma once

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/timestepping/timestep_control/default.hh>

namespace DiFfRG
{
  /**
   * @brief Proportional-integral step-size control from the solver's error measure.
   *
   * With the error \f$E_n\f$ = `newton.get_error()` of the accepted step and the one before, \f$E_{n-1}\f$, the next
   * step is \f$\Delta t_{n+1} = \Delta t_n / q\f$ with
   * \f[
   *   q = \frac{1}{0.9}\, \frac{E_n^{\beta_1}}{E_{n-1}^{\beta_2}}\,, \qquad \beta_1 = \frac{0.7}{p}\,, \quad
   *   \beta_2 = \frac{0.4}{p}\,,
   * \f]
   * where \f$p\f$ is the order of the method, \f$q\f$ is clipped to [0.1, 5] and set to 1 within [0.99, 1.2].
   * \f$E_{n-1}\f$ is floored at \f$10^{-4}\f$. A failed step is retried with at most half the step size. The
   * coefficients are close to the PI defaults of OrdinaryDiffEq.jl. For the Newton-based steppers \f$E\f$ is Newton's
   * convergence measure, not a time discretization error estimate; see Newton.
   *
   * @tparam NEWT the solver; must provide set_ignore_nonconv(bool) and get_error().
   */
  template <typename NEWT> class TC_PI : public TC_Default<NEWT>
  {
  public:
    using Base = TC_Default<NEWT>;
    TC_PI(NEWT &newton_, unsigned int alg_order_, double t_, double max_t_, double dt_, double min_dt_, double max_dt_,
          double output_dt_)
        : Base(newton_, alg_order_, t_, max_t_, dt_, min_dt_, max_dt_, output_dt_)
    {
      // Close to the PI defaults of OrdinaryDiffEq.jl, with a dead band [qsteady_min, qsteady_max] in which the
      // step size is kept.
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

    /// Update q from the current error.
    void get_q()
    {
      const double EEst = newton.get_error();
      q11 = std::pow(EEst, beta1);
      q = q11 / std::pow(qold, beta2);
      q = std::max(1. / qmax, std::min(1. / qmin, q / gamma));
    }

    /// Advance the time and suggest the next step from the current and previous error.
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
