#pragma once

#include <DiFfRG/common/run_reporter.hh>

// standard library
#include <functional>

// external libraries
#include <deal.II/base/config.h>
#include <deal.II/base/timer.h>

namespace DiFfRG
{
  using namespace dealii;

  /**
   * @brief Newton's method for \f$F(u) = 0\f$, used by TimeStepperImplicitEuler and TimeStepperTRBDF2.
   *
   * The caller sets three callbacks: residual(F, u) computes \f$F(u)\f$, update_jacobian(u) assembles (and e.g.
   * factorizes) \f$J = \partial F/\partial u\f$, lin_solve(du, F) solves \f$J\, du = F\f$. An iteration sets
   * \f$u \to u - du\f$; if that does not reduce \f$\|F\|_2\f$, the step is halved up to `n_stepsize_iterations`
   * times. Convergence is measured by
   * \f[
   *   E = \sqrt{\frac{1}{N} \sum_{i=1}^N \left(\frac{F_i(u)}{\text{abstol} + \text{reltol}\,\max(|u^0_i|, |u_i|)}
   *   \right)^2}\,,
   * \f]
   * with the initial guess \f$u^0\f$ and \f$N\f$ unknowns; the iteration has converged when \f$E \le 1\f$.
   *
   * The Jacobian is not rebuilt in the first iteration (the one from the previous call, or from reinit(), is reused);
   * later iterations rebuild it whenever \f$E_\text{new} / E_\text{old} \ge\f$ `assemble_threshold`.
   *
   * @tparam VectorType_ a serial deal.II vector type.
   */
  template <typename VectorType_> class Newton
  {
  public:
    using VectorType = VectorType_;

    /**
     * @param abstol_ absolute tolerance in \f$E\f$.
     * @param reltol_ relative tolerance in \f$E\f$.
     * @param assemble_threshold_ rebuild the Jacobian if \f$E_\text{new} / E_\text{old} \ge\f$ this.
     * @param max_steps_ maximal number of iterations.
     * @param n_stepsize_iterations_ maximal number of step halvings per iteration.
     */
    Newton(double abstol_ = 1e-10, double reltol_ = 1e-6, double assemble_threshold_ = 0., uint max_steps_ = 21,
           uint n_stepsize_iterations_ = 21)
        : abstol(abstol_), reltol(reltol_), assemble_threshold(assemble_threshold_), max_steps(max_steps_),
          n_stepsize_iterations(n_stepsize_iterations_), ignore_nonconv(false)
    {
    }

    void set_report_port(ReportPort port) { report_port = std::move(port); }

    /// Build the Jacobian at @p u now, e.g. before the first solve or after the problem size changed.
    void reinit(const VectorType &u) { update_jacobian(u); }

    /**
     * @brief Solve \f$F(u) = 0\f$.
     *
     * Throws if the residual becomes non-finite, a linear solve fails, or \f$E > 1\f$ after `max_steps` iterations
     * (unless set_ignore_nonconv() was set).
     *
     * @param iterate initial guess on entry, solution on return (unchanged if it throws).
     */
    void operator()(VectorType &iterate)
    {
      Timer timer;

      converged = false;
      step = 0;
      resnorm = 1.;
      jacobians = 0;

      initial_vector = iterate;
      iterate_vector = iterate;
      VectorType &u = iterate_vector;

      Du.reinit(u);
      residual_vector.reinit(u);

      // fill res with (f(u), v)
      residual(residual_vector, u);
      if (!std::isfinite(residual_vector.l2_norm())) throw std::runtime_error("Initial residual non-finite!");
      resnorm = get_EEst();
      double old_residual = 0.;
      double res_l2 = residual_vector.l2_norm();
      double assembly_time = 0.;

      while (check()) {
        step++;

        // assemble (Df(u), v)
        if ((step > 1) && (resnorm / old_residual >= assemble_threshold)) {
          Timer timer_assembly;
          update_jacobian(u);
          assembly_time += timer_assembly.wall_time();
          jacobians++;
        }

        Du.reinit(u);
        try {
          lin_solve(Du, residual_vector);
        } catch (std::exception &) {
          throw std::runtime_error("Linear solve failed!");
        }

        u.add(-1., Du);
        old_residual = resnorm;
        double old_res_l2 = res_l2;

        residual(residual_vector, u);
        res_l2 = residual_vector.l2_norm();
        if (!std::isfinite(res_l2)) throw std::runtime_error("Residual non-finite!");

        // Simple line search: use cheap l2_norm comparisons instead of full get_EEst()
        uint step_size = 0;
        while (res_l2 >= old_res_l2) {
          ++step_size;
          if (step_size > n_stepsize_iterations) {
            break;
          }
          u.add(1. / double(1 << step_size), Du);
          residual(residual_vector, u);
          res_l2 = residual_vector.l2_norm();
        }

        // compute full error estimate once after line search
        resnorm = get_EEst();
      }

      timings_newton.push_back(timer.wall_time() - assembly_time);

      // in case of failure: throw exception
      if (!(converged || ignore_nonconv) || !std::isfinite(resnorm)) {
        throw std::runtime_error("Newton did not converge!");
      }
      // otherwise exit as normal
      else
        iterate = iterate_vector;
    }

    /**
     * @brief Set `assemble_threshold`: the Jacobian is rebuilt when \f$E_\text{new} / E_\text{old} \ge\f$ this in an
     * iteration.
     *
     * @return the previous threshold.
     */
    double threshold(const double thr)
    {
      const double t = assemble_threshold;
      assemble_threshold = thr;
      return t;
    }

    /// \f$F\f$ at the last iterate.
    const VectorType &get_residual() const { return residual_vector; }
    /// The initial guess of the last solve.
    const VectorType &get_initial() const { return initial_vector; }
    /// The last iterate.
    const VectorType &get_iterate() const { return iterate_vector; }

    /// Mean wall time of a solve in seconds, without Jacobian assembly.
    double average_time_newton() const
    {
      double t = 0.;
      double n = timings_newton.size();
      for (const auto &t_ : timings_newton)
        t += t_ / n;
      return t;
    }
    /// Number of solves so far.
    uint num_newton_calls() const { return timings_newton.size(); }

    /// Accept the last iterate even if \f$E > 1\f$ after `max_steps` iterations. Non-finite residuals still throw.
    void set_ignore_nonconv(bool x) { ignore_nonconv = x; }

    /// residual(F, u): write \f$F(u)\f$ into F.
    std::function<void(VectorType &, const VectorType &)> residual;
    /// lin_solve(du, F): solve \f$J\, du = F\f$ with the current Jacobian.
    std::function<void(VectorType &, const VectorType &)> lin_solve;
    /// update_jacobian(u): build the Jacobian at u.
    std::function<void(const VectorType &)> update_jacobian;

    /// The convergence measure \f$E\f$ after the last solve.
    double get_error() { return resnorm; }

    /// Number of iterations of the last solve.
    uint get_step() { return step; }

    /// Number of Jacobians built during the last solve.
    uint get_jacobians() { return jacobians; }

  private:
    VectorType residual_vector;
    VectorType initial_vector;
    VectorType iterate_vector;
    VectorType Du;
    VectorType eest_tmp;
    double abstol, reltol, assemble_threshold;
    uint max_steps, n_stepsize_iterations;

    bool converged;
    bool ignore_nonconv;
    double resnorm;
    uint step, jacobians;

    std::vector<double> timings_newton;
    ReportPort report_port;

    /// The convergence measure \f$E\f$ at the current iterate.
    double get_EEst()
    {
      const VectorType &err = residual_vector;
      const VectorType &u_prev = initial_vector;
      const VectorType &u = iterate_vector;

      eest_tmp.reinit(u);

      using std::abs, std::max, std::sqrt;
      tbb::parallel_for(tbb::blocked_range<uint>(0, eest_tmp.size()), [&](tbb::blocked_range<uint> r) {
        for (uint n = r.begin(); n < r.end(); ++n) {
          eest_tmp[n] = abs(err[n]) / (abstol + max(abs(u_prev[n]), abs(u[n])) * reltol);
        }
      });
      return eest_tmp.l2_norm() / sqrt(eest_tmp.size());
    }

    /// Whether to iterate further; sets `converged`.
    bool check()
    {
      if (step == 0) return true;
      // Convergence is tested before the step limit, so that the last allowed step can still converge.
      converged = get_EEst() <= 1.;
      return !converged && step < max_steps;
    }
  };
} // namespace DiFfRG
