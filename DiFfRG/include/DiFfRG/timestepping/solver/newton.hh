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
   * @brief A newton solver, using local error estimates for each vector component.
   *
   * @tparam VectorType_ Vector type used by the simulation (i.e. any dealii vector type)
   */
  template <typename VectorType_> class Newton
  {
  public:
    using VectorType = VectorType_;

    Newton(double abstol_ = 1e-10, double reltol_ = 1e-6, double assemble_threshold_ = 0., uint max_steps_ = 21,
           uint n_stepsize_iterations_ = 21)
        : abstol(abstol_), reltol(reltol_), assemble_threshold(assemble_threshold_), max_steps(max_steps_),
          n_stepsize_iterations(n_stepsize_iterations_), ignore_nonconv(false)
    {
    }

    void set_report_port(ReportPort port) { report_port = std::move(port); }

    /**
     * @brief Should it be necessary, one can force a recalculation of the jacobian hereby.
     */
    void reinit(const VectorType &u) { update_jacobian(u); }

    /**
     * @brief Perform a newton iteration on the input vector.
     *
     * @param iterate the input vector which contains the initial guess and later the solution.
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
     * @brief Set the jacobian recalculation threshold, i.e. below which reduction we force a jacobian update.
     *
     * @return double the old threshold
     */
    double threshold(const double thr)
    {
      const double t = assemble_threshold;
      assemble_threshold = thr;
      return t;
    }

    const VectorType &get_residual() const { return residual_vector; }
    const VectorType &get_initial() const { return initial_vector; }
    const VectorType &get_iterate() const { return iterate_vector; }

    double average_time_newton() const
    {
      double t = 0.;
      double n = timings_newton.size();
      for (const auto &t_ : timings_newton)
        t += t_ / n;
      return t;
    }
    uint num_newton_calls() const { return timings_newton.size(); }

    /**
     * @brief Make the newton algorithm return a success even if the evolution did not fulfill the convergence
     * criterion. The iteration always fails if the solution contains NaNs or infinities.
     *
     * @param x
     */
    void set_ignore_nonconv(bool x) { ignore_nonconv = x; }

    std::function<void(VectorType &, const VectorType &)> residual;
    std::function<void(VectorType &, const VectorType &)> lin_solve;
    std::function<void(const VectorType &)> update_jacobian;

    /**
     * @brief Get the latest error
     */
    double get_error() { return resnorm; }

    /**
     * @brief Get the number of steps taken in the latest iteration
     */
    uint get_step() { return step; }

    /**
     * @brief Get the number of jacobians created in the latest iteration
     */
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

    /**
     * @brief Calculate the current error estimate
     *
     * @return double
     */
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

    /**
     * @brief Check if the iteration should go on.
     *
     * @return true
     * @return false
     */
    bool check()
    {
      if (step == 0) return true;
      // Convergence is tested before the step limit, so that the last allowed step can still converge.
      converged = get_EEst() <= 1.;
      return !converged && step <= max_steps;
    }
  };
} // namespace DiFfRG
