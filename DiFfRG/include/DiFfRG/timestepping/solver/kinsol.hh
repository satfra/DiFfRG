#pragma once

// standard library
#include <functional>

// external libraries
#include <deal.II/sundials/kinsol.h>

namespace DiFfRG
{
  using namespace dealii;
  /**
   * @brief Newton's method with line search from SUNDIALS KINSOL, with the same callbacks as Newton.
   *
   * KINSOL stops when the scaled residual satisfies \f$\max_i |F_i(u)| \le\f$ `abstol`. Afterwards the
   * root-mean-square residual is checked against \f$\max(\text{abstol}, \text{reltol} \cdot \text{RMS}(u))\f$:
   * if KINSOL reported a failure and this check fails, it throws; if KINSOL returned normally but the check fails,
   * KINSOL is run once more from there.
   *
   * `assemble_threshold` and `n_stepsize_iterations` are accepted for compatibility with Newton but not used.
   *
   * @tparam VectorType_ a serial deal.II vector type.
   */
  template <typename VectorType_> class KINSOL
  {
  public:
    using VectorType = VectorType_;

    KINSOL(double abstol_ = 1e-10, double reltol_ = 1e-6, double assemble_threshold_ = 0., unsigned int max_steps_ = 21,
           unsigned int n_stepsize_iterations_ = 21);

    /// Build the Jacobian at @p u now.
    void reinit(const VectorType &u);

    /// Solve \f$F(u) = 0\f$; @p iterate is the initial guess on entry and the solution on return.
    void operator()(VectorType &iterate);

    /// Has no effect; kept for compatibility with Newton. @return the previous value.
    double threshold(const double thr);

    /// \f$F\f$ at the last evaluated point.
    const VectorType &get_residual() const;
    /// The initial guess of the last solve.
    const VectorType &get_initial() const;
    /// The last iterate.
    const VectorType &get_iterate() const;

    /// Mean wall time of a solve in seconds.
    double average_time_newton() const;
    /// Number of solves so far.
    unsigned int num_newton_calls() const;

    /// Accept the iterate whenever KINSOL fails, without checking the residual.
    void set_ignore_nonconv(bool x);

    /// residual(F, u): write \f$F(u)\f$ into F.
    std::function<void(VectorType &, const VectorType &)> residual;
    /// lin_solve(du, F): solve \f$J\, du = F\f$ with the current Jacobian.
    std::function<void(VectorType &, const VectorType &)> lin_solve;
    /// update_jacobian(u): build the Jacobian at u.
    std::function<void(const VectorType &)> update_jacobian;

    /// The convergence measure \f$E\f$ of Newton at the last iterate.
    double get_error();

    /// Number of KINSOL iterations of the last solve.
    unsigned int get_step();

    /// Number of Jacobians built during the last solve.
    unsigned int get_jacobians();

  private:
    VectorType residual_vector;
    VectorType initial_vector;
    VectorType iterate_vector;
    VectorType eest_tmp;
    double abstol, reltol, assemble_threshold;
    unsigned int max_steps, n_stepsize_iterations;

    bool converged;
    bool ignore_nonconv;
    double resnorm;
    unsigned int step, jacobians;

    std::vector<double> timings_newton;

    /// The convergence measure \f$E\f$ of Newton at the current iterate.
    double get_EEst();

    /// Unused.
    bool check();

    std::shared_ptr<SUNDIALS::KINSOL<VectorType>> kinsol;
  };
} // namespace DiFfRG