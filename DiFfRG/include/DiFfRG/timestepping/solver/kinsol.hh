#pragma once

// standard library
#include <functional>

// external libraries
#include <deal.II/sundials/kinsol.h>

namespace DiFfRG
{
  using namespace dealii;
  /**
   * @brief A newton solver, using local error estimates for each vector component.
   *
   * @tparam VectorType_ Vector type used by the simulation (i.e. any dealii vector type)
   */
  template <typename VectorType_> class KINSOL
  {
  public:
    using VectorType = VectorType_;

    KINSOL(double abstol_ = 1e-10, double reltol_ = 1e-6, double assemble_threshold_ = 0., unsigned int max_steps_ = 21,
           unsigned int n_stepsize_iterations_ = 21);

    /**
     * @brief Should it be necessary, one can force a recalculation of the jacobian hereby.
     */
    void reinit(const VectorType &u);

    /**
     * @brief Perform a newton iteration on the input vector.
     *
     * @param iterate the input vector which contains the initial guess and later the solution.
     */
    void operator()(VectorType &iterate);

    /**
     * @brief Set the jacobian recalculation threshold, i.e. below which reduction we force a jacobian update.
     *
     * @return double the old threshold
     */
    double threshold(const double thr);

    const VectorType &get_residual() const;
    const VectorType &get_initial() const;
    const VectorType &get_iterate() const;

    double average_time_newton() const;
    unsigned int num_newton_calls() const;

    /**
     * @brief Make the newton algorithm return a success even if the evolution did not fulfill the convergence
     * criterion. The iteration always fails if the solution contains NaNs or infinities.
     *
     * @param x
     */
    void set_ignore_nonconv(bool x);

    std::function<void(VectorType &, const VectorType &)> residual;
    std::function<void(VectorType &, const VectorType &)> lin_solve;
    std::function<void(const VectorType &)> update_jacobian;

    /**
     * @brief Get the latest error
     */
    double get_error();

    /**
     * @brief Get the number of steps taken in the latest iteration
     */
    unsigned int get_step();

    /**
     * @brief Get the number of jacobians created in the latest iteration
     */
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

    /**
     * @brief Calculate the current error estimate
     *
     * @return double
     */
    double get_EEst();

    /**
     * @brief Check if the iteration should go on.
     *
     * @return true
     * @return false
     */
    bool check();

    std::shared_ptr<SUNDIALS::KINSOL<VectorType>> kinsol;
  };
} // namespace DiFfRG