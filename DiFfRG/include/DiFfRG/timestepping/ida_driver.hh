#pragma once

// external libraries
#include <deal.II/base/index_set.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nonlinearsolver.h>

// standard library
#include <exception>
#include <functional>
#include <vector>

namespace DiFfRG
{
  /// Settings of IDADriver.
  struct IDAParameters {
    double initial_time = 0.;
    double final_time = 1.;
    double initial_step_size = 1e-4;
    /// solve_dae() calls output_step() at multiples of this in time.
    double output_period = 1e-1;
    /// solve_dae() also asks solver_should_restart() at multiples of this in time, between output times (e.g. the
    /// mesh adaptor's AbstractAdaptor::adaptation_period()); 0: only at output times.
    double restart_check_period = 0.;
    /// Bounds on the step size; 0 means unbounded.
    double minimal_step_size = 0.;
    double maximal_step_size = 0.;
    unsigned int maximum_order = 5;
    unsigned int maximum_newton_iterations = 3;
    double absolute_tolerance = 1e-13;
    double relative_tolerance = 1e-7;
    /// Continue IDA at its order and step size after a restart, if IDADriver::warm_restart_available.
    bool warm_restart = true;
    /// Up to this many steps after its assembly, a Jacobian is adapted to a new alpha by
    /// IDADriver::update_jacobian_alpha instead of being reassembled; 0 never does. Longer windows let
    /// \f$\partial F/\partial y\f$ go stale: at 20, ONfiniteT KT without adaptation needed 25% more steps.
    unsigned int jacobian_reuse_steps = 5;
  };

  /**
   * @brief Driver of the SUNDIALS IDA(S) DAE integrator, used by the IDA timesteppers.
   *
   * The interface follows deal.II's SUNDIALS::IDA: the user sets the callbacks below and calls solve_dae(). A
   * callback signals a recoverable failure, upon which IDA retries with a smaller step, by throwing
   * dealii::StandardExceptions::RecoverableUserCallbackError; any other exception aborts the integration.
   *
   * Differences to deal.II's wrapper:
   * - The step size is bounded to [minimal_step_size, maximal_step_size].
   * - The Newton iteration measures its convergence rate in every step. IDA's own test carries the rate over
   *   from earlier steps and then accepts a step after a single Newton iteration. With a Jacobian that is no
   *   longer current, the true rate is worse than the carried one, and the under-converged corrector pollutes
   *   IDA's local error and order estimates: IDA then takes lower orders and smaller steps, which costs more
   *   residuals than the extra iteration. Here a step needs two iterations unless its first correction is
   *   negligible.
   * - Statistics and the error test vectors are accessible.
   * - IDA asks for a new Jacobian whenever a change of step size or order moved \f$\alpha\f$ out of
   *   \f$[0.6, 1.67]\f$ times its value at the last setup. With update_jacobian_alpha set, such a Jacobian is not
   *   reassembled, only changed to the new \f$\alpha\f$, keeping \f$\partial F/\partial y\f$ from the last
   *   assembly. It is reassembled after a Newton failure since that assembly, and after
   *   Parameters::jacobian_reuse_steps steps.
   *
   * At the start the algebraic components of the solution and the derivative of the differential ones are made
   * consistent (IDACalcIC with IDA_YA_YDP_INIT). A restart (solver_should_restart, e.g. after a mesh adaptation) by
   * default continues warm: IDA keeps its order and step size, with its step history (history()) transferred to the
   * new mesh by the caller. Without warm restarts (Parameters::warm_restart, warm_restart_available) IDA starts
   * over at order 1 with a fresh consistent initial condition, which costs tens of jacobians per restart.
   */
  template <typename VectorType> class IDADriver
  {
  public:
    using Parameters = IDAParameters;

    /// IDA's counters since the last (re)initialization, and its current step sizes.
    struct Statistics {
      bool valid = false;
      long int num_steps = 0;
      long int num_residual_evaluations = 0;
      long int num_linear_solver_setups = 0;
      /// Of the linear solver setups, those done by update_jacobian_alpha instead of setup_jacobian.
      long int num_jacobian_alpha_updates = 0;
      long int num_error_test_failures = 0;
      long int num_nonlinear_iterations = 0;
      long int num_nonlinear_convergence_failures = 0;
      long int num_step_solve_failures = 0;
      double last_step_size = 0.;
      double current_step_size = 0.;
      double current_time = 0.;
    };

    IDADriver(const Parameters &parameters);
    ~IDADriver();
    IDADriver(const IDADriver &) = delete;
    IDADriver &operator=(const IDADriver &) = delete;

    /**
     * @brief Integrate from Parameters::initial_time to Parameters::final_time, starting from @p solution and
     * @p solution_dot, which hold the final state on return.
     *
     * If IDA fails (no convergence or error test failures down to the minimal step), it is restarted once, cold, at
     * its last accepted state, with a warning: a step history that no longer fits the residual (e.g. after the
     * explicit variables of a hybrid stepper were updated) is a reason for that which a fresh start removes. A second
     * failure before the next stop time (output, solver_should_restart() check, or the end) throws.
     *
     * @return the number of steps since the last (re)initialization.
     */
    unsigned int solve_dae(VectorType &solution, VectorType &solution_dot);

    Statistics get_statistics() const;

    /// IDA's estimate of the local error of the last step and the weights of its error norm.
    bool get_error_test_vectors(VectorType &estimated_local_errors, VectorType &error_weights) const;

    /// Compute the residual \f$F(t, y, \dot y)\f$.
    std::function<void(const double t, const VectorType &y, const VectorType &y_dot, VectorType &res)> residual;

    /// Prepare solve_with_jacobian for the Jacobian \f$\partial F/\partial y + \alpha\,\partial F/\partial\dot y\f$.
    std::function<void(const double t, const VectorType &y, const VectorType &y_dot, const double alpha)>
        setup_jacobian;

    /**
     * @brief Optional: change the Jacobian of the last setup_jacobian to the factor @p alpha without reassembling,
     * i.e. add \f$(\alpha - \alpha_{\rm last})\,\partial F/\partial\dot y\f$ (at the state of that assembly) and
     * prepare solve_with_jacobian. Returns false if it cannot, then setup_jacobian is called.
     */
    std::function<bool(const double t, const VectorType &y, const VectorType &y_dot, const double alpha)>
        update_jacobian_alpha;

    /// Solve the system with the Jacobian of the last setup_jacobian, to the tolerance @p tolerance if iterative.
    std::function<void(const VectorType &rhs, VectorType &dst, const double tolerance)> solve_with_jacobian;

    /// Called at the start, at every multiple of the output period and at the end.
    std::function<void(const double t, const VectorType &sol, const VectorType &sol_dot, const unsigned int step)>
        output_step;

    /// Called after every output; returning true (e.g. after remeshing @p sol and @p sol_dot) reinitializes IDA.
    std::function<bool(const double t, VectorType &sol, VectorType &sol_dot)> solver_should_restart;

    /**
     * @brief Optional: called after every accepted step, with its state; returning true restarts IDA there cold,
     * i.e. without its step history, as is needed where the DAE itself is discontinuous in time.
     *
     * Setting it makes IDA run step by step up to each output time instead of interpolating to it.
     */
    std::function<bool(const double t, const VectorType &sol, const VectorType &sol_dot)> step_needs_restart;

    /// Optional: receives warnings, e.g. that a failed step was retried after a cold restart (see solve_dae()).
    std::function<void(const std::string &message)> warn;

    /// Optional: called before a failed step is retried from a cold restart at the last accepted state (t, sol), e.g.
    /// so that a hybrid stepper restarts its explicit side there as well.
    std::function<void(const double t, const VectorType &sol)> before_retry;

    /// The differential components; all others are algebraic and excluded from the error test.
    std::function<dealii::IndexSet()> differential_components;

    /// Optional per-component absolute tolerances.
    std::function<VectorType &()> get_local_tolerances;

    /**
     * @brief During solver_should_restart: IDA's step history (its divided-difference vectors), which a mesh
     * adaptation has to transfer together with the solution for IDA to continue warm, at its order and step size.
     */
    std::vector<VectorType *> history() { return warm ? history_pointers : std::vector<VectorType *>{}; }

    /// Whether this build can restart IDA warm: it needs IDA's internal headers from the dependency bundle.
    static const bool warm_restart_available;

  private:
    void reset(const double t, const double h, VectorType &solution, VectorType &solution_dot, const bool restart);
    void free_ida();
    /// Rethrow an exception a callback left behind, unless it is a recoverable one: IDA has either recovered from
    /// that, or reports its failure through its return status.
    void handle_pending_exception();

    /// IDA's Jacobian function: update_jacobian_alpha if allowed and possible, else setup_jacobian.
    int jacobian(const double t, const double alpha, const VectorType &y, const VectorType &y_dot);
    bool computing_initial_condition = false;
    /// Whether setup_jacobian was called since the last (re)initialization, and IDA's step count and number of Newton
    /// failures then.
    bool assembled = false;
    long int assembly_step = 0, assembly_newton_failures = 0, alpha_updates = 0;

    /// State of the Newton convergence test, see the class documentation.
    struct NewtonTest {
      double first_norm = 0.;
    };
    static int newton_convergence_test(SUNNonlinearSolver NLS, N_Vector ycor, N_Vector delta, double tolerance,
                                       N_Vector weights, void *test);

    /// IDA's state at a restart, see history(); restored by reset() so that IDA continues where it was: the
    /// divided-difference vectors (transferred to a new mesh by the caller) and the scalars of IDA's step control.
    void capture_history();
    void restore_history(const double t);
    struct HistoryScalars {
      int kk, kused, knew, phase, ns;
      long int nst;
      double hh, hused, tn, cjlast, cjold, cjratio, ss, eta, cj, oldnrm, h0u, epsNewt, toldel;
      double psi[6], alpha[6], beta[6], sigma[6], gamma[6];
    } history_scalars;
    std::vector<VectorType> history_vectors;
    std::vector<VectorType *> history_pointers;
    bool warm = false;

    const Parameters prm;
    SUNContext ctx = nullptr;
    void *ida_mem = nullptr;
    SUNLinearSolver linear_solver = nullptr;
    SUNMatrix matrix = nullptr;
    SUNNonlinearSolver newton = nullptr;
    NewtonTest newton_test;
    std::exception_ptr pending_exception;
  };
  namespace internal
  {
    /// The IDADriver settings from the implicit parameters of a timestepper (see abstract_timestepper.hh).
    template <typename ImplicitParameters>
    IDAParameters ida_parameters(const ImplicitParameters &impl, const double t_start, const double t_stop,
                                 const double output_dt, const double restart_check_period)
    {
      return IDAParameters{.initial_time = t_start,
                           .final_time = t_stop,
                           .initial_step_size = impl.dt,
                           .output_period = output_dt,
                           .restart_check_period = restart_check_period,
                           .minimal_step_size = impl.minimal_dt,
                           .maximal_step_size = impl.maximal_dt,
                           .maximum_newton_iterations = impl.max_non_linear_iterations,
                           .absolute_tolerance = impl.abs_tol,
                           .relative_tolerance = impl.rel_tol,
                           .warm_restart = impl.warm_restart,
                           .jacobian_reuse_steps = impl.reuse_jacobian ? IDAParameters{}.jacobian_reuse_steps : 0u};
    }
  } // namespace internal
} // namespace DiFfRG
