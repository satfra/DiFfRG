// external libraries
#include <deal.II/base/exceptions.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operation.h>
#include <deal.II/sundials/n_vector.h>
#include <deal.II/sundials/n_vector.templates.h>
#include <deal.II/sundials/utilities.h>
#ifdef DEAL_II_WITH_PETSC
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#endif
#include <idas/idas.h>
#include <idas/idas_ls.h>
#ifdef DIFFRG_IDA_WARM_RESTART
// IDA's internal state, for the warm restart; installed by the dependency bundle (dependencies/sundials).
#include <idas_impl.h>
// Exported by libsundials_idas, but declared in no header.
extern "C" int IDAInitialSetup(IDAMem IDA_mem);
#endif
#include <sunnonlinsol/sunnonlinsol_newton.h>

// standard library
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

// DiFfRG
#include <DiFfRG/timestepping/ida_driver.hh>

namespace DiFfRG
{
  using dealii::SUNDIALS::internal::make_nvector_view;
  using dealii::SUNDIALS::internal::unwrap_nvector;
  using dealii::SUNDIALS::internal::unwrap_nvector_const;
  using dealii::SUNDIALS::Utilities::call_and_possibly_capture_exception;

#ifdef DIFFRG_IDA_WARM_RESTART
  template <typename VectorType> const bool IDADriver<VectorType>::warm_restart_available = true;
#else
  template <typename VectorType> const bool IDADriver<VectorType>::warm_restart_available = false;
#endif

  namespace
  {
    void check(const int status, const char *call)
    {
      if (status < 0)
        throw std::runtime_error(std::string("IDA: ") + call + " failed with " + IDAGetReturnFlagName(status));
    }
  } // namespace

  template <typename VectorType> IDADriver<VectorType>::IDADriver(const Parameters &parameters) : prm(parameters)
  {
    check(SUNContext_Create(SUN_COMM_NULL, &ctx), "SUNContext_Create");
  }

  template <typename VectorType> IDADriver<VectorType>::~IDADriver()
  {
    free_ida();
    SUNContext_Free(&ctx);
  }

  template <typename VectorType> void IDADriver<VectorType>::free_ida()
  {
    if (ida_mem != nullptr) IDAFree(&ida_mem);
    if (newton != nullptr) SUNNonlinSolFree(newton);
    if (linear_solver != nullptr) SUNLinSolFreeEmpty(linear_solver);
    if (matrix != nullptr) SUNMatFreeEmpty(matrix);
    newton = nullptr;
    linear_solver = nullptr;
    matrix = nullptr;
  }

  template <typename VectorType>
  int IDADriver<VectorType>::newton_convergence_test(SUNNonlinearSolver NLS, N_Vector /*ycor*/, N_Vector delta,
                                                     double tolerance, N_Vector weights, void *test)
  {
    auto &state = *static_cast<NewtonTest *>(test);
    const double norm = N_VWrmsNorm(delta, weights);
    int iteration = 0;
    SUNNonlinSolGetCurIter(NLS, &iteration);
    // A first correction this small leaves a Newton error far below the tolerance, whatever the rate.
    if (iteration == 0) {
      state.first_norm = norm;
      return norm <= 1e-2 * tolerance ? SUN_SUCCESS : SUN_NLS_CONTINUE;
    }
    const double rate = std::pow(norm / state.first_norm, 1. / iteration);
    if (rate > 0.9) return SUN_NLS_CONV_RECVR;
    // rate/(1-rate) * norm bounds the remaining Newton error for a linearly converging iteration.
    return rate / (1. - rate) * norm <= tolerance ? SUN_SUCCESS : SUN_NLS_CONTINUE;
  }

  template <typename VectorType> void IDADriver<VectorType>::handle_pending_exception()
  {
    if (!pending_exception) return;
    try {
      std::rethrow_exception(pending_exception);
    } catch (const dealii::StandardExceptions::RecoverableUserCallbackError &) {
      pending_exception = nullptr;
    } catch (...) {
      pending_exception = nullptr;
      throw;
    }
  }

  template <typename VectorType>
  void IDADriver<VectorType>::reset(const double t, const double h, VectorType &solution, VectorType &solution_dot,
                                    const bool restart)
  {
    free_ida();
    ida_mem = IDACreate(ctx);
    assembled = false;
    alpha_updates = 0;

    auto yy = make_nvector_view(solution, ctx);
    auto yp = make_nvector_view(solution_dot, ctx);

    check(IDAInit(
              ida_mem,
              [](sunrealtype tt, N_Vector y, N_Vector y_dot, N_Vector res, void *user_data) -> int {
                auto &driver = *static_cast<IDADriver *>(user_data);
                return call_and_possibly_capture_exception(
                    driver.residual, driver.pending_exception, tt, *unwrap_nvector_const<VectorType>(y),
                    *unwrap_nvector_const<VectorType>(y_dot), *unwrap_nvector<VectorType>(res));
              },
              t, yy, yp),
          "IDAInit");
    check(IDASetUserData(ida_mem, this), "IDASetUserData");

    if (get_local_tolerances) {
      auto abs_tols = make_nvector_view(get_local_tolerances(), ctx);
      check(IDASVtolerances(ida_mem, prm.relative_tolerance, abs_tols), "IDASVtolerances");
    } else
      check(IDASStolerances(ida_mem, prm.relative_tolerance, prm.absolute_tolerance), "IDASStolerances");

    // Differential components are 1, algebraic ones 0. The algebraic ones are excluded from the error test.
    {
      VectorType id(solution);
      id = 0.;
      const dealii::IndexSet differential =
          differential_components ? differential_components() : solution.locally_owned_elements();
      for (const auto i : differential)
        id[i] = 1.;
      id.compress(dealii::VectorOperation::insert);
      auto id_view = make_nvector_view(id, ctx);
      check(IDASetId(ida_mem, id_view), "IDASetId");
    }
    check(IDASetSuppressAlg(ida_mem, SUNTRUE), "IDASetSuppressAlg");

    check(IDASetInitStep(ida_mem, h), "IDASetInitStep");
    check(IDASetStopTime(ida_mem, prm.final_time), "IDASetStopTime");
    check(IDASetMaxNumSteps(ida_mem, std::numeric_limits<long int>::max()), "IDASetMaxNumSteps");
    if (prm.minimal_step_size > 0) check(IDASetMinStep(ida_mem, prm.minimal_step_size), "IDASetMinStep");
    if (prm.maximal_step_size > 0) check(IDASetMaxStep(ida_mem, prm.maximal_step_size), "IDASetMaxStep");
    check(IDASetMaxOrd(ida_mem, prm.maximum_order), "IDASetMaxOrd");

    // Newton iteration with our convergence test; see the class documentation.
    newton = SUNNonlinSol_Newton(yy, ctx);
    check(IDASetNonlinearSolver(ida_mem, newton), "IDASetNonlinearSolver");
    check(SUNNonlinSolSetConvTestFn(newton, newton_convergence_test, &newton_test), "SUNNonlinSolSetConvTestFn");
    check(IDASetMaxNonlinIters(ida_mem, prm.maximum_newton_iterations), "IDASetMaxNonlinIters");

    // The linear solver forwards to solve_with_jacobian. It is declared matrix-iterative, so that IDA hands it a
    // tolerance (used by the Krylov solvers) and still scales its corrections for a changed alpha.
    linear_solver = SUNLinSolNewEmpty(ctx);
    linear_solver->content = this;
    linear_solver->ops->gettype = [](SUNLinearSolver) -> SUNLinearSolver_Type {
      return SUNLINEARSOLVER_MATRIX_ITERATIVE;
    };
    linear_solver->ops->solve = [](SUNLinearSolver LS, SUNMatrix, N_Vector x, N_Vector b, sunrealtype tol) -> int {
      auto &driver = *static_cast<IDADriver *>(LS->content);
      return call_and_possibly_capture_exception(driver.solve_with_jacobian, driver.pending_exception,
                                                 *unwrap_nvector_const<VectorType>(b), *unwrap_nvector<VectorType>(x),
                                                 tol);
    };
    linear_solver->ops->resid = [](SUNLinearSolver) -> N_Vector { return nullptr; };
    linear_solver->ops->numiters = [](SUNLinearSolver) -> int { return 1; };
    // IDA only calls the Jacobian function if a matrix is attached; the matrix itself stays empty.
    matrix = SUNMatNewEmpty(ctx);
    matrix->ops->getid = [](SUNMatrix) -> SUNMatrix_ID { return SUNMATRIX_CUSTOM; };
    check(IDASetLinearSolver(ida_mem, linear_solver, matrix), "IDASetLinearSolver");
    check(IDASetJacFn(ida_mem,
                      [](sunrealtype tt, sunrealtype cj, N_Vector y, N_Vector y_dot, N_Vector, SUNMatrix,
                         void *user_data, N_Vector, N_Vector, N_Vector) -> int {
                        return static_cast<IDADriver *>(user_data)->jacobian(
                            tt, cj, *unwrap_nvector_const<VectorType>(y), *unwrap_nvector_const<VectorType>(y_dot));
                      }),
          "IDASetJacFn");

    if (restart && warm) {
      restore_history(t);
      return;
    }
    // Consistent initial conditions: algebraic components of y and the derivative of the differential ones.
    check(IDASetMaxNumItersIC(ida_mem, 5), "IDASetMaxNumItersIC");
    computing_initial_condition = true;
    const int status = IDACalcIC(ida_mem, IDA_YA_YDP_INIT, t + h);
    computing_initial_condition = false;
    handle_pending_exception();
    check(status, "IDACalcIC");
    check(IDAGetConsistentIC(ida_mem, yy, yp), "IDAGetConsistentIC");
  }

  template <typename VectorType> void IDADriver<VectorType>::capture_history()
  {
#ifdef DIFFRG_IDA_WARM_RESTART
    auto *m = static_cast<IDAMem>(ida_mem);
    auto &s = history_scalars;
    s.kk = m->ida_kk, s.kused = m->ida_kused, s.knew = m->ida_knew, s.phase = m->ida_phase, s.ns = m->ida_ns;
    s.nst = m->ida_nst;
    s.hh = m->ida_hh, s.hused = m->ida_hused, s.tn = m->ida_tn, s.cjlast = m->ida_cjlast, s.cjold = m->ida_cjold;
    s.cjratio = m->ida_cjratio, s.ss = m->ida_ss, s.eta = m->ida_eta, s.cj = m->ida_cj, s.oldnrm = m->ida_oldnrm;
    s.h0u = m->ida_h0u, s.epsNewt = m->ida_epsNewt, s.toldel = m->ida_toldel;
    for (int j = 0; j < MXORDP1; ++j) {
      s.psi[j] = m->ida_psi[j], s.alpha[j] = m->ida_alpha[j], s.beta[j] = m->ida_beta[j];
      s.sigma[j] = m->ida_sigma[j], s.gamma[j] = m->ida_gamma[j];
    }
    // phi[0..kk+1]: the prediction uses up to kk, order increases read kk + 1.
    const int n = std::min(s.kk + 2, m->ida_maxord + 1);
    history_vectors.resize(n);
    history_pointers.clear();
    for (int j = 0; j < n; ++j) {
      history_vectors[j] = *unwrap_nvector_const<VectorType>(m->ida_phi[j]);
      history_pointers.push_back(&history_vectors[j]);
    }
#endif
  }

  template <typename VectorType> void IDADriver<VectorType>::restore_history([[maybe_unused]] const double t)
  {
#ifdef DIFFRG_IDA_WARM_RESTART
    auto *m = static_cast<IDAMem>(ida_mem);
    check(IDAInitialSetup(m), "IDAInitialSetup");
    m->ida_SetupDone = SUNTRUE;
    for (size_t j = 0; j < history_vectors.size(); ++j)
      *unwrap_nvector<VectorType>(m->ida_phi[j]) = history_vectors[j];
    const auto &s = history_scalars;
    m->ida_kk = s.kk, m->ida_kused = s.kused, m->ida_knew = s.knew, m->ida_phase = s.phase, m->ida_ns = s.ns;
    m->ida_nst = s.nst;
    m->ida_hh = s.hh, m->ida_hused = s.hused, m->ida_tn = s.tn, m->ida_cjlast = s.cjlast, m->ida_cjold = s.cjold;
    m->ida_cjratio = s.cjratio, m->ida_ss = s.ss, m->ida_eta = s.eta, m->ida_cj = s.cj, m->ida_oldnrm = s.oldnrm;
    m->ida_h0u = s.h0u, m->ida_epsNewt = s.epsNewt, m->ida_toldel = s.toldel;
    for (int j = 0; j < MXORDP1; ++j) {
      m->ida_psi[j] = s.psi[j], m->ida_alpha[j] = s.alpha[j], m->ida_beta[j] = s.beta[j];
      m->ida_sigma[j] = s.sigma[j], m->ida_gamma[j] = s.gamma[j];
    }
    m->ida_tretlast = t;
    m->ida_forceSetup = SUNTRUE; // the jacobian of the old mesh is gone
#endif
  }

  template <typename VectorType>
  int IDADriver<VectorType>::jacobian(const double t, const double alpha, const VectorType &y, const VectorType &y_dot)
  {
    long int step = 0, newton_failures = 0, failures_this_solve = 0;
    IDAGetNumSteps(ida_mem, &step);
    IDAGetNumNonlinSolvConvFails(ida_mem, &newton_failures);
    SUNNonlinSolGetNumConvFails(newton, &failures_this_solve);
    // A Newton failure, also one in the current step that IDA retries with a new Jacobian, asks for a current one.
    const bool reuse = update_jacobian_alpha && assembled && !computing_initial_condition && failures_this_solve == 0 &&
                       newton_failures == assembly_newton_failures &&
                       step - assembly_step < static_cast<long int>(prm.jacobian_reuse_steps);
    if (reuse) {
      bool updated = false;
      const int status = call_and_possibly_capture_exception(
          [&](const double t, const VectorType &y, const VectorType &y_dot, const double alpha) {
            updated = update_jacobian_alpha(t, y, y_dot, alpha);
          },
          pending_exception, t, y, y_dot, alpha);
      alpha_updates += status == 0 && updated;
      if (status != 0 || updated) return status;
    }
    assembled = true;
    assembly_step = step;
    assembly_newton_failures = newton_failures;
    return call_and_possibly_capture_exception(setup_jacobian, pending_exception, t, y, y_dot, alpha);
  }

  template <typename VectorType>
  unsigned int IDADriver<VectorType>::solve_dae(VectorType &solution, VectorType &solution_dot)
  {
    double t = prm.initial_time;
    double h = prm.initial_step_size;
    reset(t, h, solution, solution_dot, false);
    if (output_step) output_step(t, solution, solution_dot, 0);

    unsigned int output_number = 0;
    double next_time = prm.initial_time;
    auto fail_if = [&](const int status) {
      if (status < 0) {
        std::ostringstream message;
        message << std::setprecision(10) << "IDA failed at t = " << t << ": " << IDAGetReturnFlagName(status);
        throw std::runtime_error(message.str());
      }
    };
    // IDA's state at `stop`; with step_needs_restart, step by step so that every accepted step can be shown to it.
    // A failure is retried once, from a cold restart at the last accepted state (see solve_dae's documentation).
    auto advance_to = [&](const double stop) {
      bool retried = false;
      auto retry_or_fail = [&](const int status) {
        if (status >= 0) return false;
        if (retried || (status != IDA_CONV_FAIL && status != IDA_ERR_FAIL)) fail_if(status);
        retried = true;
        // On these failures IDA returns its last accepted state at t.
        check(IDAGetCurrentTime(ida_mem, &t), "IDAGetCurrentTime");
        if (warn) {
          std::ostringstream message;
          message << std::setprecision(10) << "IDA failed at t = " << t << " (" << IDAGetReturnFlagName(status)
                  << "); restarting it cold there, once";
          warn(message.str());
        }
        if (before_retry) before_retry(t, solution);
        reset(t, prm.initial_step_size, solution, solution_dot, true);
        return true;
      };
      auto yy = make_nvector_view(solution, ctx);
      auto yp = make_nvector_view(solution_dot, ctx);
      if (step_needs_restart) {
        check(IDASetStopTime(ida_mem, stop), "IDASetStopTime");
        int status = 0;
        do {
          status = IDASolve(ida_mem, stop, &t, yy, yp, IDA_ONE_STEP);
          handle_pending_exception();
          if (retry_or_fail(status)) {
            check(IDASetStopTime(ida_mem, stop), "IDASetStopTime");
            continue;
          }
          if (step_needs_restart(t, solution, solution_dot)) {
            // The flow is discontinuous at t: IDA starts over without its step history.
            reset(t, prm.initial_step_size, solution, solution_dot, true);
            check(IDASetStopTime(ida_mem, stop), "IDASetStopTime");
          }
        } while (status != IDA_TSTOP_RETURN && t < stop);
      } else {
        int status = 0;
        do {
          status = IDASolve(ida_mem, stop, &t, yy, yp, IDA_NORMAL);
          handle_pending_exception();
        } while (retry_or_fail(status));
      }
      check(IDAGetLastStep(ida_mem, &h), "IDAGetLastStep");
    };
    // The first multiple of restart_check_period (counted from the initial time) after t.
    auto next_check_after = [&](const double time) {
      const double period = prm.restart_check_period;
      const double n = std::floor((time - prm.initial_time) / period + 1e-9);
      return prm.initial_time + (n + 1.) * period;
    };

    while (t < prm.final_time) {
      next_time += prm.output_period;
      const double output_time = std::min(next_time, prm.final_time);
      // Up to the output time, through the restart checks in between.
      bool at_output = false;
      while (!at_output) {
        double stop = output_time;
        if (prm.restart_check_period > 0.) {
          const double check_time = next_check_after(t);
          if (check_time < output_time - 1e-9 * prm.restart_check_period) stop = check_time;
        }
        at_output = stop == output_time;
        advance_to(stop);

        warm = warm_restart_available && prm.warm_restart;
        if (warm) capture_history();
        while (solver_should_restart && solver_should_restart(t, solution, solution_dot))
          reset(t, h, solution, solution_dot, true);
        warm = false;
      }

      if (output_step) output_step(t, solution, solution_dot, ++output_number);
    }

    long int n_steps = 0;
    check(IDAGetNumSteps(ida_mem, &n_steps), "IDAGetNumSteps");
    return n_steps;
  }

  template <typename VectorType>
  typename IDADriver<VectorType>::Statistics IDADriver<VectorType>::get_statistics() const
  {
    Statistics s;
    if (ida_mem == nullptr) return s;
    s.valid = IDAGetNumSteps(ida_mem, &s.num_steps) >= 0 &&
              IDAGetNumResEvals(ida_mem, &s.num_residual_evaluations) >= 0 &&
              IDAGetNumLinSolvSetups(ida_mem, &s.num_linear_solver_setups) >= 0 &&
              IDAGetNumErrTestFails(ida_mem, &s.num_error_test_failures) >= 0 &&
              IDAGetNumNonlinSolvIters(ida_mem, &s.num_nonlinear_iterations) >= 0 &&
              IDAGetNumNonlinSolvConvFails(ida_mem, &s.num_nonlinear_convergence_failures) >= 0 &&
              IDAGetNumStepSolveFails(ida_mem, &s.num_step_solve_failures) >= 0 &&
              IDAGetLastStep(ida_mem, &s.last_step_size) >= 0 &&
              IDAGetCurrentStep(ida_mem, &s.current_step_size) >= 0 && IDAGetCurrentTime(ida_mem, &s.current_time) >= 0;
    s.num_jacobian_alpha_updates = alpha_updates;
    return s;
  }

  template <typename VectorType>
  bool IDADriver<VectorType>::get_error_test_vectors(VectorType &estimated_local_errors,
                                                     VectorType &error_weights) const
  {
    if (ida_mem == nullptr) return false;
    auto errors = make_nvector_view(estimated_local_errors, ctx);
    auto weights = make_nvector_view(error_weights, ctx);
    return IDAGetEstLocalErrors(ida_mem, errors) >= 0 && IDAGetErrWeights(ida_mem, weights) >= 0;
  }
} // namespace DiFfRG

template class DiFfRG::IDADriver<dealii::Vector<double>>;
template class DiFfRG::IDADriver<dealii::BlockVector<double>>;
#ifdef DEAL_II_WITH_PETSC
template class DiFfRG::IDADriver<dealii::PETScWrappers::MPI::Vector>;
template class DiFfRG::IDADriver<dealii::PETScWrappers::MPI::BlockVector>;
#endif
