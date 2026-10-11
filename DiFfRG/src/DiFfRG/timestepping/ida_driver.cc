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
#include <sunnonlinsol/sunnonlinsol_newton.h>

// standard library
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
  void IDADriver<VectorType>::reset(const double t, const double h, VectorType &solution, VectorType &solution_dot)
  {
    free_ida();
    ida_mem = IDACreate(ctx);

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
                        auto &driver = *static_cast<IDADriver *>(user_data);
                        return call_and_possibly_capture_exception(driver.setup_jacobian, driver.pending_exception, tt,
                                                                   *unwrap_nvector_const<VectorType>(y),
                                                                   *unwrap_nvector_const<VectorType>(y_dot), cj);
                      }),
          "IDASetJacFn");

    // Consistent initial conditions: algebraic components of y and the derivative of the differential ones.
    check(IDASetMaxNumItersIC(ida_mem, 5), "IDASetMaxNumItersIC");
    const int status = IDACalcIC(ida_mem, IDA_YA_YDP_INIT, t + h);
    handle_pending_exception();
    check(status, "IDACalcIC");
    check(IDAGetConsistentIC(ida_mem, yy, yp), "IDAGetConsistentIC");
  }

  template <typename VectorType>
  unsigned int IDADriver<VectorType>::solve_dae(VectorType &solution, VectorType &solution_dot)
  {
    double t = prm.initial_time;
    double h = prm.initial_step_size;
    reset(t, h, solution, solution_dot);
    if (output_step) output_step(t, solution, solution_dot, 0);

    unsigned int output_number = 0;
    double next_time = prm.initial_time;
    while (t < prm.final_time) {
      next_time += prm.output_period;
      auto yy = make_nvector_view(solution, ctx);
      auto yp = make_nvector_view(solution_dot, ctx);
      const int status = IDASolve(ida_mem, next_time, &t, yy, yp, IDA_NORMAL);
      handle_pending_exception();
      if (status < 0) {
        std::ostringstream message;
        message << std::setprecision(10) << "IDA failed at t = " << t << ": " << IDAGetReturnFlagName(status);
        throw std::runtime_error(message.str());
      }
      check(IDAGetLastStep(ida_mem, &h), "IDAGetLastStep");

      while (solver_should_restart && solver_should_restart(t, solution, solution_dot))
        reset(t, h, solution, solution_dot);

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
