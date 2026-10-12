#pragma once

#include <DiFfRG/timestepping/abstract_timestepper.hh>
#include <DiFfRG/timestepping/ida_driver.hh>
#include <DiFfRG/timestepping/jacobian_diagnostics.hh>

namespace DiFfRG
{
  using IDACallbackDiagnostics = SolverCallbackDiagnostics;

  /// Collect IDA's statistics and the callback failure counts for a progress report.
  template <typename VectorType>
  TimesteppingDiagnostics make_timestepping_diagnostics(const IDADriver<VectorType> &time_stepper,
                                                        const IDACallbackDiagnostics &callbacks)
  {
    TimesteppingDiagnostics diagnostics;

    const auto ida = time_stepper.get_statistics();
    if (ida.valid) {
      diagnostics.ida.emplace();
      diagnostics.ida->ida_steps = ida.num_steps;
      diagnostics.ida->ida_error_test_failures = ida.num_error_test_failures;
      diagnostics.ida->ida_nonlinear_convergence_failures = ida.num_nonlinear_convergence_failures;
      diagnostics.ida->ida_step_solve_failures = ida.num_step_solve_failures;
      diagnostics.ida->ida_residual_evaluations = ida.num_residual_evaluations;
      diagnostics.ida->ida_nonlinear_iterations = ida.num_nonlinear_iterations;
      diagnostics.ida->ida_last_step_size = ida.last_step_size;
      diagnostics.ida->ida_current_step_size = ida.current_step_size;
      diagnostics.ida->ida_current_time = ida.current_time;
    }

    diagnostics.callbacks = callbacks;

    return diagnostics;
  }

  /// Log IDA's totals for the run: steps, residual evaluations, linear solver setups (Jacobian requests), Newton
  /// iterations and the two kinds of failures.
  template <typename VectorType, typename Log>
  void log_ida_statistics(const IDADriver<VectorType> &time_stepper, const Log &log)
  {
    const auto ida = time_stepper.get_statistics();
    if (!ida.valid) return;
    log.info("IDA: steps={} residuals={} setups={} (alpha updates {}) newton_iters={} newton_fails={} "
             "error_test_fails={}",
             ida.num_steps, ida.num_residual_evaluations, ida.num_linear_solver_setups, ida.num_jacobian_alpha_updates,
             ida.num_nonlinear_iterations, ida.num_nonlinear_convergence_failures, ida.num_error_test_failures);
  }

  /// The Jacobian record of an IDA Jacobian build with IDA's \f$\alpha\f$; see jacobian_diagnostics.hh.
  template <typename VectorType>
  TimestepperJacobianBuildDiagnostics
  make_ida_jacobian_build_diagnostics(const IDADriver<VectorType> &time_stepper, const std::size_t build_id,
                                      const double alpha,
                                      const ImplicitTimestepperKind stepper_kind = ImplicitTimestepperKind::ida)
  {
    TimestepperJacobianBuildDiagnostics diagnostics;
    diagnostics.jacobian_build_id = build_id;
    diagnostics.stepper_kind = stepper_kind;
    diagnostics.alpha = alpha;
    diagnostics.beta = 1.;
    diagnostics.residual_weight = 1.;
    const auto ida = time_stepper.get_statistics();
    if (ida.valid) {
      diagnostics.step = static_cast<double>(ida.num_steps);
      diagnostics.ida_step = static_cast<double>(ida.num_steps);
      diagnostics.last_h = ida.last_step_size;
      diagnostics.current_h = ida.current_step_size;
    }
    return diagnostics;
  }
} // namespace DiFfRG
