#pragma once

#include <deal.II/sundials/ida.h>

#include <DiFfRG/timestepping/abstract_timestepper.hh>
#include <DiFfRG/timestepping/jacobian_diagnostics.hh>

namespace DiFfRG
{
  using IDACallbackDiagnostics = SolverCallbackDiagnostics;

  template <typename VectorType>
  TimesteppingDiagnostics make_timestepping_diagnostics(const SUNDIALS::IDA<VectorType> &time_stepper,
                                                        const IDACallbackDiagnostics &callbacks)
  {
    TimesteppingDiagnostics diagnostics;

    if constexpr (requires { time_stepper.get_statistics(); }) {
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
    }

    diagnostics.callbacks = callbacks;

    return diagnostics;
  }

  template <typename VectorType>
  TimestepperJacobianBuildDiagnostics
  make_ida_jacobian_build_diagnostics(const SUNDIALS::IDA<VectorType> &time_stepper, const std::size_t build_id,
                                      const double alpha,
                                      const ImplicitTimestepperKind stepper_kind = ImplicitTimestepperKind::ida)
  {
    TimestepperJacobianBuildDiagnostics diagnostics;
    diagnostics.jacobian_build_id = build_id;
    diagnostics.stepper_kind = stepper_kind;
    diagnostics.alpha = alpha;
    diagnostics.beta = 1.;
    diagnostics.residual_weight = 1.;
    if constexpr (requires { time_stepper.get_statistics(); }) {
      const auto ida = time_stepper.get_statistics();
      if (ida.valid) {
        diagnostics.step = static_cast<double>(ida.num_steps);
        diagnostics.ida_step = static_cast<double>(ida.num_steps);
        diagnostics.last_h = ida.last_step_size;
        diagnostics.current_h = ida.current_step_size;
      }
    }
    return diagnostics;
  }
} // namespace DiFfRG
