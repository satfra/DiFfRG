// external libraries
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/implicit_euler.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/solver/kinsol.hh>
#include <DiFfRG/timestepping/solver/newton.hh>
#include <DiFfRG/timestepping/timestep_control/local_error.hh>
#include <DiFfRG/timestepping/timestep_control/output_grid.hh>
#include <DiFfRG/timestepping/timestep_control/pi.hh>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace DiFfRG
{
  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  void TimeStepperImplicitEuler_impl<VectorType, SparseMatrixType, dim, LinearSolver>::run_segment(
      AbstractFlowingVariables<NumberType, VectorType> &initial_condition, double start, double stop)
  {
    this->require_no_variables(initial_condition, "TimeStepperImplicitEuler");

    // make some local copies of the initial condition which we use for stepping
    VectorType old_solution = initial_condition.spatial_data();
    VectorType solution = initial_condition.spatial_data();

    // newton algorithm
    ptr_newton = std::make_shared<Newton<VectorType>>(impl.abs_tol, impl.rel_tol, 2e-1, 11, 21);
    auto &newton = *ptr_newton;
    newton.set_report_port(this->log);

    // create time controller instance
    TC_PI tc(*this, 1, start, stop, impl.dt, impl.minimal_dt, impl.maximal_dt, output_dt);
    assembler.set_time(start);

    // create jacobian and solver for inverse jacobian
    SparseMatrixType jacobian(assembler.get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver;
    linSolver.set_report_port(this->log);
    if constexpr (requires { linSolver.set_iterative_refinement(true); })
      linSolver.set_iterative_refinement(this->impl.iterative_refinement);
    const bool jacobian_diagnostics_enabled = impl.jacobian_diagnostics;
    const DiagnosticPort jacobian_diagnostic_port =
        jacobian_diagnostics_enabled ? data_out.diagnostic_port() : DiagnosticPort{};

    // all functions for assembly of the problem and linear solving
    newton.residual = [&](VectorType &res, const VectorType &u) {
      CalcDtTimer calc_timer;
      res = 0;
      assembler.residual(res, u, tc.get_dt(), 1);
      assembler.mass(res, old_solution, -1.);

      this->log.progress({.topic = progress_topics::implicit_residual,
                          .time = tc.get_t(),
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1});
    };

    newton.update_jacobian = [&](const VectorType &u) {
      CalcDtTimer calc_timer;
      const auto build_diagnostics = this->jacobian_diagnostics_state.make_build(
          this->next_jacobian_build_id++, ImplicitTimestepperKind::implicit_euler, ImplicitTimestepperStage::main,
          tc.get_dt(), 1., 1., tc.get_dt());
      JacobianMatrixDiagnostics matrix_diagnostics;
      matrix_diagnostics.n_rows = jacobian.m();
      JacobianFactorizationDiagnostics factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "jacobian_diagnostics.csv", tc.get_t(), build_diagnostics,
                                    matrix_diagnostics, factorization_diagnostics);
      };

      try {
        jacobian = 0;
        assembler.jacobian(jacobian, u, tc.get_dt(), 1);
        if (jacobian_diagnostics_enabled) matrix_diagnostics = analyze_jacobian_matrix(jacobian);
        linSolver.init(jacobian);

        this->log.progress({.topic = progress_topics::jacobian,
                            .time = tc.get_t(),
                            .duration_ms = calc_timer.lap(),
                            .minimum_verbosity = 2});

        factorize_with_diagnostics(linSolver, jacobian, factorization_diagnostics, jacobian_diagnostics_enabled);
        if (factorization_diagnostics.factorization_success == 1.)
          this->log.progress({.topic = progress_topics::factorization,
                              .time = tc.get_t(),
                              .duration_ms = calc_timer.lap(),
                              .minimum_verbosity = 3});
        record_diagnostics();
      } catch (...) {
        if constexpr (decltype(linSolver)::performs_factorization)
          if (std::isnan(factorization_diagnostics.factorization_success))
            factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        throw;
      }
    };

    newton.lin_solve = [&](VectorType &Du, const VectorType &res) {
      CalcDtTimer calc_timer;
      const auto sol_iterations = linSolver.solve(res, Du, std::min(impl.abs_tol, impl.rel_tol * res.l2_norm()));
      // A direct solver reports no iterations (-1), but its time counts all the same.
      this->log.progress({.topic = progress_topics::implicit_linear_solve,
                          .time = tc.get_t(),
                          .duration_ms = calc_timer.lap(),
                          .iterations = sol_iterations,
                          .minimum_verbosity = 2});
    };

    newton.reinit(solution);

    // Frames on the grid start + n output_dt, linearly interpolated within the step that reaches them.
    VectorType frame_state;
    auto save_data = [&](double t, const VectorType &u) {
      assembler.set_time(t);
      data_out.write_frame(t, [&](auto &frame) { assembler.attach_data_output(frame, u); });
    };
    internal::OutputGrid output_grid(start, output_dt);
    // R(u) without the mass term, at the start (R_old) and end (R_new) of a step.
    VectorType R_old, R_new, error_rhs, error;
    bool R_old_valid = false;
    auto evaluate_R = [&](VectorType &R, const VectorType &u, const double t) {
      assembler.set_time(t);
      R.reinit(u);
      assembler.residual(R, u, 1., 0.);
    };

    auto dt_step = [&](double t, double dt) {
      error_estimate = std::numeric_limits<double>::infinity();
      if (!R_old_valid) evaluate_R(R_old, old_solution, t);
      R_old_valid = true;

      assembler.set_time(t + dt);
      solution = old_solution;
      newton(solution);

      // Local error of backward Euler, (dt/2)(f_{n+1} - f_n) with M f = -R(u), filtered by (M + dt J)^{-1}.
      evaluate_R(R_new, solution, t + dt);
      error_rhs = R_old;
      error_rhs -= R_new;
      error_rhs *= dt / 2.;
      error.reinit(solution);
      linSolver.solve(error_rhs, error, impl.abs_tol);
      error_estimate = internal::weighted_rms(error, old_solution, solution, impl.abs_tol, impl.rel_tol);
      if (error_estimate > 1. && dt > impl.minimal_dt * (1. + 1e-10))
        throw internal::StepRejected("ImplicitEuler: local error estimate " + std::to_string(error_estimate) + " > 1");

      output_grid.advance_to(t + dt, [&](const double t_out) {
        const double theta = (t_out - t) / dt;
        frame_state = old_solution;
        frame_state.sadd(1. - theta, theta, solution);
        save_data(t_out, frame_state);
      });

      old_solution = solution;
      std::swap(R_old, R_new);
    };

    // the actual time loop
    save_data(start, old_solution);
    const auto no_controller_output = [](double) {};
    while (!tc.finished()) {
      if (adaptor(tc.get_t(), old_solution)) {
        solution = old_solution;
        jacobian.reinit(assembler.get_sparsity_pattern_jacobian());
        newton.reinit(old_solution);
        R_old_valid = false;
      }
      const double t_before_attempt = tc.get_t();
      const double attempted_h = tc.get_dt();
      tc.advance(dt_step, no_controller_output);
      this->jacobian_diagnostics_state.finish_attempt(tc.get_t() > t_before_attempt, attempted_h);
    }

    initial_condition.spatial_data() = solution;
    this->drain_output();
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                     DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                     DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                     DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                     DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                     DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                     DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                     DiFfRG::GMRES>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                     DiFfRG::GMRES>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                     DiFfRG::GMRES>;

template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                     DiFfRG::GMRES>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                     DiFfRG::GMRES>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                     DiFfRG::GMRES>;

// The default-solver spelling. DefaultLinearSolver is deliberately an indirect alias and so
// stays a distinct template argument from UMFPack on every compiler, which means
// TimeStepper<Assembler> needs its own instantiations alongside TimeStepper<Assembler, UMFPack>.
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                     DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                     DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                     DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                     DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                     DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperImplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                     DiFfRG::DefaultLinearSolver>;
