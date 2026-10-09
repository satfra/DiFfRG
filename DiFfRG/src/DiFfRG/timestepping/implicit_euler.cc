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
#include <DiFfRG/timestepping/timestep_control/pi.hh>
#include <vector>

namespace DiFfRG
{
  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  void TimeStepperImplicitEuler_impl<VectorType, SparseMatrixType, dim, LinearSolver>::run_segment(
      AbstractFlowingVariables<NumberType, VectorType> &initial_condition, double start, double stop)
  {

    // make some local copies of the initial condition which we use for stepping
    VectorType old_solution = initial_condition.spatial_data();
    VectorType solution = initial_condition.spatial_data();

    // newton algorithm
    Newton<VectorType> newton(impl.abs_tol, impl.rel_tol, 2e-1, 11, 21);
    newton.set_report_port(this->log);

    // create time controller instance
    TC_PI tc(newton, 1, start, stop, impl.dt, impl.minimal_dt, impl.maximal_dt, output_dt);
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

    // saving and stepping helper functions
    auto save_data = [&](double t) {
      assembler.set_time(t);

      data_out.write_frame(t, [&](auto &frame) {
        if (t > start)
          assembler.attach_data_output(frame, solution, newton.get_residual());
        else
          assembler.attach_data_output(frame, solution);
      });
    };
    auto dt_step = [&](double t, double dt) {
      assembler.set_time(t + dt);
      solution = old_solution;
      newton(solution);
      old_solution = solution;
    };

    // the actual time loop
    save_data(start);
    while (!tc.finished()) {
      if (adaptor(tc.get_t(), old_solution)) solution = old_solution;
      const double t_before_attempt = tc.get_t();
      const double attempted_h = tc.get_dt();
      tc.advance(dt_step, save_data);
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
