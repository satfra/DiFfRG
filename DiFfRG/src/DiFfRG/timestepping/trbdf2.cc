// external libraries
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/solver/kinsol.hh>
#include <DiFfRG/timestepping/solver/newton.hh>
#include <DiFfRG/timestepping/timestep_control/pi.hh>
#include <DiFfRG/timestepping/trbdf2.hh>

namespace DiFfRG
{
  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  uint TimeStepperTRBDF2_impl<VectorType, SparseMatrixType, dim, LinearSolver>::get_jacobians()
  {
    return ptr_newton_TR->get_jacobians() + ptr_newton_BDF2->get_jacobians();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  double TimeStepperTRBDF2_impl<VectorType, SparseMatrixType, dim, LinearSolver>::get_error()
  {
    return std::sqrt(powr<2>(ptr_newton_TR->get_error()) + powr<2>(ptr_newton_BDF2->get_error()));
  }
  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  void TimeStepperTRBDF2_impl<VectorType, SparseMatrixType, dim, LinearSolver>::set_ignore_nonconv(bool x)
  {
    ptr_newton_TR->set_ignore_nonconv(x);
    ptr_newton_BDF2->set_ignore_nonconv(x);
  }

  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  void TimeStepperTRBDF2_impl<VectorType, SparseMatrixType, dim, LinearSolver>::run_segment(
      AbstractFlowingVariables<NumberType, VectorType> &initial_condition, double start, double stop)
  {

    const double gamma = 2. - std::sqrt(2.);

    // make some local copies of the initial condition which we use for stepping
    VectorType u_n = initial_condition.spatial_data();
    VectorType u_npgamma = initial_condition.spatial_data();
    VectorType u_np1 = initial_condition.spatial_data();

    // newton algorithm
    ptr_newton_TR = std::make_shared<Newton<VectorType>>(impl.abs_tol, impl.rel_tol, 2e-1, 11, 21);
    ptr_newton_BDF2 = std::make_shared<Newton<VectorType>>(impl.abs_tol, impl.rel_tol, 2e-1, 11, 21);
    auto &newton_TR = *ptr_newton_TR;
    auto &newton_BDF2 = *ptr_newton_BDF2;
    newton_TR.set_report_port(this->log);
    newton_BDF2.set_report_port(this->log);

    // create time controller instance
    TC_PI tc(*this, 2, start, stop, impl.dt, impl.minimal_dt, impl.maximal_dt, output_dt);
    assembler.set_time(start);

    // shared jacobian matrix and linear solver (TR and BDF2 stages are sequential, never simultaneous)
    SparseMatrixType jacobian(assembler.get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver;
    linSolver.set_report_port(this->log);
    if constexpr (requires { linSolver.set_iterative_refinement(true); })
      linSolver.set_iterative_refinement(this->impl.iterative_refinement);
    const bool jacobian_diagnostics_enabled = impl.jacobian_diagnostics;
    const DiagnosticPort jacobian_diagnostic_port =
        jacobian_diagnostics_enabled ? data_out.diagnostic_port() : DiagnosticPort{};

    // all functions for assembly of the problem and linear solving
    newton_TR.residual = [&](VectorType &res, const VectorType &u) {
      const auto t = tc.get_t();
      const auto dt = tc.get_dt();
      res = 0;

      assembler.set_time(t);
      assembler.residual(res, u_n, gamma * dt / 2., -1.);

      assembler.set_time(t + dt * gamma);
      assembler.residual(res, u, gamma * dt / 2., 1.);
    };
    newton_TR.update_jacobian = [&](const VectorType &u) {
      const auto t = tc.get_t();
      const auto dt = tc.get_dt();
      const double residual_weight = gamma * dt / 2.;
      const auto build_diagnostics = this->jacobian_diagnostics_state.make_build(
          this->next_jacobian_build_id++, ImplicitTimestepperKind::trbdf2, ImplicitTimestepperStage::trapezoidal, dt,
          1., 1., residual_weight);
      JacobianMatrixDiagnostics matrix_diagnostics;
      matrix_diagnostics.n_rows = jacobian.m();
      JacobianFactorizationDiagnostics factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "jacobian_diagnostics.csv", t + dt * gamma,
                                    build_diagnostics, matrix_diagnostics, factorization_diagnostics);
      };

      try {
        jacobian = 0;

        assembler.set_time(t + dt * gamma);
        assembler.jacobian(jacobian, u, residual_weight, 1.);
        if (jacobian_diagnostics_enabled) matrix_diagnostics = analyze_jacobian_matrix(jacobian);

        this->log.progress(
            {.topic = progress_topics::trapezoidal_jacobian, .time = t + dt * gamma, .minimum_verbosity = 2});

        linSolver.init(jacobian);
        factorize_with_diagnostics(linSolver, jacobian, factorization_diagnostics, jacobian_diagnostics_enabled);
        record_diagnostics();
      } catch (...) {
        if constexpr (decltype(linSolver)::performs_factorization)
          if (std::isnan(factorization_diagnostics.factorization_success))
            factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        throw;
      }
    };
    newton_TR.lin_solve = [&](VectorType &Du, const VectorType &res) {
      linSolver.solve(res, Du, std::min(impl.abs_tol, impl.rel_tol * res.l2_norm()));
    };
    newton_TR.reinit(u_n);

    newton_BDF2.residual = [&](VectorType &res, const VectorType &u) {
      const auto t = tc.get_t();
      const auto dt = tc.get_dt();
      res = 0;

      assembler.set_time(t);
      assembler.mass(res, u_n, powr<2>(1. - gamma) / gamma);

      assembler.set_time(t + dt * gamma);
      assembler.mass(res, u_npgamma, -1. / gamma);

      assembler.set_time(t + dt);
      assembler.residual(res, u, (1. - gamma) * dt, (2. - gamma));
    };
    newton_BDF2.update_jacobian = [&](const VectorType &u) {
      const auto t = tc.get_t();
      const auto dt = tc.get_dt();
      const double mass_weight = 2. - gamma;
      const double residual_weight = (1. - gamma) * dt;
      const auto build_diagnostics = this->jacobian_diagnostics_state.make_build(
          this->next_jacobian_build_id++, ImplicitTimestepperKind::trbdf2, ImplicitTimestepperStage::bdf2, dt,
          mass_weight, mass_weight, residual_weight);
      JacobianMatrixDiagnostics matrix_diagnostics;
      matrix_diagnostics.n_rows = jacobian.m();
      JacobianFactorizationDiagnostics factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "jacobian_diagnostics.csv", t + dt, build_diagnostics,
                                    matrix_diagnostics, factorization_diagnostics);
      };

      try {
        jacobian = 0;

        assembler.set_time(t + dt);
        assembler.jacobian(jacobian, u, residual_weight, mass_weight);
        if (jacobian_diagnostics_enabled) matrix_diagnostics = analyze_jacobian_matrix(jacobian);

        this->log.progress({.topic = progress_topics::bdf2_jacobian, .time = t + dt, .minimum_verbosity = 2});

        linSolver.init(jacobian);
        factorize_with_diagnostics(linSolver, jacobian, factorization_diagnostics, jacobian_diagnostics_enabled);
        record_diagnostics();
      } catch (...) {
        if constexpr (decltype(linSolver)::performs_factorization)
          if (std::isnan(factorization_diagnostics.factorization_success))
            factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        throw;
      }
    };
    newton_BDF2.lin_solve = [&](VectorType &Du, const VectorType &res) {
      linSolver.solve(res, Du, std::min(impl.abs_tol, impl.rel_tol * res.l2_norm()));
    };
    newton_BDF2.reinit(u_n);

    // saving and stepping helper functions
    auto save_data = [&](double t) {
      assembler.set_time(t);
      data_out.write_frame(t, [&](auto &frame) { assembler.attach_data_output(frame, u_n); });
    };
    auto dt_step = [&](double /*t*/, double /*dt*/) {
      u_npgamma = u_n;
      newton_TR(u_npgamma);
      u_np1 = u_npgamma;
      newton_BDF2(u_np1);
      u_n = u_np1;
    };

    // the actual time loop
    save_data(start);
    while (!tc.finished()) {
      if (adaptor(tc.get_t(), u_n)) {
        u_np1 = u_n;
        u_npgamma = u_n;
        jacobian.reinit(assembler.get_sparsity_pattern_jacobian());
        newton_TR.reinit(u_n);
        newton_BDF2.reinit(u_n);
      }
      this->log.progress({.topic = progress_topics::timestep, .time = tc.get_t(), .minimum_verbosity = 1});
      const double t_before_attempt = tc.get_t();
      const double attempted_h = tc.get_dt();
      tc.advance(dt_step, save_data);
      this->jacobian_diagnostics_state.finish_attempt(tc.get_t() > t_before_attempt, attempted_h);
    }

    initial_condition.spatial_data() = u_n;
    this->drain_output();
  }

} // namespace DiFfRG

template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1, DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2, DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3, DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                              DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                              DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                              DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3, DiFfRG::GMRES>;

template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                              DiFfRG::GMRES>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                              DiFfRG::GMRES>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                              DiFfRG::GMRES>;

// The default-solver spelling. DefaultLinearSolver is deliberately an indirect alias and so
// stays a distinct template argument from UMFPack on every compiler, which means
// TimeStepper<Assembler> needs its own instantiations alongside TimeStepper<Assembler, UMFPack>.
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                              DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                              DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                              DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                              DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                              DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperTRBDF2_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                              DiFfRG::DefaultLinearSolver>;
