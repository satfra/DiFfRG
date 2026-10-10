// external libraries
#include <deal.II/lac/block_vector.h>

// standard library
#include <limits>
#include <string>
#include <utility>

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
#include <DiFfRG/timestepping/timestep_control/local_error.hh>
#include <DiFfRG/timestepping/timestep_control/output_grid.hh>
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
  void TimeStepperTRBDF2_impl<VectorType, SparseMatrixType, dim, LinearSolver>::set_ignore_nonconv(bool x)
  {
    ptr_newton_TR->set_ignore_nonconv(x);
    ptr_newton_BDF2->set_ignore_nonconv(x);
  }

  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  void TimeStepperTRBDF2_impl<VectorType, SparseMatrixType, dim, LinearSolver>::run_segment(
      AbstractFlowingVariables<NumberType, VectorType> &initial_condition, double start, double stop)
  {
    this->require_no_variables(initial_condition, "TimeStepperTRBDF2");

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

      // The BDF2 stage divided by (2 - gamma), so that its Jacobian equals the trapezoidal one.
      assembler.set_time(t);
      assembler.mass(res, u_n, powr<2>(1. - gamma) / (gamma * (2. - gamma)));

      assembler.set_time(t + dt * gamma);
      assembler.mass(res, u_npgamma, -1. / (gamma * (2. - gamma)));

      assembler.set_time(t + dt);
      assembler.residual(res, u, (1. - gamma) / (2. - gamma) * dt, 1.);
    };
    newton_BDF2.update_jacobian = [&](const VectorType &u) {
      const auto t = tc.get_t();
      const auto dt = tc.get_dt();
      const double mass_weight = 1.;
      const double residual_weight = (1. - gamma) / (2. - gamma) * dt;
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

    // Frames on the grid start + n output_dt, interpolated quadratically through u_n, u_gamma and u_np1 within the
    // step that reaches them (second order, like the method).
    VectorType frame_state;
    auto save_data = [&](double t, const VectorType &u) {
      assembler.set_time(t);
      data_out.write_frame(t, [&](auto &frame) { assembler.attach_data_output(frame, u); });
    };
    internal::OutputGrid output_grid(start, output_dt);
    // R(u) without the mass term at t_n (R_n), t_n + gamma dt (R_gamma) and t_n + dt (R_np1).
    VectorType R_n, R_gamma, R_np1, error_rhs, error;
    bool R_n_valid = false;
    auto evaluate_R = [&](VectorType &R, const VectorType &u, const double t) {
      assembler.set_time(t);
      R.reinit(u);
      assembler.residual(R, u, 1., 0.);
    };
    // Local error estimate of Bank et al. (1985): with f = -M^{-1} R,
    //   M e = 2 k dt (f_n / gamma - f_gamma / (gamma (1 - gamma)) + f_np1 / (1 - gamma)),
    //   k = (-3 gamma^2 + 4 gamma - 2) / (12 (2 - gamma)),
    // filtered by the stage matrix (M + gamma dt J / 2)^{-1}.
    const double k_err = (-3. * gamma * gamma + 4. * gamma - 2.) / (12. * (2. - gamma));

    auto dt_step = [&](double t, double dt) {
      error_estimate = std::numeric_limits<double>::infinity();
      if (!R_n_valid) evaluate_R(R_n, u_n, t);
      R_n_valid = true;

      u_npgamma = u_n;
      newton_TR(u_npgamma);
      u_np1 = u_npgamma;
      newton_BDF2(u_np1);

      evaluate_R(R_gamma, u_npgamma, t + gamma * dt);
      evaluate_R(R_np1, u_np1, t + dt);
      error_rhs.reinit(u_n);
      error_rhs.add(-1. / gamma, R_n, 1. / (gamma * (1. - gamma)), R_gamma);
      error_rhs.add(-1. / (1. - gamma), R_np1);
      error_rhs *= 2. * k_err * dt;
      error.reinit(u_n);
      linSolver.solve(error_rhs, error, impl.abs_tol);
      error_estimate = internal::weighted_rms(error, u_n, u_np1, impl.abs_tol, impl.rel_tol);
      if (error_estimate > 1. && dt > impl.minimal_dt * (1. + 1e-10))
        throw internal::StepRejected("TRBDF2: local error estimate " + std::to_string(error_estimate) + " > 1");

      output_grid.advance_to(t + dt, [&](const double t_out) {
        const double theta = (t_out - t) / dt;
        // Lagrange polynomials on the nodes theta = 0, gamma, 1.
        const double l_n = (theta - gamma) * (theta - 1.) / gamma;
        const double l_gamma = theta * (theta - 1.) / (gamma * (gamma - 1.));
        const double l_np1 = theta * (theta - gamma) / (1. - gamma);
        frame_state.reinit(u_n);
        frame_state.add(l_n, u_n, l_gamma, u_npgamma);
        frame_state.add(l_np1, u_np1);
        save_data(t_out, frame_state);
      });

      u_n = u_np1;
      std::swap(R_n, R_np1);
    };

    // the actual time loop
    save_data(start, u_n);
    const auto no_controller_output = [](double) {};
    while (!tc.finished()) {
      if (adaptor(tc.get_t(), u_n)) {
        u_np1 = u_n;
        u_npgamma = u_n;
        jacobian.reinit(assembler.get_sparsity_pattern_jacobian());
        newton_TR.reinit(u_n);
        newton_BDF2.reinit(u_n);
        R_n_valid = false;
      }
      this->log.progress({.topic = progress_topics::timestep, .time = tc.get_t(), .minimum_verbosity = 1});
      const double t_before_attempt = tc.get_t();
      const double attempted_h = tc.get_dt();
      tc.advance(dt_step, no_controller_output);
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
