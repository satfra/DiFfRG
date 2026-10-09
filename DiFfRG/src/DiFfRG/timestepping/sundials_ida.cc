// external libraries
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/sundials/ida.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/ida_rank_agreement.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/PETScDirect.hh>
#include <DiFfRG/timestepping/linear_solver/PETScKrylov.hh>
#include <DiFfRG/timestepping/linear_solver/ScaledGMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/local_tolerances.hh>
#include <DiFfRG/timestepping/sundials_diagnostics.hh>
#include <DiFfRG/timestepping/sundials_ida.hh>

namespace DiFfRG
{

  using namespace dealii;

  namespace
  {
    using internal::agreed_ida_result;
    using internal::recoverable_ida_callback_failure;

    template <typename VectorType> double l1_norm_or_nan(const VectorType *vector)
    {
      if (vector == nullptr) return std::numeric_limits<double>::quiet_NaN();
      try {
        return vector->l1_norm();
      } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    template <typename VectorType> bool is_finite_vector(const VectorType &vector)
    {
      return std::isfinite(l1_norm_or_nan(&vector));
    }

    struct IDACallbackTrace {
      bool enabled = false;
      double min_t = 0.;
      uint max_lines = 200;
      bool trace_successes = false;
      double Lambda = -1.;
      ReportPort report;

      size_t call_index = 0;
      size_t printed_lines = 0;
      double last_t = std::numeric_limits<double>::quiet_NaN();
      size_t same_t_index = 0;

      IDACallbackTrace(const bool enabled, const double min_t, const uint max_lines, const bool trace_successes,
                       const double Lambda, ReportPort report)
          : enabled(enabled), min_t(min_t), max_lines(max_lines), trace_successes(trace_successes), Lambda(Lambda),
            report(std::move(report))
      {
      }

      template <typename IDAType, typename VectorType, typename ResidualType = VectorType>
      void record(const char *phase, const double t, const uint failure_counter, const IDAType &time_stepper,
                  const IDACallbackDiagnostics &callback_diagnostics, const char *outcome, const VectorType *y,
                  const VectorType *y_dot, const ResidualType *residual, const std::string &detail = {})
      {
        ++call_index;
        if (std::isfinite(last_t) && std::abs(t - last_t) <= 1e-14)
          ++same_t_index;
        else {
          same_t_index = 0;
          last_t = t;
        }

        if (!enabled || report.verbosity() < 5 || t < min_t) return;
        const bool failure = std::string(outcome) != "success";
        if (!failure && printed_lines >= max_lines) return;
        if (!trace_successes && !failure) return;

        const auto diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        std::ostringstream stream;
        stream << std::scientific << std::setprecision(16);
        stream << "[IDA TRACE] phase=" << phase << " outcome=" << outcome << " call=" << call_index
               << " same_t=" << same_t_index
               << " hint=" << (same_t_index == 0 ? "first-trial-candidate" : "repeat-corrector-candidate")
               << " t=" << t;
        if (Lambda > 0.) stream << " k=" << std::exp(-t) * Lambda;
        stream << " failure_counter=" << failure_counter;
        if (diagnostics.ida) {
          const auto &ida = *diagnostics.ida;
          stream << " ida_steps=" << ida.ida_steps << " residual_evals=" << ida.ida_residual_evaluations
                 << " precision_rejects=" << ida.ida_error_test_failures
                 << " nonlinear_iterations=" << ida.ida_nonlinear_iterations
                 << " nonlinear_failures=" << ida.ida_nonlinear_convergence_failures
                 << " step_solve_failures=" << ida.ida_step_solve_failures << " last_h=" << ida.ida_last_step_size
                 << " current_h=" << ida.ida_current_step_size;
        }
        stream << " y_l1=" << l1_norm_or_nan(y) << " ydot_l1=" << l1_norm_or_nan(y_dot)
               << " residual_l1=" << l1_norm_or_nan(residual);
        if (!detail.empty()) stream << " detail=\"" << detail << "\"";
        report.info(stream.str());
        ++printed_lines;
      }

      template <typename IDAType, typename VectorType>
      void record(const char *phase, const double t, const uint failure_counter, const IDAType &time_stepper,
                  const IDACallbackDiagnostics &callback_diagnostics, const char *outcome, const VectorType *y,
                  const VectorType *y_dot, std::nullptr_t, const std::string &detail = {})
      {
        record(phase, t, failure_counter, time_stepper, callback_diagnostics, outcome, y, y_dot,
               static_cast<const VectorType *>(nullptr), detail);
      }
    };

    template <typename VectorType> class IDAErrorDofMonitor
    {
    public:
      IDAErrorDofMonitor(const bool enabled, const uint top_n, const double Lambda, ReportPort report)
          : enabled(enabled), top_n(top_n), Lambda(Lambda), report(std::move(report))
      {
      }

      template <typename IDAType, typename Callback>
      void observe(const double t, const IDAType &time_stepper, const TimesteppingDiagnostics &diagnostics,
                   const VectorType &solution, const Callback &callback)
      {
        if (!diagnostics.ida) return;
        const auto &ida = *diagnostics.ida;

        const long int previous_error_test_failures = initialized ? last_error_test_failures : 0;
        initialized = true;
        const long int reject_delta = ida.ida_error_test_failures - previous_error_test_failures;
        last_error_test_failures = ida.ida_error_test_failures;
        if (!enabled || reject_delta <= 0) return;

        if constexpr (requires(const IDAType &ida, VectorType &errors, VectorType &weights) {
                        ida.get_error_test_vectors(errors, weights);
                      }) {
          estimated_local_errors.reinit(solution);
          error_weights.reinit(solution);

          if (!time_stepper.get_error_test_vectors(estimated_local_errors, error_weights)) {
            report.warn("[IDA error DOF] rejects +{} at t={}: failed to query local-error vectors", reject_delta, t);
            return;
          }

          IDAErrorDofDiagnostics report;
          report.t = t;
          report.k = Lambda > 0. ? std::exp(-t) * Lambda : std::numeric_limits<double>::quiet_NaN();
          report.reject_delta = reject_delta;
          report.total_rejects = ida.ida_error_test_failures;
          report.ida_steps = ida.ida_steps;
          report.ida_last_step_size = ida.ida_last_step_size;
          report.ida_current_step_size = ida.ida_current_step_size;
          report.ida_current_time = ida.ida_current_time;

          const std::size_t n_dofs = solution.size();
          report.top_dofs.reserve(std::min<std::size_t>(top_n, n_dofs));
          double weighted_error_norm_squared = 0.;
          std::vector<IDAErrorDofRecord> records;
          records.reserve(n_dofs);
          for (std::size_t i = 0; i < n_dofs; ++i) {
            const double estimated_local_error = static_cast<double>(estimated_local_errors[i]);
            const double error_weight = static_cast<double>(error_weights[i]);
            const double contribution = std::abs(estimated_local_error * error_weight);
            weighted_error_norm_squared += contribution * contribution;
            records.push_back({static_cast<types::global_dof_index>(i), static_cast<double>(solution[i]),
                               estimated_local_error, error_weight, contribution});
          }

          if (n_dofs > 0) report.wrms = std::sqrt(weighted_error_norm_squared / static_cast<double>(n_dofs));
          const std::size_t report_count = std::min<std::size_t>(top_n, records.size());
          std::partial_sort(records.begin(), records.begin() + report_count, records.end(),
                            [](const auto &a, const auto &b) { return a.contribution > b.contribution; });
          report.top_dofs.insert(report.top_dofs.end(), records.begin(), records.begin() + report_count);

          if (callback) {
            callback(report);
          } else {
            log_generic_report(report);
          }
        } else {
          report.warn("[IDA error DOF] rejects +{} at t={}: local-error vectors unavailable", reject_delta, t);
        }
      }

    private:
      bool enabled = false;
      uint top_n = 0;
      double Lambda = -1.;
      ReportPort report;
      bool initialized = false;
      long int last_error_test_failures = 0;
      VectorType estimated_local_errors;
      VectorType error_weights;

      void log_generic_report(const IDAErrorDofDiagnostics &diagnostics) const
      {
        report.warn("[IDA error DOF] rejects +{} t={} k={} total={} steps={} last_h={} h={} wrms={}",
                    diagnostics.reject_delta, diagnostics.t, diagnostics.k, diagnostics.total_rejects,
                    diagnostics.ida_steps, diagnostics.ida_last_step_size, diagnostics.ida_current_step_size,
                    diagnostics.wrms);
        for (std::size_t rank = 0; rank < diagnostics.top_dofs.size(); ++rank) {
          const auto &record = diagnostics.top_dofs[rank];
          report.warn("[IDA error DOF] rank={} dof={} y={} ele={} ewt={} contribution={}", rank, record.dof,
                      record.value, record.estimated_local_error, record.error_weight, record.contribution);
        }
      }
    };
  } // namespace

  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA_impl<VectorType, SparseMatrixType, dim, LinearSolver>::run_segment(
      AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start, const double t_stop)
  {

    auto &full_data = initial_condition.data();
    if constexpr (dim == 0)
      run_vars(full_data.block(1), t_start, t_stop);
    else {
      if (full_data.n_blocks() != 1)
        run(full_data, t_start, t_stop);
      else
        run(full_data.block(0), t_start, t_stop);
    }
  }

  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA_impl<VectorType, SparseMatrixType, dim, LinearSolver>::run(VectorType &initial_data,
                                                                                          const double t_start,
                                                                                          const double t_stop)
  {
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    SparseMatrixType jacobian;
    assembler.reinit_matrix(jacobian);
    LinearSolver<SparseMatrixType, VectorType> linSolver;
    linSolver.set_report_port(this->log);
    if constexpr (requires { linSolver.set_iterative_refinement(true); })
      linSolver.set_iterative_refinement(this->impl.iterative_refinement);
    const bool jacobian_diagnostics_enabled = impl.jacobian_diagnostics;
    const DiagnosticPort jacobian_diagnostic_port =
        jacobian_diagnostics_enabled ? data_out.diagnostic_port() : DiagnosticPort{};

    // Create a SUNDIALS IDA object with the right settings
    typename SUNDIALS::IDA<VectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5,
                                                                impl.max_non_linear_iterations, 0, impl.abs_tol,
                                                                impl.rel_tol);
    typename SUNDIALS::IDA<VectorType> time_stepper(ida_data);

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = t_start;
    uint failure_counter = 0;
    IDACallbackDiagnostics callback_diagnostics;
    IDACallbackTrace callback_trace(impl.ida_callback_trace, impl.ida_callback_trace_min_t,
                                    impl.ida_callback_trace_max_lines, impl.ida_callback_trace_successes,
                                    this->log.Lambda(), this->log);
    IDAErrorDofMonitor<VectorType> error_dof_monitor(
        impl.ida_error_dof_diagnostics, impl.ida_error_dof_diagnostics_top_n, this->log.Lambda(), this->log);
    bool output_after_failure = false;

    // Initialize initial condition
    VectorType y = initial_data;
    VectorType y_dot = initial_data;
    y_dot *= 0.;

    // Pointer to current residual for monitoring. Must be initialised before
    // SUNDIALS::IDA calls `output_step` at the initial `t = 0` (the IDA wrapper
    // may emit an initial-condition output before the first residual lambda
    // has run and set this pointer). Without this placeholder the first
    // `output_step` dereferences uninitialised garbage and the process
    // segfaults inside `time_stepper.solve_dae`.
    VectorType residual_placeholder = initial_data;
    residual_placeholder *= 0.;
    VectorType *residual = &residual_placeholder;

    // Per-dof absolute tolerances, if the model sets them (def::HasAbsTolerances); see LocalAbsTolerances.
    LocalAbsTolerances<VectorType, SparseMatrixType, dim> local_tol(assembler, impl.abs_tol, impl.rel_tol,
                                                                    impl.local_tolerance_refresh);
    if (local_tol.init(y)) {
      time_stepper.get_local_tolerances = [&]() -> VectorType & { return local_tol.get(); };
      this->log.info("IDA: per-dof absolute tolerances from the model, {:.3e} .. {:.3e}", local_tol.min(),
                     local_tol.max());
    }

    // Tells SUNDIALS to do an internal reset, e.g. if we do local refinement, or if the per-dof tolerances drifted
    time_stepper.solver_should_restart = [&](const double t, VectorType &sol, VectorType &sol_dot) -> bool {
      if (adaptor(t, sol)) {
        assembler.reinit_vector(sol_dot);
        assembler.reinit_matrix(jacobian);
        local_tol.init(sol);
        return true;
      }
      if (local_tol.refresh_needed(sol)) {
        this->log.info("IDA: per-dof absolute tolerances refreshed at t = {:.5f} ({:.3e} .. {:.3e}), restarting", t,
                       local_tol.min(), local_tol.max());
        return true;
      }
      return false;
    };

    time_stepper.differential_components = [&]() { return assembler.get_differential_indices(); };

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](VectorType &v) { assembler.reinit_vector(v); };

    // Fully-replicated read-only views for the output path; see SolutionView.
    SolutionView<VectorType> sol_view, sol_dot_view, residual_view;
    assembler.reinit_solution_view(sol_view);
    assembler.reinit_solution_view(sol_dot_view);
    assembler.reinit_solution_view(residual_view);

    // Replicated views of the state handed to the assemblers.
    //
    // IDA's vectors hold only this rank's rows. An assembler visiting one of its own cells reads
    // EVERY dof of that cell, and on a partition boundary some of those belong to a neighbour.
    // deal.II's non-ghosted read path is `ptr[index - local_begin]` with no bounds check, so such a
    // read does not fail -- it silently returns whatever is next in memory. The matrix survives
    // that (for a linear flux the Jacobian does not depend on the state at all, which is exactly
    // why the Jacobian diagnostics matched across rank counts while the residual did not), but the
    // residual is quietly wrong and IDA's Newton then fails to converge.
    SolutionView<VectorType> y_state, y_dot_state;
    assembler.reinit_solution_view(y_state);
    assembler.reinit_solution_view(y_dot_state);

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    // SUNDIALS::IDA::solve_dae reports the initial condition as t = 0, whatever the start time is.
    // The first call is always that one, so it is relabelled to the actual start of the run.
    bool initial_output = true;
    time_stepper.output_step = [&](const double t_reported, const VectorType &sol, const VectorType &sol_dot,
                                   unsigned int /*step_number*/) {
      const double t = std::exchange(initial_output, false) ? t_start : t_reported;
      if (!is_close(last_save, t, 1e-10)) {
        assembler.set_time(t);
        // Refreshed here, OUTSIDE write_frame: refreshing a ghosted replica communicates, while
        // write_frame runs its contributor on rank 0 only. Doing it inside would have rank 0 enter a
        // collective the other ranks never reach.
        sol_view.refresh(sol);
        sol_dot_view.refresh(sol_dot);
        residual_view.refresh(*residual);
        data_out.write_frame(t, [&](auto &frame) {
          assembler.attach_data_output(frame, sol_view.get(), VectorType(), sol_dot_view.get(), residual_view.get());
        });

        last_save = t;
      }
      if (!output_after_failure && !is_close(t, t_start)) {
        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        error_dof_monitor.observe(t, time_stepper, current_diagnostics, sol, ida_error_dof_callback);
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &res) -> int {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (failure_counter == 0 && !is_close(t, t_start) && stuck > 100) {
        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent event{.topic = progress_topics::implicit_residual, .time = t, .minimum_verbosity = 1};
        current_diagnostics.append_to(event);
        this->log.progress(event);
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter == 0 && is_close(t, t_start) && stuck > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "failure-limit", &y,
                              &y_dot, &res);
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
        callback_diagnostics.nonfinite_solution_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-y", &y,
                              &y_dot, &res);
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      assembler.set_time(t);

      res = 0;
      y_state.refresh(y);
      y_dot_state.refresh(y_dot);
      assembler.residual(res, y_state.get(), 1., y_dot_state.get(), 1.);
      residual = &res;

      if (!std::isfinite(res.l1_norm())) {
        callback_diagnostics.nonfinite_residual_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-residual",
                              &y, &y_dot, &res);
        return agreed_ida_result(assembler.get_communicator(), true);
      }
      const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
      ProgressEvent event{.topic = progress_topics::implicit_residual,
                          .time = t,
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1};
      current_diagnostics.append_to(event);
      this->log.progress(event);
      callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            &res);

      failure_counter = 0;
      return agreed_ida_result(assembler.get_communicator(), false);
    };
    // Calculate the jacobian d(y_dot + F(y))/dy + d(y_dot*alpha)/dy_dot
    time_stepper.setup_jacobian = [&](const double t, const VectorType &y, const VectorType &y_dot,
                                      const double alpha) -> int {
      CalcDtTimer calc_timer;
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      const auto build_diagnostics =
          make_ida_jacobian_build_diagnostics(time_stepper, this->next_jacobian_build_id++, alpha);
      JacobianMatrixDiagnostics matrix_diagnostics;
      JacobianFactorizationDiagnostics factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "jacobian_diagnostics.csv", t, build_diagnostics,
                                    matrix_diagnostics, factorization_diagnostics);
      };

      assembler.set_time(t);

      try {
        if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-state",
                                &y, &y_dot, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }

        jacobian = 0;
        y_state.refresh(y);
        y_dot_state.refresh(y_dot);
        assembler.jacobian(jacobian, y_state.get(), 1., y_dot_state.get(), alpha, 1.);
        if (jacobian_diagnostics_enabled) matrix_diagnostics = analyze_jacobian_matrix(jacobian);
        if (!std::isfinite(jacobian.frobenius_norm())) {
          factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-jacobian", &y, &y_dot, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }
        linSolver.init(jacobian);

        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent jacobian_event{
            .topic = progress_topics::jacobian, .time = t, .duration_ms = calc_timer.lap(), .minimum_verbosity = 2};
        current_diagnostics.append_to(jacobian_event);
        this->log.progress(jacobian_event);

        factorize_with_diagnostics(linSolver, jacobian, factorization_diagnostics, jacobian_diagnostics_enabled);
        if (factorization_diagnostics.factorization_success == 1.) {
          const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
          ProgressEvent event{.topic = progress_topics::factorization,
                              .time = t,
                              .duration_ms = calc_timer.lap(),
                              .minimum_verbosity = 3};
          current_diagnostics.append_to(event);
          this->log.progress(event);
        }
        record_diagnostics();
      } catch (std::exception &e) {
        if (jacobian_diagnostics_enabled && matrix_diagnostics.n_rows == 0.)
          matrix_diagnostics = analyze_jacobian_matrix(jacobian);
        if constexpr (decltype(linSolver)::performs_factorization)
          if (std::isnan(factorization_diagnostics.factorization_success))
            factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        callback_diagnostics.jacobian_failures++;
        ++failure_counter;
        callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, nullptr, e.what());
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            nullptr);
      failure_counter = 0;
      return agreed_ida_result(assembler.get_communicator(), false);
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const VectorType &src, VectorType &dst, const double tol) -> int {
      CalcDtTimer calc_timer;
      try {
        if (!is_finite_vector(src)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-source", &src, &dst, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }

        const auto sol_iterations = linSolver.solve(src, dst, tol);
        if (!is_finite_vector(dst)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-solution", &src, &dst, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }
        // A direct solver reports no iterations (-1), but its time counts all the same.
        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent event{.topic = progress_topics::implicit_linear_solve,
                            .time = stuck_t,
                            .duration_ms = calc_timer.lap(),
                            .iterations = sol_iterations,
                            .minimum_verbosity = 2};
        current_diagnostics.append_to(event);
        this->log.progress(event);
      } catch (std::exception &) {
        callback_diagnostics.linear_solver_failures++;
        ++failure_counter;
        callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "exception",
                              &src, &dst, nullptr);
        return agreed_ida_result(assembler.get_communicator(), true);
      }
      callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "success",
                            &src, &dst, nullptr);
      return agreed_ida_result(assembler.get_communicator(), false);
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(y, y_dot);
    } catch (const std::exception &e) {
      this->log.error("Timestepping failed: {}", e.what());
      output_after_failure = true;
      this->finalize_output_after_failure([&]() { time_stepper.output_step(stuck_t, y, y_dot, 0); });
      throw;
    }

    initial_data = y;
    this->drain_output();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA_impl<VectorType, SparseMatrixType, dim, LinearSolver>::run(BlockVectorType &initial_data,
                                                                                          const double t_start,
                                                                                          const double t_stop)
  {
    if (initial_data.n_blocks() != 2)
      throw std::runtime_error("TimeStepperSUNDIALS_ARKode_vars::run: y must have two blocks!");
    if (initial_data.block(1).size() == 0)
      throw std::runtime_error(
          "TimeStepperSUNDIALS_ARKode_vars::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    SparseMatrixType spatial_jacobian;
    assembler.reinit_matrix(spatial_jacobian);
    LinearSolver<SparseMatrixType, VectorType> linSolver;
    linSolver.set_report_port(this->log);
    if constexpr (requires { linSolver.set_iterative_refinement(true); })
      linSolver.set_iterative_refinement(this->impl.iterative_refinement);
    const uint n_FE_dofs = initial_data.block(0).size();
    const uint n_vars = initial_data.block(1).size();
    FullMatrix<NumberType> variable_jacobian(n_vars);
    FullMatrix<NumberType> variable_jacobian_inverse(n_vars);
    const bool jacobian_diagnostics_enabled = impl.jacobian_diagnostics;
    const DiagnosticPort jacobian_diagnostic_port =
        jacobian_diagnostics_enabled ? data_out.diagnostic_port() : DiagnosticPort{};

    // Create a SUNDIALS IDA object with the right settings
    typename SUNDIALS::IDA<BlockVectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt,
                                                                     impl.minimal_dt, 5, impl.max_non_linear_iterations,
                                                                     0, impl.abs_tol, impl.rel_tol);
    typename SUNDIALS::IDA<BlockVectorType> time_stepper(ida_data);

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = t_start;
    uint failure_counter = 0;
    IDACallbackDiagnostics callback_diagnostics;
    IDACallbackTrace callback_trace(impl.ida_callback_trace, impl.ida_callback_trace_min_t,
                                    impl.ida_callback_trace_max_lines, impl.ida_callback_trace_successes,
                                    this->log.Lambda(), this->log);

    // Initialize initial condition
    BlockVectorType y = initial_data;
    BlockVectorType y_dot = initial_data;
    y_dot *= 0.;

    // Pointer to current residual for monitoring. See the VectorType overload
    // for why this needs a placeholder before SUNDIALS::IDA's first
    // `output_step` call at `t = 0`.
    BlockVectorType residual_placeholder = initial_data;
    residual_placeholder *= 0.;
    BlockVectorType *residual = &residual_placeholder;

    // Tells SUNDIALS to do an internal reset, e.g. if we do local refinement
    time_stepper.solver_should_restart = [&](const double t, BlockVectorType &sol, BlockVectorType &sol_dot) -> bool {
      if (adaptor(t, sol.block(0))) {
        assembler.reinit_vector(sol_dot.block(0));
        assembler.reinit_matrix(spatial_jacobian);
        return true;
      }
      return false;
    };

    time_stepper.differential_components = [&]() {
      IndexSet dof_indices = assembler.get_differential_indices();
      IndexSet differential_indices(n_FE_dofs + n_vars);
      differential_indices.add_indices(dof_indices.begin(), dof_indices.end());
      differential_indices.add_range(n_FE_dofs, n_FE_dofs + n_vars);
      return differential_indices;
    };

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](BlockVectorType &v) {
      v.reinit(2);
      assembler.reinit_vector(v.block(0));
      reinit_la_variables_vector(v.block(1), n_vars, assembler.get_communicator());
      v.collect_sizes();
    };

    // Fully-replicated read-only views for the output path; see SolutionView. The variables block
    // needs its own layout -- rank 0 owns it outright, so a dof-shaped view would not fit it.
    SolutionView<VectorType> sol_view, vars_view, sol_dot_view, residual_view;
    assembler.reinit_solution_view(sol_view);
    assembler.reinit_solution_view(sol_dot_view);
    assembler.reinit_solution_view(residual_view);
    reinit_variables_view(vars_view, n_vars, assembler.get_communicator());

    // Replicated views of the state handed to the assemblers.
    //
    // IDA's vectors hold only this rank's rows. An assembler visiting one of its own cells reads
    // EVERY dof of that cell, and on a partition boundary some of those belong to a neighbour.
    // deal.II's non-ghosted read path is `ptr[index - local_begin]` with no bounds check, so such a
    // read does not fail -- it silently returns whatever is next in memory. The matrix survives
    // that (for a linear flux the Jacobian does not depend on the state at all, which is exactly
    // why the Jacobian diagnostics matched across rank counts while the residual did not), but the
    // residual is quietly wrong and IDA's Newton then fails to converge.
    //
    // The variables block needs the same treatment for a different reason: rank 0 owns it outright,
    // so on every other rank y.block(1) is locally *empty* and the model cannot read a single one of
    // its own variables from it.
    SolutionView<VectorType> y_state, y_dot_state, y_vars_state;
    assembler.reinit_solution_view(y_state);
    assembler.reinit_solution_view(y_dot_state);
    reinit_variables_view(y_vars_state, n_vars, assembler.get_communicator());

    // Scratch for the redundant variables computation; see compute_variables_into.
    VectorType vars_scratch;
    reinit_local_variables_vector(vars_scratch, n_vars);

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    // SUNDIALS::IDA::solve_dae reports the initial condition as t = 0, whatever the start time is.
    // The first call is always that one, so it is relabelled to the actual start of the run.
    bool initial_output = true;
    time_stepper.output_step = [&](const double t_reported, const BlockVectorType &sol, const BlockVectorType &sol_dot,
                                   unsigned int /*step_number*/) {
      const double t = std::exchange(initial_output, false) ? t_start : t_reported;
      if (!is_close(last_save, t, 1e-10)) {
        assembler.set_time(t);
        // Refreshed here, OUTSIDE write_frame: refreshing a ghosted replica communicates, while
        // write_frame runs its contributor on rank 0 only. Doing it inside would have rank 0 enter a
        // collective the other ranks never reach.
        sol_view.refresh(sol.block(0));
        vars_view.refresh(sol.block(1));
        sol_dot_view.refresh(sol_dot.block(0));
        residual_view.refresh((*residual).block(0));
        data_out.write_frame(t, [&](auto &frame) {
          assembler.attach_data_output(frame, sol_view.get(), vars_view.get(), sol_dot_view.get(), residual_view.get());
        });

        last_save = t;
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const BlockVectorType &y, const BlockVectorType &y_dot,
                                BlockVectorType &res) -> int {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t, impl.minimal_dt))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (failure_counter == 0 && !is_close(t, t_start) && stuck > 100) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter == 0 && is_close(t, t_start) && stuck > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "failure-limit", &y,
                              &y_dot, &res);
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
        callback_diagnostics.nonfinite_solution_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-y", &y,
                              &y_dot, &res);
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      try {
        res = 0;
        assembler.set_time(t);
        y_state.refresh(y.block(0));
        y_dot_state.refresh(y_dot.block(0));
        y_vars_state.refresh(y.block(1));
        compute_variables_into(res.block(1), vars_scratch, [&](VectorType &out) {
          assembler.residual_variables(out, y_vars_state.get(), y_state.get());
        });
        assembler.residual(res.block(0), y_state.get(), 1., y_dot_state.get(), 1., y_vars_state.get());
        res.block(1) += y_dot.block(1);
        residual = &res;
      } catch (std::exception &e) {
        callback_diagnostics.residual_exceptions++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, &res, e.what());
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      if (!std::isfinite(res.l1_norm())) {
        callback_diagnostics.nonfinite_residual_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-residual",
                              &y, &y_dot, &res);
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
      ProgressEvent event{.topic = progress_topics::implicit_residual,
                          .time = t,
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1};
      current_diagnostics.append_to(event);
      this->log.progress(event);
      callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            &res);

      failure_counter = 0;
      return agreed_ida_result(assembler.get_communicator(), false);
    };
    // Calculate the jacobian d(y_dot + F(y))/dy + d(y_dot*alpha)/dy_dot
    time_stepper.setup_jacobian = [&](const double t, const BlockVectorType &y, const BlockVectorType &y_dot,
                                      const double alpha) -> int {
      CalcDtTimer calc_timer;
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      const auto build_diagnostics =
          make_ida_jacobian_build_diagnostics(time_stepper, this->next_jacobian_build_id++, alpha);
      JacobianMatrixDiagnostics spatial_matrix_diagnostics;
      JacobianMatrixDiagnostics variable_matrix_diagnostics;
      spatial_matrix_diagnostics.n_rows = spatial_jacobian.m();
      variable_matrix_diagnostics.n_rows = variable_jacobian.m();
      JacobianFactorizationDiagnostics spatial_factorization_diagnostics;
      JacobianFactorizationDiagnostics variable_factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "jacobian_diagnostics.csv", t, build_diagnostics,
                                    spatial_matrix_diagnostics, spatial_factorization_diagnostics);
        record_jacobian_diagnostics(jacobian_diagnostic_port, "variable_jacobian_diagnostics.csv", t, build_diagnostics,
                                    variable_matrix_diagnostics, variable_factorization_diagnostics);
      };

      try {
        if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
          spatial_factorization_diagnostics.factorization_success = 0.;
          variable_factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-state",
                                &y, &y_dot, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }

        spatial_jacobian = 0;
        variable_jacobian = 0;
        assembler.set_time(t);
        y_state.refresh(y.block(0));
        y_dot_state.refresh(y_dot.block(0));
        y_vars_state.refresh(y.block(1));
        assembler.jacobian(spatial_jacobian, y_state.get(), 1., y_dot_state.get(), alpha, 1., y_vars_state.get());
        assembler.jacobian_variables(variable_jacobian, y_vars_state.get(), y_state.get());
        variable_jacobian *= -1.;
        variable_jacobian.diagadd(alpha);
        if (jacobian_diagnostics_enabled) {
          spatial_matrix_diagnostics = analyze_jacobian_matrix(spatial_jacobian);
          variable_matrix_diagnostics = analyze_jacobian_matrix(variable_jacobian);
        }

        if (!std::isfinite(spatial_jacobian.frobenius_norm()) || !std::isfinite(variable_jacobian.frobenius_norm())) {
          spatial_factorization_diagnostics.factorization_success = 0.;
          variable_factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-jacobian", &y, &y_dot, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }

        linSolver.init(spatial_jacobian);

        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent jacobian_event{
            .topic = progress_topics::jacobian, .time = t, .duration_ms = calc_timer.lap(), .minimum_verbosity = 2};
        current_diagnostics.append_to(jacobian_event);
        this->log.progress(jacobian_event);

        factorize_with_diagnostics(linSolver, spatial_jacobian, spatial_factorization_diagnostics,
                                   jacobian_diagnostics_enabled);
        const auto variable_factorization_start = std::chrono::steady_clock::now();
        try {
          variable_jacobian_inverse.invert(variable_jacobian);
          variable_factorization_diagnostics.factorization_success = 1.;
        } catch (...) {
          variable_factorization_diagnostics.factorization_success = 0.;
          variable_factorization_diagnostics.factorization_ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - variable_factorization_start)
                  .count();
          throw;
        }
        variable_factorization_diagnostics.factorization_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - variable_factorization_start)
                .count();
        const auto current_diagnostics_after_inversion =
            make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent factorization_event{.topic = progress_topics::factorization,
                                          .time = t,
                                          .duration_ms = calc_timer.lap(),
                                          .minimum_verbosity = 3};
        current_diagnostics_after_inversion.append_to(factorization_event);
        this->log.progress(factorization_event);
        record_diagnostics();
      } catch (std::exception &e) {
        if constexpr (decltype(linSolver)::performs_factorization)
          if (std::isnan(spatial_factorization_diagnostics.factorization_success))
            spatial_factorization_diagnostics.factorization_success = 0.;
        if (std::isnan(variable_factorization_diagnostics.factorization_success))
          variable_factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        callback_diagnostics.jacobian_failures++;
        ++failure_counter;
        callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, nullptr, e.what());
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            nullptr);
      failure_counter = 0;
      return agreed_ida_result(assembler.get_communicator(), false);
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const BlockVectorType &src, BlockVectorType &dst, const double tol) -> int {
      CalcDtTimer calc_timer;
      try {
        if (!is_finite_vector(src)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-source", &src, &dst, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }

        const auto sol_iterations = linSolver.solve(src.block(0), dst.block(0), tol);
        dense_vmult_variables(variable_jacobian_inverse, dst.block(1), src.block(1), assembler.get_communicator());
        if (!is_finite_vector(dst)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-solution", &src, &dst, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }

        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent event{.topic = progress_topics::implicit_linear_solve,
                            .time = stuck_t,
                            .duration_ms = calc_timer.lap(),
                            .iterations = sol_iterations,
                            .minimum_verbosity = 2};
        current_diagnostics.append_to(event);
        this->log.progress(event);
      } catch (std::exception &) {
        callback_diagnostics.linear_solver_failures++;
        ++failure_counter;
        callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "exception",
                              &src, &dst, nullptr);
        return agreed_ida_result(assembler.get_communicator(), true);
      }
      callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "success",
                            &src, &dst, nullptr);
      return agreed_ida_result(assembler.get_communicator(), false);
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(y, y_dot);
    } catch (const std::exception &e) {
      this->log.error("Timestepping failed: {}", e.what());
      this->finalize_output_after_failure([&]() { time_stepper.output_step(stuck_t, y, y_dot, 0); });
      throw;
    }

    initial_data = y;
    this->drain_output();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA_impl<VectorType, SparseMatrixType, dim, LinearSolver>::run_vars(VectorType &initial_data,
                                                                                               const double t_start,
                                                                                               const double t_stop)
  {
    if (initial_data.size() == 0)
      throw std::runtime_error(
          "TimeStepperSUNDIALS_IDA_impl::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    const uint n_vars = initial_data.size();
    FullMatrix<NumberType> variable_jacobian(n_vars);
    FullMatrix<NumberType> variable_jacobian_inverse(n_vars);
    const bool jacobian_diagnostics_enabled = impl.jacobian_diagnostics;
    const DiagnosticPort jacobian_diagnostic_port =
        jacobian_diagnostics_enabled ? data_out.diagnostic_port() : DiagnosticPort{};

    // Create a SUNDIALS IDA object with the right settings
    typename SUNDIALS::IDA<VectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5,
                                                                impl.max_non_linear_iterations, 0, impl.abs_tol,
                                                                impl.rel_tol);
    typename SUNDIALS::IDA<VectorType> time_stepper(ida_data);

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = t_start;
    uint failure_counter = 0;
    IDACallbackDiagnostics callback_diagnostics;
    IDACallbackTrace callback_trace(impl.ida_callback_trace, impl.ida_callback_trace_min_t,
                                    impl.ida_callback_trace_max_lines, impl.ida_callback_trace_successes,
                                    this->log.Lambda(), this->log);
    IDAErrorDofMonitor<VectorType> error_dof_monitor(
        impl.ida_error_dof_diagnostics, impl.ida_error_dof_diagnostics_top_n, this->log.Lambda(), this->log);
    bool output_after_failure = false;

    // Initialize initial condition
    VectorType y = initial_data;
    VectorType y_dot = initial_data;
    y_dot *= 0.;

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](VectorType &v) {
      reinit_la_variables_vector(v, n_vars, assembler.get_communicator());
    };

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    // SUNDIALS::IDA::solve_dae reports the initial condition as t = 0, whatever the start time is.
    // The first call is always that one, so it is relabelled to the actual start of the run.
    bool initial_output = true;
    time_stepper.output_step = [&](const double t_reported, const VectorType &sol, const VectorType & /*sol_dot*/,
                                   uint /*step_number*/) {
      const double t = std::exchange(initial_output, false) ? t_start : t_reported;
      if (!is_close(last_save, t, 1e-10)) {
        assembler.set_time(t);
        // dim == 0 has no FE space and the variables vector is serial by construction, so no
        // gather is needed here -- the view would be a passthrough.
        data_out.write_frame(t, [&](auto &frame) { assembler.attach_data_output(frame, VectorType(), sol); });

        last_save = t;
      }
      if (!output_after_failure && !is_close(t, t_start)) {
        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        error_dof_monitor.observe(t, time_stepper, current_diagnostics, sol, ida_error_dof_callback);
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &res) -> int {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t, impl.minimal_dt))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (failure_counter == 0 && !is_close(t, t_start) && stuck > 100) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter == 0 && is_close(t, t_start) && stuck > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "stuck", &y, &y_dot,
                              &res);
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }
      if (failure_counter > 200) {
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "failure-limit", &y,
                              &y_dot, &res);
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
        callback_diagnostics.nonfinite_solution_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-y", &y,
                              &y_dot, &res);
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      try {
        res = 0;
        assembler.set_time(t);
        assembler.residual_variables(res, y, VectorType());
        res += y_dot;
      } catch (std::exception &e) {
        callback_diagnostics.residual_exceptions++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, &res, e.what());
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      if (!std::isfinite(res.l1_norm())) {
        callback_diagnostics.nonfinite_residual_failures++;
        ++failure_counter;
        callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-residual",
                              &y, &y_dot, &res);
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
      ProgressEvent event{.topic = progress_topics::implicit_residual,
                          .time = t,
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1};
      current_diagnostics.append_to(event);
      this->log.progress(event);
      callback_trace.record("residual", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            &res);

      failure_counter = 0;
      return agreed_ida_result(assembler.get_communicator(), false);
    };
    // Calculate the jacobian d(y_dot + F(y))/dy + d(y_dot*alpha)/dy_dot
    time_stepper.setup_jacobian = [&](const double t, const VectorType &y, const VectorType &y_dot,
                                      const double alpha) -> int {
      CalcDtTimer calc_timer;
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      const auto build_diagnostics =
          make_ida_jacobian_build_diagnostics(time_stepper, this->next_jacobian_build_id++, alpha);
      JacobianMatrixDiagnostics matrix_diagnostics;
      matrix_diagnostics.n_rows = variable_jacobian.m();
      JacobianFactorizationDiagnostics factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "variable_jacobian_diagnostics.csv", t, build_diagnostics,
                                    matrix_diagnostics, factorization_diagnostics);
      };

      try {
        if (!is_finite_vector(y) || !is_finite_vector(y_dot)) {
          factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "nonfinite-state",
                                &y, &y_dot, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }

        variable_jacobian = 0;
        assembler.set_time(t);
        assembler.jacobian_variables(variable_jacobian, y, VectorType());
        variable_jacobian *= -1.;
        variable_jacobian.diagadd(alpha);
        if (jacobian_diagnostics_enabled) matrix_diagnostics = analyze_jacobian_matrix(variable_jacobian);

        if (!std::isfinite(variable_jacobian.frobenius_norm())) {
          factorization_diagnostics.factorization_success = 0.;
          record_diagnostics();
          callback_diagnostics.jacobian_failures++;
          ++failure_counter;
          callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-variable-jacobian", &y, &y_dot, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }

        const auto factorization_start = std::chrono::steady_clock::now();
        try {
          variable_jacobian_inverse.invert(variable_jacobian);
          factorization_diagnostics.factorization_success = 1.;
        } catch (...) {
          factorization_diagnostics.factorization_success = 0.;
          factorization_diagnostics.factorization_ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - factorization_start).count();
          throw;
        }
        factorization_diagnostics.factorization_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - factorization_start).count();
        record_diagnostics();
      } catch (std::exception &e) {
        if (std::isnan(factorization_diagnostics.factorization_success))
          factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        callback_diagnostics.jacobian_failures++;
        ++failure_counter;
        callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "exception", &y,
                              &y_dot, nullptr, e.what());
        return agreed_ida_result(assembler.get_communicator(), true);
      }

      const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
      ProgressEvent event{
          .topic = progress_topics::jacobian, .time = t, .duration_ms = calc_timer.lap(), .minimum_verbosity = 2};
      current_diagnostics.append_to(event);
      this->log.progress(event);
      callback_trace.record("jacobian", t, failure_counter, time_stepper, callback_diagnostics, "success", &y, &y_dot,
                            nullptr);

      failure_counter = 0;
      return agreed_ida_result(assembler.get_communicator(), false);
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const VectorType &src, VectorType &dst, const double) -> int {
      try {
        if (!is_finite_vector(src)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-source", &src, &dst, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }

        dense_vmult_variables(variable_jacobian_inverse, dst, src, assembler.get_communicator());
        if (!is_finite_vector(dst)) {
          callback_diagnostics.linear_solver_failures++;
          ++failure_counter;
          callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics,
                                "nonfinite-solution", &src, &dst, nullptr);
          return agreed_ida_result(assembler.get_communicator(), true);
        }
      } catch (std::exception &) {
        callback_diagnostics.linear_solver_failures++;
        ++failure_counter;
        callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "exception",
                              &src, &dst, nullptr);
        return agreed_ida_result(assembler.get_communicator(), true);
      }
      callback_trace.record("linear-solve", stuck_t, failure_counter, time_stepper, callback_diagnostics, "success",
                            &src, &dst, nullptr);
      return agreed_ida_result(assembler.get_communicator(), false);
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(y, y_dot);
    } catch (const std::exception &e) {
      this->log.error("Timestepping failed: {}", e.what());
      output_after_failure = true;
      this->finalize_output_after_failure([&]() { time_stepper.output_step(stuck_t, y, y_dot, 0); });
      throw;
    }

    initial_data = y;
    this->drain_output();
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                                    DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                    DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                    DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                    DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                                    DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                    DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                    DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                    DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                                    DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                    DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                    DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                    DiFfRG::GMRES>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                                    DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                    DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                    DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                    DiFfRG::ScaledUMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                                    DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                    DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                    DiFfRG::ScaledUMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                    DiFfRG::ScaledUMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                                    DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                    DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                    DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                    DiFfRG::ScaledGMRES>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                                    DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                    DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                    DiFfRG::ScaledGMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                    DiFfRG::ScaledGMRES>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                                    DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                    DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                    DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                    DiFfRG::GMRES>;

// ##############################################################################
// Distributed (PETSc-backed) instantiations
// ##############################################################################
//
// These compile and link as of Stage E+F. The route from 12.3 compiler errors per instantiation to
// zero, measured after each change:
//
//     after Stage B                    10.0   eager InverseSparseMatrixType alias; AbstractFlowingVariables
//     after the D2 discretization work 10.0   prerequisite only -- see below
//     assembler.reinit_matrix()        6.0   a PETSc matrix has no constructor from a bare pattern
//     reinit_la_variables_vector()      4.0   a PETSc vector needs (owned IndexSet, comm), not a size
//     VectorType() for empty arguments  2.0   four `Vector<double>()` literals in generic code
//     dense_vmult_variables()           0.0   FullMatrix applied to the rank-0-owned variables block
//
// The flat step is worth remembering: the whole discretization layer (index sets, the la_policy
// seam, all four assemblers) moved this number not at all, because every blocked line was in THIS
// file, constructing its own matrices and vectors rather than asking the assembler for them. The
// count measures the timestepper body, not the work that unblocked it.
//
// The purely explicit steppers stay serial deliberately: they are not viable for these stiff flows
// anyway, and they integrate the whole state through Eigen. (The Boost-hybrid IDA steppers are
// distributed: their explicit part only ever holds the small, replicated variables.) dim == 0 is
// pure variables and has no FE space to distribute.

#ifdef DEAL_II_WITH_PETSC
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::PETScWrappers::MPI::Vector,
                                                    dealii::PETScWrappers::MPI::SparseMatrix, 1, DiFfRG::PETScKrylov>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::PETScWrappers::MPI::Vector,
                                                    dealii::PETScWrappers::MPI::SparseMatrix, 2, DiFfRG::PETScKrylov>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::PETScWrappers::MPI::Vector,
                                                    dealii::PETScWrappers::MPI::SparseMatrix, 3, DiFfRG::PETScKrylov>;
#endif

// The default-solver spelling. DefaultLinearSolver is deliberately an indirect alias and so
// stays a distinct template argument from UMFPack on every compiler, which means
// TimeStepper<Assembler> needs its own instantiations alongside TimeStepper<Assembler, UMFPack>.
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                                    DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                    DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                    DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                    DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                                    DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                    DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                    DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                    DiFfRG::DefaultLinearSolver>;

// ##############################################################################
// Distributed instantiations for the default solver
// ##############################################################################
//
// An MPI build's default mesh is partitioned and its default linear algebra is PETSc-backed, so
// TimeStepperSUNDIALS_IDA<Assembler> resolves here rather than to any of the serial rows above.
// PETScDirect is instantiated in its own right as well: DefaultLinearSolver names it where PETSc
// was built with MUMPS, but an application may also ask for it explicitly.
#ifdef DEAL_II_WITH_PETSC
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 1, DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 2, DiFfRG::DefaultLinearSolver>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 3, DiFfRG::DefaultLinearSolver>;

#ifdef DEAL_II_PETSC_WITH_MUMPS
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::PETScWrappers::MPI::Vector,
                                                    dealii::PETScWrappers::MPI::SparseMatrix, 1, DiFfRG::PETScDirect>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::PETScWrappers::MPI::Vector,
                                                    dealii::PETScWrappers::MPI::SparseMatrix, 2, DiFfRG::PETScDirect>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_impl<dealii::PETScWrappers::MPI::Vector,
                                                    dealii::PETScWrappers::MPI::SparseMatrix, 3, DiFfRG::PETScDirect>;
#endif
#endif
