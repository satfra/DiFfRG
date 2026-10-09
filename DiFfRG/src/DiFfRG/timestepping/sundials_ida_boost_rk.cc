// external libraries
#include <algorithm>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <cstddef>
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/sundials/ida.h>
#include <limits>
#include <utility>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/discretization/common/solution_view.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/ida_rank_agreement.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/PETScDirect.hh>
#include <DiFfRG/timestepping/linear_solver/PETScKrylov.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/local_tolerances.hh>
#include <DiFfRG/timestepping/sundials_diagnostics.hh>
#include <DiFfRG/timestepping/sundials_ida_boost_rk.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver,
            int prec>
  void TimeStepperSUNDIALS_IDA_BoostRK_impl<VectorType, SparseMatrixType, dim, LinearSolver, prec>::run_segment(
      AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start, const double t_stop)
  {

    auto &full_data = initial_condition.data();
    if (full_data.n_blocks() != 1)
      run(full_data, t_start, t_stop);
    else
      throw std::runtime_error(
          "TimeStepperSUNDIALS_IDA_BoostRK_impl::run: initial condition must have at least two blocks!");
  }

  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver,
            int prec>
  void TimeStepperSUNDIALS_IDA_BoostRK_impl<VectorType, SparseMatrixType, dim, LinearSolver, prec>::run(
      BlockVectorType &initial_data, const double t_start, const double t_stop)
  {
    if (initial_data.n_blocks() != 2)
      throw std::runtime_error("TimeStepperSUNDIALS_BoostRK::run: y must have two blocks!");
    if (initial_data.block(1).size() == 0)
      throw std::runtime_error(
          "TimeStepperSUNDIALS_BoostRK::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    SparseMatrixType spatial_jacobian;
    assembler.reinit_matrix(spatial_jacobian);
    LinearSolver<SparseMatrixType, VectorType> linSolver;
    linSolver.set_report_port(this->log);
    if constexpr (requires { linSolver.set_iterative_refinement(true); })
      linSolver.set_iterative_refinement(this->impl.iterative_refinement);
    const bool jacobian_diagnostics_enabled = impl.jacobian_diagnostics;
    const DiagnosticPort jacobian_diagnostic_port =
        jacobian_diagnostics_enabled ? data_out.diagnostic_port() : DiagnosticPort{};
    const MPI_Comm comm = assembler.get_communicator();
    const uint n_vars = initial_data.block(1).size();

    // Create a SUNDIALS IDA object with the right settings for spatial data
    typename SUNDIALS::IDA<VectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5,
                                                                impl.max_non_linear_iterations, 0, impl.abs_tol,
                                                                impl.rel_tol);
    typename SUNDIALS::IDA<VectorType> time_stepper(ida_data);

    // Initialize initial condition
    VectorType spatial_y = initial_data.block(0);
    VectorType spatial_y_dot = initial_data.block(0);

    // The explicit variables are integrated redundantly on every rank, so every rank holds all of them in
    // process-local vectors. Under MPI the variables block of the state is owned by rank 0 alone and is read through
    // a replicated view; serially the view is a passthrough and these are plain vectors as before.
    SolutionView<VectorType> initial_variables_view;
    reinit_variables_view(initial_variables_view, n_vars, comm);
    initial_variables_view.refresh(initial_data.block(1));
    const VectorType initial_variables = initial_variables_view.get();
    VectorType variable_y = initial_variables;

    // Buffers for the explicit right hand side: the variables it is evaluated at, its result, and the spatial state
    // it couples to -- interpolated in time (spatial_eval) and replicated for the assembler (spatial_eval_view).
    VectorType variable_eval = initial_variables;
    VectorType variable_dy = initial_variables;
    VectorType spatial_eval = spatial_y;
    SolutionView<VectorType> spatial_eval_view;
    assembler.reinit_solution_view(spatial_eval_view);

    // dx/dt of the explicit variables at x, coupled to the spatial state currently in spatial_eval.
    //
    // Under MPI every rank evaluates this from identical inputs, but the model's reductions may still sum in a
    // rank-dependent order. Rank 0's result is broadcast so the ranks stay bitwise identical: the explicit stepper
    // branches on these values, and ranks taking different steps would enter the next collective out of step.
    auto evaluate_variable_rhs = [&](const Eigen::VectorXd &x, Eigen::VectorXd &dxdt) {
      eigen_to_dealii(x, variable_eval);
      variable_dy = 0;
      spatial_eval_view.refresh(spatial_eval);
      assembler.residual_variables(variable_dy, variable_eval, spatial_eval_view.get());
      // The model assigns entry by entry (=, as everywhere for the variables); a PETSc vector stages those writes.
      variable_dy.compress(VectorOperation::insert);
      dealii_to_eigen(variable_dy, dxdt);
      dxdt *= -1;
      if constexpr (is_distributed_la<VectorType>)
        MPI::bcast(comm, dxdt.data(), static_cast<size_t>(dxdt.size()) * sizeof(double), /*root=*/0);
    };

    // The explicit variables are integrated "on demand" with the spatial solution held
    // fixed over each IDA trial step. Rather than pinning it to the step's right endpoint
    // -- which biases the explicit variable, because the coupling then always lags or
    // leads the true spatial trajectory -- the spatial solution is linearly interpolated
    // across the step between these two endpoints: spatial_lo at spatial_lo_time (the
    // start of the segment) and spatial_hi at spatial_hi_time (its end).
    VectorType spatial_lo = spatial_y;
    VectorType spatial_hi = spatial_y;
    double spatial_lo_time = t_start;
    double spatial_hi_time = t_start;

    using namespace boost::numeric::odeint;
    auto variable_stepper = make_controlled<error_stepper_type>(expl.abs_tol, expl.rel_tol);

    auto get_variable_residual = [&](const Eigen::VectorXd &x, Eigen::VectorXd &dxdt, const double t) {
      CalcDtTimer calc_timer;

      // linearly interpolate the spatial solution across the current segment so the
      // coupling tracks the spatial trajectory instead of being pinned to one endpoint
      double alpha =
          (spatial_hi_time > spatial_lo_time) ? (t - spatial_lo_time) / (spatial_hi_time - spatial_lo_time) : 1.;
      alpha = std::clamp(alpha, 0., 1.);
      spatial_eval = spatial_lo;
      spatial_eval *= (1. - alpha);
      spatial_eval.add(alpha, spatial_hi);

      assembler.set_time(t);
      evaluate_variable_rhs(x, dxdt);
      if (!dxdt.allFinite()) throw std::runtime_error("TimeStepperBoostRK_impl::run_vars: dy is not finite!");

      this->log.progress({.topic = progress_topics::explicit_residual,
                          .time = t,
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1});
    };

    Eigen::VectorXd variable_sol;
    Eigen::VectorXd variable_ret;
    // Trajectory of the explicit variables, sampled at the times the explicit stepper
    // visited. All entries are committed -- they correspond to IDA trial steps that have
    // been accepted. No speculative entries are ever appended, so a rejected trial step
    // cannot pollute the buffer.
    std::vector<Eigen::VectorXd> variable_buffer;
    std::vector<double> variable_buffer_times;
    // End time of the last accepted IDA trial step that has been *folded into the
    // committed buffer*. The committed segments cover [t_start, t_committed].
    double t_committed = t_start;
    // Latest IDA trial endpoint that has been confirmed accepted but not yet folded
    // into the committed buffer. We defer running an explicit substep until the gap
    // (t_pending - t_committed) is at least cur_dt -- this decouples the explicit
    // stepper's call rate from IDA's internal step count and keeps the (heavy)
    // dt_variables evaluation cadence at the explicit timescale even when IDA collapses
    // its dt to << cur_dt at marginal points.
    double t_pending = t_start;
    VectorType spatial_pending = spatial_y;
    // End time of the IDA trial step the controller is currently solving. A later
    // request at a strictly larger time means that trial step was accepted (the trial's
    // converged spatial state is then transferred to spatial_pending); a request at a
    // smaller time means IDA rejected it (spatial_pending is left untouched).
    double frontier = -std::numeric_limits<double>::infinity();
    // Cached values of dv/dt at the two most recent committed points. dv_committed is
    // estimated by backward-differencing the two most recent buffer entries (no extra
    // dt_variables call is required); dv_prev_committed is the dv_committed value
    // cached one commit ago. Together they drive a quadratic predictor used to serve
    // `v` to IDA during Newton iterations on a speculative trial step. The predictor is
    // deterministic (it does not depend on IDA's still-unconverged trial spatial guess),
    // which is what Newton needs to converge consistently when the FEM equation reads
    // `v` back through an extractor.
    Eigen::VectorXd dv_committed;
    Eigen::VectorXd dv_prev_committed;
    double t_prev_committed = -std::numeric_limits<double>::infinity();
    bool have_prev_dv = false;
    double cur_dt = expl.dt;

    // Linearly interpolate the buffered trajectory to time t (clamped to the buffer range).
    auto interpolate_buffer = [&](VectorType &variable_y, const double t) {
      if (t <= variable_buffer_times.front()) {
        eigen_to_dealii(variable_buffer.front(), variable_y);
        return;
      }
      if (t >= variable_buffer_times.back()) {
        eigen_to_dealii(variable_buffer.back(), variable_y);
        return;
      }
      std::size_t idx = 0;
      for (std::size_t i = 0; i + 1 < variable_buffer_times.size(); ++i)
        if (variable_buffer_times[i] <= t && t <= variable_buffer_times[i + 1]) idx = i;
      const double t_prev = variable_buffer_times[idx];
      const double t_next = variable_buffer_times[idx + 1];
      const double alpha = (t - t_prev) / (t_next - t_prev);
      variable_ret = alpha * variable_buffer[idx + 1] + (1. - alpha) * variable_buffer[idx];
      eigen_to_dealii(variable_ret, variable_y);
    };

    // Refresh the predictor's cached dv/dt by evaluating the variable residual at the
    // current committed point. Costs ONE dt_variables call per commit (commits happen
    // once per cur_dt of accepted t under the batching scheme).
    auto refresh_dv_committed_from_buffer = [&]() {
      if (dv_committed.size() > 0) {
        if (dv_prev_committed.size() != dv_committed.size()) dv_prev_committed.resize(dv_committed.size());
        dv_prev_committed = dv_committed;
        t_prev_committed = t_committed;
        have_prev_dv = true;
      }
      const std::size_t N = variable_buffer.size();
      if (N == 0) return;
      spatial_eval = spatial_lo;
      assembler.set_time(t_committed);
      evaluate_variable_rhs(variable_buffer[N - 1], dv_committed);
    };

    // Quadratic predictor at time t. Falls back to linear before the first d2v
    // estimate is available.
    auto eval_predictor = [&](Eigen::VectorXd &out, const double t) {
      out = variable_buffer.back();
      const double dt = t - t_committed;
      out += dt * dv_committed;
      if (have_prev_dv && t_committed > t_prev_committed) {
        const double inv_dtc = 1. / (t_committed - t_prev_committed);
        out += (0.5 * dt * dt * inv_dtc) * (dv_committed - dv_prev_committed);
      }
    };

    // Re-integrate the explicit variables forward from the last committed time to time
    // `t` -- which is required to be the endpoint of an *accepted* IDA trial step --
    // using `accepted_spatial` as the spatial solution at `t`. The spatial solution is
    // linearly interpolated across the segment between `spatial_lo` (at `t_committed`)
    // and `accepted_spatial` (at `t`), both of which are converged values, so the
    // explicit integration sees a faithful spatial trajectory and never speculative
    // intermediate Newton iterates. Sub-steps are never allowed to overshoot t.
    auto commit_segment_to = [&](const VectorType &accepted_spatial, const double t) {
      spatial_hi = accepted_spatial;
      spatial_hi_time = t;
      // spatial_lo / spatial_lo_time were left at the previous committed point and are
      // the correct left endpoint for this segment.
      variable_sol = variable_buffer.back();
      double step_time = variable_buffer_times.back();
      failed_step_checker fail_checker;
      while (step_time < t && !is_close(step_time, t)) {
        controlled_step_result res;
        cur_dt = std::min(cur_dt, expl.maximal_dt);
        const bool capped = cur_dt > t - step_time;
        double dt = capped ? t - step_time : cur_dt;
        do {
          res = variable_stepper.try_step(get_variable_residual, variable_sol, step_time, dt);
          fail_checker();
        } while (res == fail);
        if (!capped) cur_dt = dt;
        variable_buffer.push_back(variable_sol);
        variable_buffer_times.push_back(step_time);
        fail_checker.reset();
      }
      // Slide the committed point forward
      t_committed = t;
      spatial_lo = accepted_spatial;
      spatial_lo_time = t;
      // Refresh the predictor used by the next speculative Newton solve. This costs
      // zero dt_variables calls -- the slope is read off the buffer.
      refresh_dv_committed_from_buffer();
    };

    // Serve the explicit variables at time t for a speculative IDA call (residual /
    // jacobian). The trial endpoint `frontier` is only committed once IDA confirms its
    // acceptance (signalled by a later request at a strictly larger time, in which case
    // the saved `spatial_hi` is the converged spatial state at `frontier`). Within a
    // single still-unaccepted Newton solve we serve a first-order predictor in v built
    // entirely from the last committed state -- this makes the FEM residual a
    // deterministic function of IDA's spatial trial state, which is required for Newton
    // convergence when the FEM source reads v through an extractor.
    auto request_variables = [&](VectorType &variable_y, const VectorType &spatial_y, const double t) {
      if (variable_buffer.empty()) {
        // Initialise at t = t_start; the initial condition is by construction accepted.
        variable_y = initial_variables;
        dealii_to_eigen(variable_y, variable_sol);
        variable_buffer.push_back(variable_sol);
        variable_buffer_times.push_back(t);
        t_committed = t;
        t_pending = t;
        frontier = t;
        spatial_lo = spatial_y;
        spatial_hi = spatial_y;
        spatial_pending = spatial_y;
        spatial_lo_time = t;
        spatial_hi_time = t;
        if (dv_committed.size() != static_cast<Eigen::Index>(variable_sol.size()))
          dv_committed.resize(variable_sol.size());
        dv_committed.setZero();
        assembler.set_time(t);
        return;
      }

      if (t > frontier && !is_close(t, frontier)) {
        // IDA advanced -> the previous trial at `frontier` was accepted. spatial_hi
        // (saved from the last call at `frontier`) is the converged spatial state
        // there; commit the segment immediately.
        if (frontier > t_committed && !is_close(frontier, t_committed)) commit_segment_to(spatial_hi, frontier);
        frontier = t;
        spatial_hi = spatial_y;
        spatial_hi_time = t;
      } else if (t < frontier && !is_close(t, frontier)) {
        // IDA rejected the previous trial at `frontier`; spatial_pending is left alone.
        frontier = t;
        spatial_hi = spatial_y;
        spatial_hi_time = t;
      } else {
        // Another residual / jacobian evaluation at the same trial endpoint; track the
        // latest spatial guess so that, if this trial is later accepted, the eventual
        // commit uses the converged spatial state.
        spatial_hi = spatial_y;
      }

      // Quadratic predictor (linear before the first d2v estimate is available). The
      // result is independent of `spatial_y`, so repeated Newton iterations at the same
      // trial endpoint (and even retries after rejection) all see the same v(t).
      eval_predictor(variable_ret, t);
      eigen_to_dealii(variable_ret, variable_y);
      assembler.set_time(t);
    };

    // Sample the explicit variables at an accepted time t (called from output_step and
    // at the end of the run). Any pending accepted-but-not-yet-committed segment is
    // folded into the committed buffer first regardless of the cur_dt threshold; the
    // buffer is then extended to t using the converged spatial state passed in.
    auto commit_variables = [&](VectorType &variable_y, const VectorType &spatial_y, const double t) {
      if (variable_buffer.empty()) {
        request_variables(variable_y, spatial_y, t);
        return;
      }
      if (frontier > t_committed && !is_close(frontier, t_committed)) commit_segment_to(spatial_hi, frontier);
      if (t > t_committed && !is_close(t, t_committed)) {
        commit_segment_to(spatial_y, t);
        if (t > frontier) frontier = t;
      }
      interpolate_buffer(variable_y, t);
      assembler.set_time(t);
    };

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = t_start;
    uint failure_counter = 0;
    IDACallbackDiagnostics callback_diagnostics;

    // Pointer to current residual for monitoring
    VectorType residual_placeholder = spatial_y;
    residual_placeholder *= 0.;
    VectorType *residual = &residual_placeholder;

    // Replicated views of the state for the assembler and the output path: IDA's vectors hold only this rank's rows,
    // while assembly and output read arbitrary dofs. See SolutionView and the same views in sundials_ida.cc.
    SolutionView<VectorType> y_state, y_dot_state, sol_view, sol_dot_view, residual_view;
    for (auto *view : {&y_state, &y_dot_state, &sol_view, &sol_dot_view, &residual_view})
      assembler.reinit_solution_view(*view);

    // Per-dof absolute tolerances, if the model sets them (def::HasAbsTolerances); see LocalAbsTolerances.
    LocalAbsTolerances<VectorType, SparseMatrixType, dim> local_tol(assembler, impl.abs_tol, impl.rel_tol,
                                                                    impl.local_tolerance_refresh);
    if (local_tol.init(spatial_y)) {
      time_stepper.get_local_tolerances = [&]() -> VectorType & { return local_tol.get(); };
      this->log.info("IDA: per-dof absolute tolerances from the model, {:.3e} .. {:.3e}", local_tol.min(),
                     local_tol.max());
    }

    // Tells SUNDIALS to do an internal reset, e.g. if we do local refinement, or if the per-dof tolerances drifted
    time_stepper.solver_should_restart = [&](const double t, VectorType &sol, VectorType &sol_dot) -> bool {
      if (adaptor(t, sol)) {
        assembler.reinit_vector(sol_dot);
        assembler.reinit_matrix(spatial_jacobian);
        for (auto *view : {&y_state, &y_dot_state, &sol_view, &sol_dot_view, &residual_view, &spatial_eval_view})
          assembler.reinit_solution_view(*view);
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

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    // SUNDIALS::IDA::solve_dae reports the initial condition as t = 0, whatever the start time is.
    // The first call is always that one, so it is relabelled to the actual start of the run.
    bool initial_output = true;
    time_stepper.output_step = [&](const double t_reported, const VectorType &sol, const VectorType &sol_dot,
                                   uint /*step_number*/) {
      const double t = std::exchange(initial_output, false) ? t_start : t_reported;
      if (!is_close(last_save, t, 1e-10)) {
        assembler.set_time(t);

        commit_variables(variable_y, sol, t);
        assembler.set_time(t);
        // Refreshed here, OUTSIDE write_frame: it runs its contributor on rank 0 only, and a refresh communicates.
        sol_view.refresh(sol);
        sol_dot_view.refresh(sol_dot);
        residual_view.refresh(*residual);
        data_out.write_frame(t, [&](auto &frame) {
          assembler.attach_data_output(frame, sol_view.get(), variable_y, sol_dot_view.get(), residual_view.get());
        });

        last_save = t;
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &res) -> int {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t, impl.minimal_dt * 1e-1))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (!is_close(t, t_start) && stuck > 100)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (is_close(t, t_start) && stuck > 200)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (failure_counter > 200) {
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!std::isfinite(y.l1_norm())) {
        callback_diagnostics.nonfinite_solution_failures++;
        return ++failure_counter;
      }

      bool failed = false;
      try {
        res = 0;
        assembler.set_time(t);
        const auto request_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent variables_event{.topic = progress_topics::variables, .time = t, .minimum_verbosity = 2};
        request_diagnostics.append_to(variables_event);
        this->log.progress(variables_event);
        request_variables(variable_y, y, t);
        y_state.refresh(y);
        y_dot_state.refresh(y_dot);
        assembler.residual(res, y_state.get(), 1., y_dot_state.get(), 1., variable_y);
        residual = &res;
      } catch (std::exception &e) {
        callback_diagnostics.residual_exceptions++;
        failed = true;
      }
      // An exception is local to the rank that threw; agree before the next collective.
      if (internal::agreed_ida_result(comm, failed) != 0) return ++failure_counter;

      if (!std::isfinite(res.l1_norm())) {
        callback_diagnostics.nonfinite_residual_failures++;
        return ++failure_counter;
      }

      const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
      ProgressEvent residual_event{.topic = progress_topics::implicit_residual,
                                   .time = t,
                                   .duration_ms = calc_timer.lap(),
                                   .minimum_verbosity = 1};
      current_diagnostics.append_to(residual_event);
      this->log.progress(residual_event);

      failure_counter = 0;
      return 0;
    };
    // Calculate the jacobian d(y_dot + F(y))/dy + d(y_dot*alpha)/dy_dot
    time_stepper.setup_jacobian = [&](const double t, const VectorType &y, const VectorType &y_dot,
                                      const double alpha) -> int {
      CalcDtTimer calc_timer;
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      const auto build_diagnostics = make_ida_jacobian_build_diagnostics(time_stepper, this->next_jacobian_build_id++,
                                                                         alpha, ImplicitTimestepperKind::ida_boost_rk);
      JacobianMatrixDiagnostics matrix_diagnostics;
      matrix_diagnostics.n_rows = spatial_jacobian.m();
      JacobianFactorizationDiagnostics factorization_diagnostics;
      bool diagnostics_recorded = false;
      const auto record_diagnostics = [&]() {
        if (diagnostics_recorded) return;
        diagnostics_recorded = true;
        record_jacobian_diagnostics(jacobian_diagnostic_port, "jacobian_diagnostics.csv", t, build_diagnostics,
                                    matrix_diagnostics, factorization_diagnostics);
      };

      bool failed = false;
      try {
        spatial_jacobian = 0;
        assembler.set_time(t);
        const auto request_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent variables_event{.topic = progress_topics::variables, .time = t, .minimum_verbosity = 2};
        request_diagnostics.append_to(variables_event);
        this->log.progress(variables_event);
        request_variables(variable_y, y, t);
        y_state.refresh(y);
        y_dot_state.refresh(y_dot);
        assembler.jacobian(spatial_jacobian, y_state.get(), 1., y_dot_state.get(), alpha, 1., variable_y);
        if (jacobian_diagnostics_enabled) matrix_diagnostics = analyze_jacobian_matrix(spatial_jacobian);
        linSolver.init(spatial_jacobian);

        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent jacobian_event{
            .topic = progress_topics::jacobian, .time = t, .duration_ms = calc_timer.lap(), .minimum_verbosity = 2};
        current_diagnostics.append_to(jacobian_event);
        this->log.progress(jacobian_event);

        factorize_with_diagnostics(linSolver, spatial_jacobian, factorization_diagnostics,
                                   jacobian_diagnostics_enabled);
        if (factorization_diagnostics.factorization_success == 1.) {
          const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
          ProgressEvent factorization_event{.topic = progress_topics::factorization,
                                            .time = t,
                                            .duration_ms = calc_timer.lap(),
                                            .minimum_verbosity = 3};
          current_diagnostics.append_to(factorization_event);
          this->log.progress(factorization_event);
        }
        record_diagnostics();
      } catch (std::exception &e) {
        if constexpr (decltype(linSolver)::performs_factorization)
          if (std::isnan(factorization_diagnostics.factorization_success))
            factorization_diagnostics.factorization_success = 0.;
        record_diagnostics();
        callback_diagnostics.jacobian_failures++;
        failed = true;
      }
      if (internal::agreed_ida_result(comm, failed) != 0) return ++failure_counter;

      failure_counter = 0;
      return 0;
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const VectorType &src, VectorType &dst, const double tol) -> int {
      CalcDtTimer calc_timer;
      bool failed = false;
      try {
        const auto sol_iterations = linSolver.solve(src, dst, tol);
        // A direct solver reports no iterations (-1), but its time counts all the same.
        const auto current_diagnostics = make_timestepping_diagnostics(time_stepper, callback_diagnostics);
        ProgressEvent solve_event{.topic = progress_topics::implicit_linear_solve,
                                  .time = stuck_t,
                                  .duration_ms = calc_timer.lap(),
                                  .iterations = sol_iterations,
                                  .minimum_verbosity = 2};
        current_diagnostics.append_to(solve_event);
        this->log.progress(solve_event);
      } catch (std::exception &) {
        callback_diagnostics.linear_solver_failures++;
        failed = true;
      }
      if (internal::agreed_ida_result(comm, failed) != 0) return ++failure_counter;
      return 0;
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(spatial_y, spatial_y_dot);
    } catch (const std::exception &e) {
      this->log.error("Timestepping failed: {}", e.what());
      this->finalize_output_after_failure([&]() { time_stepper.output_step(stuck_t, spatial_y, spatial_y_dot, 0); });
      throw;
    }

    initial_data.block(0) = spatial_y;
    commit_variables(variable_y, spatial_y, t_stop);
    VectorType variables_scratch;
    reinit_local_variables_vector(variables_scratch, n_vars);
    compute_variables_into(initial_data.block(1), variables_scratch, [&](VectorType &out) { out = variable_y; });
    this->drain_output();
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                            DiFfRG::UMFPack, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                            DiFfRG::UMFPack, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                            DiFfRG::UMFPack, 0>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                            DiFfRG::UMFPack, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                            DiFfRG::UMFPack, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                            DiFfRG::UMFPack, 1>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            1, DiFfRG::UMFPack, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            2, DiFfRG::UMFPack, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            3, DiFfRG::UMFPack, 0>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            1, DiFfRG::UMFPack, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            2, DiFfRG::UMFPack, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            3, DiFfRG::UMFPack, 1>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                            DiFfRG::GMRES, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                            DiFfRG::GMRES, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                            DiFfRG::GMRES, 0>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                            DiFfRG::GMRES, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                            DiFfRG::GMRES, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                            DiFfRG::GMRES, 1>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            1, DiFfRG::GMRES, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            2, DiFfRG::GMRES, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            3, DiFfRG::GMRES, 0>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            1, DiFfRG::GMRES, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            2, DiFfRG::GMRES, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            3, DiFfRG::GMRES, 1>;

// The default-solver spelling. DefaultLinearSolver is deliberately an indirect alias and so
// stays a distinct template argument from UMFPack on every compiler, which means
// TimeStepper<Assembler> needs its own instantiations alongside TimeStepper<Assembler, UMFPack>.
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                            DiFfRG::DefaultLinearSolver, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                            DiFfRG::DefaultLinearSolver, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                            DiFfRG::DefaultLinearSolver, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                            DiFfRG::DefaultLinearSolver, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                            DiFfRG::DefaultLinearSolver, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                            DiFfRG::DefaultLinearSolver, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            1, DiFfRG::DefaultLinearSolver, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            2, DiFfRG::DefaultLinearSolver, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            3, DiFfRG::DefaultLinearSolver, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            1, DiFfRG::DefaultLinearSolver, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            2, DiFfRG::DefaultLinearSolver, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>,
                                                            3, DiFfRG::DefaultLinearSolver, 1>;

// ##############################################################################
// Distributed (PETSc-backed) instantiations
// ##############################################################################
//
// An MPI build's default mesh is partitioned and its default linear algebra is PETSc-backed, so
// TimeStepper<Assembler> resolves here. IDA advances the distributed spatial block; the explicit
// variables are small and integrated redundantly on every rank (see evaluate_variable_rhs).
#ifdef DEAL_II_WITH_PETSC
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 1, DiFfRG::PETScKrylov, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 2, DiFfRG::PETScKrylov, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 3, DiFfRG::PETScKrylov, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 1, DiFfRG::PETScKrylov, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 2, DiFfRG::PETScKrylov, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 3, DiFfRG::PETScKrylov, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 1, DiFfRG::DefaultLinearSolver, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 2, DiFfRG::DefaultLinearSolver, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 3, DiFfRG::DefaultLinearSolver, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 1, DiFfRG::DefaultLinearSolver, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 2, DiFfRG::DefaultLinearSolver, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 3, DiFfRG::DefaultLinearSolver, 1>;
#ifdef DEAL_II_PETSC_WITH_MUMPS
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 1, DiFfRG::PETScDirect, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 2, DiFfRG::PETScDirect, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 3, DiFfRG::PETScDirect, 0>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 1, DiFfRG::PETScDirect, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 2, DiFfRG::PETScDirect, 1>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_BoostRK_impl<
    dealii::PETScWrappers::MPI::Vector, dealii::PETScWrappers::MPI::SparseMatrix, 3, DiFfRG::PETScDirect, 1>;
#endif
#endif
