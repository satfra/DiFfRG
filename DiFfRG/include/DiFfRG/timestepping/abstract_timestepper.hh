#pragma once

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/common/la_policy.hh>
#include <DiFfRG/discretization/common/solution_view.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/data/snapshot.hh>
#include <DiFfRG/discretization/mesh/no_adaptivity.hh>
#include <DiFfRG/timestepping/jacobian_diagnostics.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// clang-format off
/**
 * @file abstract_timestepper.hh
 * @brief Common base of all timesteppers, the parameters they read, and how to choose one.
 *
 * ### What is solved
 * The discretized flow consists of the FE/FV unknowns \f$u\f$ (one value per degree of freedom) and, optionally,
 * a few extra scalar *variables* \f$v\f$ (Model::dt_variables). The assembler provides
 * - the mass term \f$m(u, \dot u)\f$ (Model::mass), which for an ordinary flow is \f$M\dot u\f$ with the mass matrix
 *   \f$M\f$,
 * - the spatial residual \f$R(u, v)\f$, i.e. the flux and source terms of the model integrated against the test
 *   functions,
 * - the right-hand side of the variables, \f$D(v, u)\f$ = Model::dt_variables. Note the residual convention:
 *   `dt_variables` returns \f$-\partial_t v\f$.
 *
 * All timesteppers evolve the RG time \f$t\f$ (with \f$k = \Lambda e^{-t}\f$) according to
 * \f[
 *   m(u, \dot u) + R(u, v) = 0\,, \qquad \dot v + D(v, u) = 0\,.
 * \f]
 * Implicit steppers solve this in residual form, \f$F(t, y, \dot y) = 0\f$ with \f$y = (u, v)\f$. Explicit steppers
 * use \f$\dot u = -M^{-1} R(u, v)\f$, which requires a mass term linear in \f$\dot u\f$.
 *
 * ### Choosing a timestepper
 * All of them take the Assembler as template argument, e.g. `TimeStepperSUNDIALS_IDA<Assembler>`, and are run through
 * AbstractTimestepper::run.
 *
 * | Timestepper                          | Method                                         | Integrates             | Step size        | Mesh adaptor | MPI |
 * |--------------------------------------|------------------------------------------------|------------------------|------------------|--------------|-----|
 * | TimeStepperSUNDIALS_IDA              | implicit BDF, order 1 to 5 (SUNDIALS IDA)      | \f$u\f$, \f$u+v\f$, \f$v\f$ | adaptive         | yes          | yes |
 * | TimeStepperSUNDIALS_IDA_BoostABM     | IDA for \f$u\f$, explicit Adams method for \f$v\f$ | \f$u+v\f$               | adaptive         | yes          | yes |
 * | TimeStepperSUNDIALS_IDA_BoostRK54/78 | IDA for \f$u\f$, explicit Runge-Kutta for \f$v\f$  | \f$u+v\f$               | adaptive         | yes          | yes |
 * | TimeStepperBoostRK54/78              | explicit Runge-Kutta, 5th / 8th order          | \f$u\f$, \f$u+v\f$, \f$v\f$ | adaptive         | no           | no  |
 * | TimeStepperBoostABM                  | explicit 8-step Adams-Bashforth-Moulton        | \f$u\f$, \f$u+v\f$, \f$v\f$ | adaptive or fixed | no          | no  |
 * | TimeStepperRK                        | explicit Runge-Kutta (deal.II), see set_method | \f$v\f$ only            | fixed            | no           | no  |
 * | TimeStepperExplicitEuler             | forward Euler                                  | \f$u\f$ only            | fixed            | yes          | no  |
 * | TimeStepperImplicitEuler             | backward Euler                                 | \f$u\f$ only            | adaptive         | yes          | no  |
 * | TimeStepperTRBDF2                    | TR-BDF2 (trapezoidal + BDF2 stage)             | \f$u\f$ only            | adaptive         | yes          | no  |
 *
 * Steppers that integrate only \f$u\f$ keep the variables \f$v\f$ fixed at their initial values. Steppers without mesh adaptor
 * support accept one in the constructor but never call it. IDA is the default choice: it handles stiff flows,
 * algebraic constraints (\f$m\f$ independent of \f$\dot u\f$ for some components) and distributed runs.
 *
 * ### Parameters
 * Read from the ConfigTree on construction. A stepper reads the `implicit` section if it is implicit, the `explicit`
 * section if it is explicit, and both if it is a hybrid. Keys marked "warns" print a warning if missing.
 *
 * | Key                                              | Type   | Default | Used by |
 * |--------------------------------------------------|--------|---------|---------|
 * | `/timestepping/output_dt`                        | double | 0.1     | all: output every `output_dt` in \f$t\f$ |
 * | `/timestepping/implicit/dt`                      | double | 1e-4    | initial step of the implicit steppers |
 * | `/timestepping/implicit/minimal_dt`              | double | 1e-8    | ImplicitEuler, TRBDF2: smallest step; below it Newton failures are accepted. Currently no effect on the IDA step size (IDA: only the time resolution of its stuck check). |
 * | `/timestepping/implicit/maximal_dt`              | double | 1       | ImplicitEuler, TRBDF2: largest step. Currently no effect on IDA. |
 * | `/timestepping/implicit/abs_tol`                 | double | 1e-13 (warns) | IDA: local error test; ImplicitEuler, TRBDF2: Newton convergence |
 * | `/timestepping/implicit/rel_tol`                 | double | 1e-7 (warns)  | as abs_tol |
 * | `/timestepping/implicit/max_steps`               | uint   | 1e6     | currently no effect |
 * | `/timestepping/implicit/max_non_linear_iterations` | uint | 10      | IDA steppers: Newton iterations per step |
 * | `/timestepping/implicit/local_tolerance_refresh` | double | 4       | IDA without variables and the IDA hybrids, for models with per-dof tolerances; see LocalAbsTolerances |
 * | `/timestepping/implicit/iterative_refinement`    | bool   | true    | UMFPack: refine every direct solve |
 * | `/timestepping/implicit/jacobian_diagnostics`    | bool   | false   | implicit steppers: write `<run>_jacobian_diagnostics.csv` |
 * | `/timestepping/implicit/ida_callback_trace`      | bool   | false   | TimeStepperSUNDIALS_IDA: print every failed callback, at `/output/verbosity` >= 5 |
 * | `/timestepping/implicit/ida_callback_trace_min_t` | double | 0      | only trace from this \f$t\f$ on |
 * | `/timestepping/implicit/ida_callback_trace_max_lines` | uint | 200  | cap on traced successful calls |
 * | `/timestepping/implicit/ida_callback_trace_successes` | bool | false | also trace successful calls |
 * | `/timestepping/implicit/ida_error_dof_diagnostics` | bool | false    | TimeStepperSUNDIALS_IDA (without variables, or variables only): on rejected steps, report the dofs with the largest local error |
 * | `/timestepping/implicit/ida_error_dof_diagnostics_top_n` | uint | 8 | number of dofs reported |
 * | `/timestepping/explicit/dt`                      | double | 1e-2 (warns) | fixed-step steppers: the step; adaptive ones: the initial (BoostRK) or largest (ABM) step |
 * | `/timestepping/explicit/minimal_dt`              | double | 1e-16   | ABM error control: smallest step; also the time resolution of the stuck check (BoostRK, BoostABM) |
 * | `/timestepping/explicit/maximal_dt`              | double | 1e16    | BoostRK (variables only) and the IDA hybrids: largest explicit step |
 * | `/timestepping/explicit/abs_tol`                 | double | 1e-3 (warns) | BoostRK, BoostABM, IDA hybrids: local error test of the explicit steps |
 * | `/timestepping/explicit/rel_tol`                 | double | 1e-3 (warns) | as abs_tol |
 * | `/timestepping/explicit/detect_stuck`            | bool   | true    | BoostRK: abort when the step size collapses |
 * | `/timestepping/explicit/error_control`           | bool   | true    | BoostABM, IDA_BoostABM: adapt the step to abs_tol/rel_tol; false: fixed step `explicit/dt` |
 * | `/timestepping/explicit/coupling_mode`           | int    | 1       | IDA_BoostABM, see TimeStepperSUNDIALS_IDA_BoostABM_impl |
 *
 * The explicit error test of BoostRK and the ABM steppers requires, per component \f$i\f$,
 * \f$|e_i| \le \mathrm{abs\_tol} + \mathrm{rel\_tol}\,|y_i|\f$ for the estimated local error \f$e_i\f$ (BoostRK adds
 * \f$\Delta t\,|\dot y_i|\f$ to \f$|y_i|\f$). IDA uses the weighted RMS norm of SUNDIALS with the same two
 * tolerances; dofs whose equation has no time derivative are excluded from its error test.
 *
 * Snapshots and restarts (optional; see AbstractTimestepper::run and documentation/getting_started/snapshots.md):
 * - `/timestepping/snapshots/k`, `/timestepping/snapshots/t`: lists of RG scales (needs `/physical/Lambda`) or RG
 *   times at which to write a snapshot.
 * - `/timestepping/snapshots/snap_to_output_grid` (default true): move snapshot times onto the output grid.
 * - `/timestepping/snapshots/stop_after_last` (default false): end the run after its last snapshot.
 * - `/restart/file`: a snapshot to continue the flow from (set by the `--restart` command line flag).
 *
 * `/output/verbosity` controls the progress output: 0 none, 1 residual work, 2 adds Jacobian, linear solver and
 * solver statistics, 3 adds factorization and output work; up to 4 the reports are aggregated, 5 prints every event.
 */
// clang-format on

namespace DiFfRG
{
  /// One dof in an IDAErrorDofDiagnostics report.
  struct IDAErrorDofRecord {
    types::global_dof_index dof = dealii::numbers::invalid_dof_index;
    double value = 0.;
    double estimated_local_error = 0.;
    double error_weight = 0.;
    double contribution = 0.;
  };

  /**
   * @brief Stopwatch feeding structured progress durations.
   *
   * `lap()` returns the milliseconds elapsed since construction or the previous
   * `lap()` and re-arms, so consecutive operations inside one callback
   * (residual, jacobian construction, jacobian inversion, linear solve) each
   * report their own duration rather than a cumulative total.
   */
  struct CalcDtTimer {
    std::chrono::high_resolution_clock::time_point mark = std::chrono::high_resolution_clock::now();

    double lap()
    {
      const auto now = std::chrono::high_resolution_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(now - mark).count();
      mark = now;
      return ms;
    }
  };

  /// Report of /timestepping/implicit/ida_error_dof_diagnostics after IDA rejected steps; see
  /// TimeStepperSUNDIALS_IDA_impl::set_ida_error_dof_callback.
  struct IDAErrorDofDiagnostics {
    double t = 0.;
    double k = std::numeric_limits<double>::quiet_NaN();
    long int reject_delta = 0;
    long int total_rejects = 0;
    long int ida_steps = 0;
    double ida_last_step_size = 0.;
    double ida_current_step_size = 0.;
    double ida_current_time = 0.;
    double wrms = 0.;
    std::vector<IDAErrorDofRecord> top_dofs;
  };

  /// IDA's step statistics, attached to progress reports.
  struct IDAProgressDiagnostics {
    long int ida_steps = 0;
    long int ida_error_test_failures = 0;
    long int ida_nonlinear_convergence_failures = 0;
    long int ida_step_solve_failures = 0;
    long int ida_residual_evaluations = 0;
    long int ida_nonlinear_iterations = 0;
    double ida_last_step_size = 0.;
    double ida_current_step_size = 0.;
    double ida_current_time = 0.;
    void append_to(ProgressEvent &event) const
    {
      event.field("h", ida_current_step_size, 2)
          .field("last_h", ida_last_step_size, 2)
          .field("steps", ida_steps, 2)
          .field("rejects", ida_error_test_failures, 2)
          .field("nl_fail", ida_nonlinear_convergence_failures, 2);
    }
  };

  /// Counts of failed residual, Jacobian and linear-solver callbacks, attached to progress reports.
  struct SolverCallbackDiagnostics {
    size_t nonfinite_solution_failures = 0;
    size_t nonfinite_residual_failures = 0;
    size_t residual_exceptions = 0;
    size_t jacobian_failures = 0;
    size_t linear_solver_failures = 0;

    size_t nonfinite_failures() const { return nonfinite_solution_failures + nonfinite_residual_failures; }
    bool has_failures() const
    {
      return nonfinite_failures() > 0 || residual_exceptions > 0 || jacobian_failures > 0 || linear_solver_failures > 0;
    }
    void append_to(ProgressEvent &event) const
    {
      if (!has_failures()) return;
      event.field("nan", nonfinite_failures(), 2)
          .field("res_exc", residual_exceptions, 2)
          .field("jac_fail", jacobian_failures, 2)
          .field("lin_fail", linear_solver_failures, 2);
    }
  };

  /// Everything a timestepper attaches to a progress report.
  struct TimesteppingDiagnostics {
    std::optional<IDAProgressDiagnostics> ida;
    SolverCallbackDiagnostics callbacks;

    void append_to(ProgressEvent &event) const
    {
      if (ida) ida->append_to(event);
      callbacks.append_to(event);
    }
    bool empty() const { return !ida && !callbacks.has_failures(); }
  };

  /**
   * @brief Base class of all timesteppers.
   *
   * Reads the /timestepping parameters (see abstract_timestepper.hh for the list), and implements run(), which
   * handles snapshots and restarts around the stepper's run_segment().
   *
   * @tparam VectorType_ vector type of the assembler: dealii::Vector<double>, or a PETSc MPI vector in an MPI build.
   * @tparam SparseMatrixType_ sparse matrix type of the assembler.
   * @tparam dim_ spatial dimension; 0 for a pure-variables system.
   */
  template <typename VectorType_, typename SparseMatrixType_, uint dim_> class AbstractTimestepper
  {
  protected:
    static constexpr uint dim = dim_;
    using VectorType = VectorType_;
    using NumberType = typename get_type::NumberType<VectorType>;
    using SparseMatrixType = SparseMatrixType_;
    // NOTE: InverseSparseMatrixType is deliberately NOT declared here. A member alias of a
    // class template is instantiated with the class, so declaring it in the common base forces
    // every timestepper to have a mass-matrix inverse type -- including the implicit ones,
    // which never invert a mass matrix. That made the whole hierarchy unusable with any matrix
    // type lacking a SparseDirectUMFPACK-shaped inverse (i.e. every distributed matrix). It now
    // lives only in the explicit steppers that actually use it.
    static_assert(SupportedVectorType<VectorType>,
                  "VectorType is not a vector type DiFfRG supports; see get_type in common/types.hh.");
    using BlockVectorType = typename get_type::BlockVectorType<VectorType>;

  public:
    /**
     * @brief Construct without mesh adaptation.
     *
     * @param config parameters; see abstract_timestepper.hh.
     * @param assembler assembles residuals and Jacobians.
     * @param data_out where the output frames go.
     * @param implicit_stepper whether to read /timestepping/implicit.
     * @param explicit_stepper whether to read /timestepping/explicit.
     */
    AbstractTimestepper(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                        OutputSession_impl<dim, VectorType> &data_out, const bool implicit_stepper,
                        const bool explicit_stepper)
        : config(config), adaptor_default(), assembler(assembler), data_out(data_out), adaptor(adaptor_default),
          log(data_out.report_port()), m_is_implicit(implicit_stepper), m_is_explicit(explicit_stepper)
    {
      read_parameters();
    }

    /**
     * @brief Construct with a mesh adaptor, which is called between steps (where the stepper supports it, see
     * abstract_timestepper.hh).
     */
    AbstractTimestepper(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                        OutputSession_impl<dim, VectorType> &data_out, AbstractAdaptor<VectorType> &adaptor,
                        const bool implicit_stepper, const bool explicit_stepper)
        : config(config), adaptor_default(), assembler(assembler), data_out(data_out), adaptor(adaptor),
          log(data_out.report_port()), m_is_implicit(implicit_stepper), m_is_explicit(explicit_stepper)
    {
      read_parameters();
    }

  private:
    void read_parameters()
    {
      output_dt = config.get_double("/timestepping/output_dt", 1e-1);

      if (m_is_implicit) {
        // Stuff you should really set
        impl.abs_tol = config.get_double_or_warn("/timestepping/implicit/abs_tol", 1e-13);
        impl.rel_tol = config.get_double_or_warn("/timestepping/implicit/rel_tol", 1e-7);
        impl.local_tolerance_refresh = config.get_double("/timestepping/implicit/local_tolerance_refresh", 4.);
        impl.iterative_refinement = config.get_bool("/timestepping/implicit/iterative_refinement", true);

        // Stuff you can set, but defaults are reasonable
        impl.dt = config.get_double("/timestepping/implicit/dt", 1e-4);
        impl.minimal_dt = config.get_double("/timestepping/implicit/minimal_dt", 1e-8);
        impl.maximal_dt = config.get_double("/timestepping/implicit/maximal_dt", 1.);
        impl.max_steps = config.get_uint("/timestepping/implicit/max_steps", 1e6);
        impl.max_non_linear_iterations = config.get_uint("/timestepping/implicit/max_non_linear_iterations", 10);
        impl.jacobian_diagnostics = config.get_bool("/timestepping/implicit/jacobian_diagnostics", false);
        impl.ida_callback_trace = config.get_bool("/timestepping/implicit/ida_callback_trace", false);
        impl.ida_callback_trace_min_t = config.get_double("/timestepping/implicit/ida_callback_trace_min_t", 0.0);
        impl.ida_callback_trace_max_lines = config.get_uint("/timestepping/implicit/ida_callback_trace_max_lines", 200);
        impl.ida_callback_trace_successes =
            config.get_bool("/timestepping/implicit/ida_callback_trace_successes", false);
        impl.ida_error_dof_diagnostics = config.get_bool("/timestepping/implicit/ida_error_dof_diagnostics", false);
        impl.ida_error_dof_diagnostics_top_n =
            config.get_uint("/timestepping/implicit/ida_error_dof_diagnostics_top_n", 8);

        // Sanity checks:
        if (impl.minimal_dt <= 0.0) throw std::invalid_argument("Minimal timestep size must be positive.");
        if (impl.maximal_dt <= 0.0) throw std::invalid_argument("Maximal timestep size must be positive.");
        if (impl.minimal_dt > impl.maximal_dt)
          throw std::invalid_argument("Minimal timestep size must be smaller than maximal timestep size.");
        if (impl.dt < impl.minimal_dt || impl.dt > impl.maximal_dt)
          throw std::invalid_argument("Initial timestep size must be within the minimal and maximal timestep size.");
        if (impl.abs_tol <= 0.0) throw std::invalid_argument("Absolute tolerance must be > 0.");
        if (impl.rel_tol <= 0.0) throw std::invalid_argument("Relative tolerance must be > 0.");
      }

      if (m_is_explicit) {
        expl.dt = config.get_double_or_warn("/timestepping/explicit/dt", 1e-2);
        expl.minimal_dt = config.get_double("/timestepping/explicit/minimal_dt", 1e-16);
        expl.maximal_dt = config.get_double("/timestepping/explicit/maximal_dt", 1e16);
        expl.abs_tol = config.get_double_or_warn("/timestepping/explicit/abs_tol", 1e-3);
        expl.rel_tol = config.get_double_or_warn("/timestepping/explicit/rel_tol", 1e-3);
        expl.detect_stuck = config.get_bool("/timestepping/explicit/detect_stuck", true);
        expl.error_control = config.get_bool("/timestepping/explicit/error_control", true);

        // Sanity checks:
        if (expl.minimal_dt <= 0.0) throw std::invalid_argument("Minimal timestep size must be positive.");
        if (expl.maximal_dt <= 0.0) throw std::invalid_argument("Maximal timestep size must be positive.");
        if (expl.minimal_dt > expl.maximal_dt)
          throw std::invalid_argument("Minimal timestep size must be smaller than maximal timestep size.");
        if (expl.dt < expl.minimal_dt || expl.dt > expl.maximal_dt)
          throw std::invalid_argument("Initial timestep size must be within the minimal and maximal timestep size.");
        if (expl.abs_tol <= 0.0) throw std::invalid_argument("Absolute tolerance must be > 0.");
        if (expl.rel_tol <= 0.0) throw std::invalid_argument("Relative tolerance must be > 0.");
      }
    }

  protected:
    void drain_output()
    {
      log.summary(assembler.summary());
      // Only rank 0 writes, but every rank may read the run's files back once run() returns (e.g.
      // through HDF5Input). Without this barrier the other ranks race ahead of rank 0's last frame:
      // they read stale data, or fail to open a file the writer still holds locked. The barrier is
      // reached even if the drain throws, so a write error on rank 0 cannot hang the other ranks.
      std::exception_ptr drain_error;
      try {
        data_out.drain();
      } catch (...) {
        drain_error = std::current_exception();
      }
      MPI::barrier(MPI_COMM_WORLD);
      if (drain_error) std::rethrow_exception(drain_error);
    }

    /**
     * @brief Best-effort output cleanup while a timestepping failure is being reported.
     *
     * Writes one final frame for the state that failed, then pushes everything pending to
     * disk. Both steps are swallowed on error: a readout that cannot cope with a diverged
     * solution must not replace the failure the caller is actually trying to report, and a
     * drain error is latched anyway and resurfaces at finish().
     *
     * Callers rethrow the original exception afterwards.
     */
    template <typename EmitFinalFrame> void finalize_output_after_failure(EmitFinalFrame &&emit_final_frame)
    {
      try {
        std::forward<EmitFinalFrame>(emit_final_frame)();
      } catch (const std::exception &e) {
        log.error("Output of the final frame after the failure failed: {}", e.what());
      } catch (...) {
        log.error("Output of the final frame after the failure failed.");
      }
      try {
        drain_output();
      } catch (const std::exception &e) {
        log.error("Draining pending output after the failure failed: {}", e.what());
      } catch (...) {
        log.error("Draining pending output after the failure failed.");
      }
    }

  public:
    /**
     * @brief Evolve @p initial_condition from @p t_start to @p t_stop.
     *
     * The time stepping itself is done by the derived class in run_segment(). Around it, run()
     * provides two optional features, both driven by the configuration:
     *
     * - **Snapshots.** At every time of /timestepping/snapshots (see SnapshotSchedule) the flow is
     *   split into segments, and the state at each boundary is written to
     *   `<run name>_snapshot_<nnn>.h5` next to the run's other output. The stepper is restarted at
     *   each snapshot, which costs one solver restart per snapshot.
     * - **Restart.** If /restart/file names a snapshot, the first call to run() restores the mesh,
     *   the state, the model's history-dependent state and the adaptation schedule from it, logs
     *   every configuration entry that differs from the snapshot's, and starts at the snapshot's
     *   time; @p t_start is then ignored. @p initial_condition must still have been set up as usual
     *   (interpolate()), so that its block structure exists.
     *
     * Snapshots are never overwritten: run() throws before stepping if one it would write exists. A
     * restart from one of this run's own snapshots continues the run, numbering its snapshots on from
     * there; finding a later one already on disk then means the run has continued before.
     *
     * Without either, this is exactly one call to run_segment().
     *
     * @param initial_condition The state; on return it holds the state at the end of the run.
     * @param t_start The start time of the simulation.
     * @param t_stop The run method will evolve the system from t_start to t_stop.
     */
    void run(AbstractFlowingVariables<NumberType, VectorType> &initial_condition, double t_start, const double t_stop)
    {
      const SnapshotSchedule schedule(config, log.Lambda(), output_dt);
      if (!restart_consumed && config.contains("/restart/file") && !config.get_string("/restart/file").empty()) {
        restart_consumed = true;
        t_start = restore_snapshot(initial_condition, t_start);
        if (t_stop < t_start)
          throw std::runtime_error("Restart: the snapshot is at t = " + std::to_string(t_start) +
                                   ", which is past the requested final time t = " + std::to_string(t_stop) + ".");
      }

      const auto snapshot_times = schedule.times_in(t_start, t_stop);
      // Checked now rather than when the first snapshot is due, which may be most of a flow away.
      for (std::size_t i = 0; i < snapshot_times.size(); ++i) {
        const auto path = snapshot_file(data_out.path(), snapshot_count + i);
        if (!std::filesystem::exists(path)) continue;
        if (continues_own_run)
          throw std::runtime_error("This run already continued past the snapshot it restarts from: '" + path.string() +
                                   "' exists. Restart from the latest snapshot instead.");
        throw std::runtime_error("The snapshot '" + path.string() +
                                 "' already exists; snapshots are never overwritten. Choose another /output/name or "
                                 "/output/folder, or remove the old snapshots.");
      }

      // A failed flow (an exception from the model, a non-finite residual, a stuck step) leaves the
      // stepper without draining, but callers routinely read the run's files right after catching
      // it -- e.g. to find where the flow diverged. Drain here, once for every stepper: this pushes
      // rank 0's pending frames to disk (its async writer must not race a reader in the same
      // process; HDF5 is not thread safe in this build) and holds the other ranks until it has.
      // No extra frame is written: the steppers output every accepted step already, and the state
      // at the failure is often non-finite. The barrier assumes the failure is collective, which
      // flow-driven failures are, since every rank computes a bitwise identical flow.
      try {
        double t = t_start;
        for (const double t_snapshot : snapshot_times) {
          run_segment(initial_condition, t, t_snapshot);
          write_snapshot(initial_condition, t_snapshot);
          t = t_snapshot;
        }
        if (!snapshot_times.empty() && schedule.stop_after_last()) {
          if (!is_close(t, t_stop, 1e-12))
            log.info("Stopping at t = {} after the last snapshot (/timestepping/snapshots/stop_after_last).", t);
          return;
        }
        if (snapshot_times.empty() || !is_close(t, t_stop, 1e-12)) run_segment(initial_condition, t, t_stop);
      } catch (...) {
        finalize_output_after_failure([] {});
        throw;
      }
    }

    /// Whether this stepper reads /timestepping/implicit.
    bool is_implicit() const { return m_is_implicit; }
    /// Whether this stepper reads /timestepping/explicit.
    bool is_explicit() const { return m_is_explicit; }

  protected:
    /**
     * @brief Any derived class must implement this method to run the timestepping algorithm.
     *
     * It must return the state at @p t_stop in @p initial_condition, because run() writes snapshots
     * from it. Called once per segment, i.e. possibly several times per
     * run() with consecutive intervals.
     *
     * @param initial_condition The flowing variables object that contains the initial condition.
     * @param t_start The start time of the segment.
     * @param t_stop The end time of the segment.
     */
    virtual void run_segment(AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start,
                             const double t_stop) = 0;

  private:
    /**
     * @brief Write the state at time @p t to the next snapshot file.
     *
     * Collective: the replicas are gathered on every rank, then the writer rank alone assembles
     * and writes the file, and the ranks agree on whether that worked.
     */
    void write_snapshot(const AbstractFlowingVariables<NumberType, VectorType> &state, const double t)
    {
      const auto &data = state.data();
      const bool has_variables = data.n_blocks() > 1;

      // Gathered here, outside the writer-rank region: a refresh communicates.
      SolutionView<VectorType> spatial_view, variables_view;
      assembler.reinit_solution_view(spatial_view);
      spatial_view.refresh(state.spatial_data());
      if (has_variables) {
        reinit_variables_view(variables_view, data.block(1).size(), assembler.get_communicator());
        variables_view.refresh(data.block(1));
      }

      const auto path = snapshot_file(data_out.path(), snapshot_count);
      ++snapshot_count;

      bool failed = false;
      std::string failure;
      if (data_out.is_writer_rank()) {
        try {
          SnapshotData snapshot;
          snapshot.t = t;
          snapshot.Lambda = log.Lambda();
          if (snapshot.Lambda > 0.) snapshot.k = snapshot.Lambda * std::exp(-t);
          snapshot.dim = dim;
          snapshot.spatial = assembler.capture_snapshot_state(spatial_view.get());
          if (has_variables) {
            const auto &variables = variables_view.get();
            snapshot.variables.resize(variables.size());
            for (std::size_t i = 0; i < snapshot.variables.size(); ++i)
              snapshot.variables[i] = variables(i);
          }
          assembler.save_model_state(snapshot.model);
          snapshot.last_adaptation_time = adaptor.last_adaptation_time();
          snapshot.config_json = json::serialize(static_cast<json::value>(config));
          DiFfRG::write_snapshot(path, snapshot);
          log.info("Wrote the snapshot at t = {} (k = {}) to '{}'.", t, snapshot.k, path.string());
        } catch (const std::exception &e) {
          failed = true;
          failure = e.what();
        }
      }
      if (DiFfRG::MPI::any_of(MPI_COMM_WORLD, failed))
        throw std::runtime_error("Writing the snapshot at t = " + std::to_string(t) + " to '" + path.string() +
                                 "' failed" + (failure.empty() ? std::string(" on the writer rank.") : ": " + failure));
    }

    /**
     * @brief Restore @p state from /restart/file. @return the snapshot's time.
     *
     * Collective: every rank reads the file and restores the part of the state it owns.
     */
    double restore_snapshot(AbstractFlowingVariables<NumberType, VectorType> &state, const double requested_t_start)
    {
      const std::filesystem::path path = config.get_string("/restart/file");
      const SnapshotData snapshot = read_snapshot(path);
      if (snapshot.dim != int(dim))
        throw std::runtime_error("Restart: the snapshot '" + path.string() +
                                 "' was written by a flow in dim = " + std::to_string(snapshot.dim) +
                                 ", but this flow has dim = " + std::to_string(dim) + ".");

      log.info("Restarting from the snapshot '{}' at t = {} (k = {}).", path.string(), snapshot.t, snapshot.k);
      // Continuing this very run: its later snapshots keep the numbers they would have had.
      if (const auto index = own_snapshot_index(data_out.path(), path)) {
        continues_own_run = true;
        snapshot_count = *index + 1;
      }
      if (!is_close(requested_t_start, snapshot.t, 1e-12))
        log.info("The requested start time t = {} is replaced by the snapshot's time.", requested_t_start);

      // /output always differs between runs and /restart, /timestepping/snapshots are about the
      // runs themselves, not the flow; listing them would only bury the entries that matter.
      const auto differences = config_diff(json::parse(snapshot.config_json), static_cast<json::value>(config),
                                           {"/output", "/restart", "/timestepping/snapshots"});
      if (differences.empty())
        log.info("The configuration matches the snapshot's.");
      else {
        log.warn("The configuration differs from the snapshot's in {} entries. The flow above the snapshot's scale "
                 "was computed with the snapshot's values, so a restart is only meaningful if these changes do not "
                 "affect it:",
                 differences.size());
        for (const auto &difference : differences)
          log.warn("    {}", difference);
      }

      auto &data = state.data();
      if (data.n_blocks() == 0)
        throw std::runtime_error("Restart: the initial condition has no block structure. Set it up as usual, e.g. "
                                 "with interpolate(model), before calling run().");

      // Before the spatial restore: a mesh rebuild reinitializes the assembler, which asks the model for its
      // constraints, so the model must already be at the snapshot's time and state.
      assembler.set_time(snapshot.t);
      if (!snapshot.model.empty() && !assembler.load_model_state(snapshot.model))
        log.warn("The snapshot carries model state ({} entries), but the model does not implement load_state(); the "
                 "state is ignored.",
                 snapshot.model.data().size());

      assembler.restore_snapshot_state(snapshot.spatial, data.block(0));

      if (data.n_blocks() > 1) {
        auto &variables = data.block(1);
        if (variables.size() != snapshot.variables.size())
          throw std::runtime_error("Restart: the snapshot holds " + std::to_string(snapshot.variables.size()) +
                                   " variables, but this flow has " + std::to_string(variables.size()) + ".");
        for (const auto i : variables.locally_owned_elements())
          variables(i) = snapshot.variables[i];
        variables.compress(dealii::VectorOperation::insert);
      } else if (!snapshot.variables.empty())
        throw std::runtime_error("Restart: the snapshot holds " + std::to_string(snapshot.variables.size()) +
                                 " variables, but this flow has none.");
      data.collect_sizes();

      if (std::isfinite(snapshot.last_adaptation_time)) adaptor.set_last_adaptation_time(snapshot.last_adaptation_time);
      return snapshot.t;
    }

    bool restart_consumed = false;
    bool continues_own_run = false;
    unsigned int snapshot_count = 0;

  protected:
    const ConfigTree config;
    NoAdaptivity<VectorType> adaptor_default;
    AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler;
    OutputSession_impl<dim, VectorType> &data_out;
    AbstractAdaptor<VectorType> &adaptor;
    ReportPort log;

    const bool m_is_implicit;
    const bool m_is_explicit;

    double output_dt;
    struct ImplicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
      double local_tolerance_refresh;
      /** UMFPACK's iterative refinement of every direct solve (default true). Off halves the cost of a solve;
       * inside IDA the Newton iteration corrects the solve's error anyway (2D O(2) KT flow, 80^2: 16.3 -> 12.8 s),
       * but a badly scaled system then only gets ~1e-7 relative accuracy per solve. */
      bool iterative_refinement;
      uint max_steps;
      uint max_non_linear_iterations;
      /** Enables the `<run>_jacobian_diagnostics.csv` tables. Off by default because the records
       * are not free: each one costs a full sweep over the assembled Jacobian and, for factorizing
       * solvers, a condition estimate worth several extra triangular solves. */
      bool jacobian_diagnostics;
      bool ida_callback_trace;
      double ida_callback_trace_min_t;
      uint ida_callback_trace_max_lines;
      bool ida_callback_trace_successes;
      bool ida_error_dof_diagnostics;
      uint ida_error_dof_diagnostics_top_n;
    } impl;

    struct ExplicitParameters {
      double dt;
      double minimal_dt;
      double maximal_dt;
      double abs_tol;
      double rel_tol;
      bool detect_stuck;
      bool error_control;
    } expl;

    std::size_t next_jacobian_build_id = 0;
    TimestepperJacobianDiagnosticsState jacobian_diagnostics_state;
  };
} // namespace DiFfRG
