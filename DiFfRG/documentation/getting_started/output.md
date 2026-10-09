# Runtime output

Runtime output uses one value-owned `OutputPath` inside an `OutputSession`. The session owns the scientific sinks,
bounded asynchronous VTK queue, effective-configuration snapshot, and run reporter. There is no global named-logger
registry and required timestepper dependencies are passed by reference.

Applications may adapt their JSON configuration at the boundary:

```cpp
OutputSession<Assembler> output(json);
```

The session is keyed on the assembler so that its spatial dimension and vector type cannot drift from the ones the
timestepper is built on. Anything exposing `dim` and `VectorType` works — pass the `Discretization` instead wherever
there is no single assembler type, e.g. a test that runs one discretization against several models.

If `/output/folder` is present, the run log, configuration record, HDF5 file, CSV tables, and field directory all use
that configured root. Without `/output/folder`, the session creates a unique system-temporary root and removes it at
destruction. A completely default session therefore needs no output configuration:

```cpp
OutputSession<Assembler> output;
```

Unit tests request an automatically cleaned system-temporary directory without constructing output JSON:

```cpp
OutputPath path = OutputPath::temporary();
OutputSession<Assembler> output(path, Config::OutputSettings{});
```

Use `OutputPath::temporary(TemporaryRetention::keep)` and pass it to the session while debugging a test. `OutputPath`
is a copyable immutable ownership handle: copies share temporary-root lifetime, and cleanup occurs after the last copy
or owning session is destroyed. Explicitly configured paths are persistent and are never deleted automatically.

Each scheduled output event is collected in one scoped `OutputFrame`:

```cpp
output.write_frame(time, [&](auto &frame) {
  assembler.attach_data_output(frame, solution, variables);
});
```

`frame.fields()` collects the primary field series. Additional field series can be kept separate by giving them a
name:

```cpp
output.write_frame(time, [&](auto &frame) {
  auto auxiliary = frame.fields("auxiliary");
  auxiliary.attach(dof_handler, auxiliary_solution, "auxiliary_u");
});
```

Named series follow the same `/output/vtk` and `/output/hdf5` settings as the primary fields. With VTK enabled, the
example writes `<run>_auxiliary.pvd` and its VTU frames. With HDF5 enabled, it writes
`/auxiliary/<six-digit frame>` inside the run's single `<run>.h5` file. A series name must be one safe path component;
`FE`, `potential`, `eom_potential`, `scalars`, `maps`, and `coordinates` are reserved by the standard file layout.

All assembler families retain the familiar `attach_data_output` seam. Model `readouts` receive the frame through it.
`readouts_multiple` assigns every readout a stable ID so duplicate contributions fail immediately. A frame error makes
the session fail-stop, preventing staged data from contaminating a later timestep. Existing CSV, HDF5, PVD, and VTU
names and schemas remain unchanged.

## The CSV format

`CsvOutput` writes one header row and then one row per output step, comma-separated, with numbers in
scientific notation. `DiFfRG/common/csv.hh` defines the same format for reading it back, and the
Julia (`julia/src/io/csv.jl`) and Python (`DiFfRG.file_io.csv`) helpers follow it too, so a file a
run wrote loads identically in all three languages:

- `#` comment lines and blank lines are skipped.
- A cell that is not a finite number reads back as `NaN` rather than raising, so a run that wrote an
  inf or left a gap still loads and the gap stays visible in the data.
- Rows whose field count does not match the header are skipped.
- Column names may be quoted, which is the only way a name can contain the separator. Names written
  by `CsvOutput` pass through `strip_name` and so never need it.
- A UTF-8 byte order mark, CRLF line endings and a missing final newline are all accepted.

Numbers are parsed with `std::from_chars`, so reading does not depend on the global locale.

Code which runs independently of scheduled frames receives a `DiagnosticPort` by value. For example, a timestepper can
export Jacobian eigenvalues without knowing a filename or owning a writer:

```cpp
class MyStepper {
public:
  explicit MyStepper(DiagnosticPort diagnostics) : diagnostics(std::move(diagnostics)) {}

  void inspect(double time, double lambda_min, double lambda_max) {
    diagnostics.record("jacobian_eigenvalues.csv", time,
                       {{"lambda_min", lambda_min}, {"lambda_max", lambda_max}});
  }

private:
  DiagnosticPort diagnostics;
};
```

The same injection works for a model pre-assembly hook: store a `DiagnosticPort` in the model and record the diffusion
coefficient when the hook computes it. The port is thread-safe and serializes a complete record atomically. It should
not be used for high-volume field output; add that data to an `OutputFrame` instead.

With `/timestepping/implicit/jacobian_diagnostics` set to `true` (it defaults to `false`), all implicit time steppers
write one record per Jacobian callback to `<run>_jacobian_diagnostics.csv`. Runs with a separate dense variable Jacobian
also write `<run>_variable_jacobian_diagnostics.csv`; both records use the same `jacobian_build_id`. The switch is off by
default because the records are not free: each one costs a full sweep over the assembled Jacobian, and a factorizing
linear solver additionally estimates the condition number, which takes several extra triangular solves per build. The numeric `stepper_kind` values are `0=IDA`, `1=IDA+Boost RK`, `2=IDA+Boost ABM`, `3=implicit
Euler`, and `4=TRBDF2`. The `stage` values are `0=main`, `1=TR`, and `2=BDF2`.

For implicit Euler and TRBDF2, `step` counts accepted steps and `retry_index` counts rejected attempts at the same step.
For IDA, `step` and `ida_step` both contain the native IDA step counter and `retry_index` is `NaN`; `ida_step` is `NaN`
for other steppers. `current_h` is the attempted step size and `last_h` is the previous accepted step size. `alpha` and
`beta` are the mass-Jacobian weights from the
assembler interface, while `residual_weight` is the residual-Jacobian weight. For implicit Euler these are
`alpha=beta=1` and `residual_weight=h`. For TRBDF2, the TR stage uses `alpha=beta=1` and
`residual_weight=gamma*h/2`; the BDF2 stage uses `alpha=beta=2-gamma` and
`residual_weight=(1-gamma)*h`, where `gamma=2-sqrt(2)`.

Each record also contains matrix dimensions, actual nonzero count, entry and matrix norms, diagonal and row/column
scale measures, and factorization diagnostics. `min_diagonal_dominance` is the minimum of
`abs(diagonal) / sum(abs(off-diagonal))` over all rows. A diagonal-only nonzero row has infinite dominance, while an
empty or zero row has zero dominance.

`factorization_ms`, `factorization_success`, and `scaled_rcond_estimate` are populated for UMFPACK-based solvers. The
condition estimate applies the same maximum-entry row/column equilibration used by `ScaledLinearSolver` and performs
at most five estimator iterations after numeric factorization. Dense variable Jacobians report inversion time and
success but leave the condition estimate as `NaN`. Iterative GMRES paths leave all three factorization fields as `NaN`;
no eigenvalue computation is performed.

JSON is only an adapter for the typed `Config::OutputSettings`. The corresponding JSON settings are:

```json
{
  "output": {
    "max_pending_frames": 2,
    "log_queue_size": 8192
  }
}
```

`OutputSession::report_port()` returns a cheap, copyable `ReportPort`. Assemblers, timesteppers, solvers, and models use
that port for `info`, `warn`, `error`, and structured `progress` records; they neither construct loggers nor inspect
output JSON. A default-constructed `ReportPort` also works immediately and lazily creates a console-only reporter on
its first message; it does not create a directory or file. No setup or named spdlog logger is required.
`ReportPort::log_file()` returns an optional path that is empty for console-only ports.

Spatial discretizations accept an optional port and assemblers inherit it from their discretization:

```cpp
OutputSession<Assembler> output(json);
Discretization discretization(mesh, json, output.report_port());
Assembler assembler(discretization, model, json);
TimeStepper timestepper(json, assembler, output);
```

Standalone discretizations may simply use `Discretization(mesh, json)`; their constructor diagnostics go to the
automatic console-only fallback. Required timestepper dependencies and initial conditions are references, so `run` is
called as `timestepper.run(initial_condition, start, stop)`.

Warnings and errors are flushed promptly, and callers can request an explicit flush. Progress is coalesced by topic before it reaches
the asynchronous sinks, so tight callback loops cannot fill the terminal or make logging dominate the calculation.

The progress policy is deliberately fixed in C++ and has no JSON controls:

- verbosity 0 suppresses progress;
- verbosity 1 reports residual/timestep work;
- verbosity 2 adds Jacobian, linear-solver, and solver-specific diagnostics;
- verbosity 3 adds factorization and output work;
- verbosity 4 remains aggregated;
- verbosity 5 emits every event and prints a performance warning.

Every progress line carries `wall=`, the total wall time elapsed since the reporter was constructed; the per-event
`avg=` column instead reports the mean cost of the operation being logged.

At levels 1--4, each topic produces at most one update per second in both the console and the run log.
Each progress record is at most 100 columns and uses at most two lines: a stable summary line followed, when needed, by
one solver-diagnostic line. The reporter writes the final pending aggregates during shutdown. These intervals and the
line width are intentionally not configurable.

Custom solvers can add numeric diagnostics without formatting their own console text:

```cpp
static constexpr ProgressTopic krylov_solve{"kry", "restarted"};
ProgressEvent event{.topic = krylov_solve,
                    .time = time,
                    .duration_ms = elapsed_ms,
                    .iterations = iterations,
                    .minimum_verbosity = 2};
event.field("restart", restart_count, 2).field("residual", final_residual, 2);
report.progress(event);
```

Built-in timesteppers require and call `set_report_port(ReportPort)` on their linear solver. Solvers derived from
`AbstractLinearSolver` receive the implementation automatically and may use its protected `report_port`; independently
composed solvers must provide the same explicit reporting contract.

Topic labels and field names must be string literals (or otherwise have static storage duration). Standard topics live
in `progress_topics`, while custom solvers can define their own `constexpr ProgressTopic` without changing the central
reporter. The reporter copies all numeric data, aggregates repeated events, and renders the stable `[step]`, `[res ]`,
`[jac ]`, `[fac ]`, `[lin ]`, `[vars]`, and `[out ]` tags. Assemblers return a structured `SummaryEvent` from
`summary()`; timesteppers submit it automatically while draining output at the end of a run.

A `QuadratureProvider` constructed from the configuration reports every quadrature it builds into the run log,
tagged `[quadrature]`, whenever `/output/folder` is configured. It owns that reporter instead of borrowing the
session's: integrators request their quadratures from within their constructors, usually before any `OutputSession`
exists. The log file sink is shared process-wide per path, so the session opening the same file later appends to the
quadrature inventory instead of truncating it. The inventory never echoes to the console, where it would interleave
with the timestepper progress report; without a configured folder the provider stays silent. Pass an explicit
`ReportPort` as the second constructor argument to route the messages into an existing run reporter instead.

Debugging and process-memory policy stay in C++ rather than simulation configuration. Set
`Config::OutputSettings::asynchronous` to `false` for synchronous field writes. The default pending-byte limit is 2
GiB and can be changed through `Config::OutputSettings::max_pending_bytes` when constructing the session.

## HDF5 writing

A spatial run stores all field series in one `<run>.h5` file. The primary fields use `/FE/<six-digit frame>`, the
common raw-potential reconstruction uses `/potential/<six-digit frame>`, and readout-specific EoM reconstructions use
`/eom_potential/<six-digit frame>`. A series created with `frame.fields(name)` uses `/<name>/<six-digit frame>`.
Separate `<run>_potential.h5`, `<run>_eom_potential.h5`, or named-series HDF5 files are not created. Existing scalar,
map, and coordinate data remain under `/scalars`, `/maps`, and `/coordinates`.

`scalar()`, `map()` and the finite-element field data do not touch the file when they are called. They validate their
arguments, copy their payload, and stage the write. `flush()` then commits the whole frame in a single
open → write → `H5Fflush` → close cycle. Two consequences are worth knowing:

- **A frame is on disk only after its flush.** Under an `OutputSession` this is automatic, once per `write_frame`.
  Code that drives an `HDF5Output` directly and then reopens the file to read it back must flush first.
- **Every frame is still closed on disk.** The file stays readable while the run continues, and an abort leaves a
  complete prefix of frames behind rather than a corrupt file.

One writer thread per session performs those cycles, so the integrator does not wait for the disk. It is one thread
and not one per sink because the HDF5 library is built here without thread safety, which means every HDF5 call in the
process has to be serialised. For the same reason, code that opens an HDF5 file itself in the middle of a run — an
`HDF5Input`, say — must call `output.drain()` first.

`Config::OutputSettings::hdf5_queue_depth` (default 1) bounds how many frames may be in flight. It is separate from
`max_pending_frames`, which bounds VTK/`DataOut` memory. The default of 1 is a durability choice: a hard abort
(`SIGKILL`, node eviction) can lose the queued frames on top of the one being written, so depth 1 caps that at two
frames while still hiding the whole write behind the next output interval. Setting
`Config::OutputSettings::asynchronous` to `false` removes the thread entirely and restores commit-on-return, at the
cost of putting the write back on the critical path.

Each run reports where its output time went, at `info` level, when the session finishes: the contributor callback and
the potential reconstructions inside it, `build_patches`, the data filter, the HDF5 write and open/close, queue
waiting, and the writer thread's own total, which is listed separately because it is off the critical path. Set
`/output/verbosity` to 3 or more to additionally get one line and one `output_timings.csv` row per frame.

`/output/verbosity` also decides how much of the run log reaches the console: at 0 only warnings and errors, at 1 the
ordinary `info` messages, from 2 upwards also `debug` records such as the per-frame timing line. The `.log` file is
never gated this way and always receives everything `/output/log_level` admits.

Only MPI rank zero creates sinks. A full queue blocks the producer rather than dropping scientific data. At the end of
each timestepper `run()`, the session drains pending writers and reports worker errors without closing, so the same
session can continue across multiple time intervals. Normal applications do not call `finish()` explicitly: the
`OutputSession` destructor performs terminal cleanup. Advanced callers may use `finish()` to close a session before it
leaves scope; subsequent frame writes are then rejected.
