# Output

Everything a run writes goes through one `OutputSession`, constructed in `main` from the parameter file:

```cpp
OutputSession<Assembler> data_out(json);
const auto log = data_out.report_port();     // the run log, for messages
TimeStepper time_stepper(json, assembler, data_out);
```

The timestepper writes one *frame* every `/timestepping/output_dt` in RG time, and the session takes care of the
files.

## Files of a run

All files go to `/output/folder` (default `./`) and start with `/output/name` (default `output`), written `<run>`
below.

| file | content |
| --- | --- |
| `<run>.log` | The run log: progress, messages, final performance summaries. |
| `<run>.h5` | All data in one HDF5 file (if `/output/hdf5`, default on); layout below. |
| `<run>.pvd` + `<run>/*.vtu` | The FE solution for ParaView (if `/output/vtk`, default on). The VTU frames go to the subfolder `/output/field_directory` (default: the run name). |
| `<run>_potential.pvd`, `<run>_eom_potential.pvd` | The potential reconstructed from the FE functions, see [Models](models.md#the-reconstructed-potential). |
| `<run>_<table>.csv` | CSV tables written by the model's readouts, e.g. `<run>_data.csv` for `output.table("data.csv")`. |
| `<run>.log.json` | The parameters, only if HDF5 is off or `/output/json = true`. |
| `<run>_snapshot_<nnn>.h5` | [Snapshots](snapshots.md), if requested. |

The parameters of the run are always recorded: in `/config` of the HDF5 file, else in the `.log.json`.

### HDF5 layout

| location | content |
| --- | --- |
| `/config` | The full parameter tree of the run. |
| root attributes `finished`, `crashed` | How the run ended: `(1, 0)` completed, `(1, 1)` stopped on an error with its output closed cleanly, `(0, 0)` killed or still running. |
| `/scalars/<name>` | One dataset per scalar written by the readouts, one entry per frame. |
| `/maps/<name>`, `/coordinates/<name>` | Grid-valued data written with `hdf5.map(...)`, e.g. dressing functions on a momentum grid. |
| `/FE/<frame>` | The FE solution of each frame (six-digit frame number). |
| `/potential/<frame>`, `/eom_potential/<frame>` | The reconstructed potentials. |

Each frame is written completely and the file is closed after it, so the file can be read while the run
continues, and a killed run leaves all frames before the last one intact.

### Reading output in Python

```python
import DiFfRG.file_io as io

sim = io.SimulationData("output.h5")
sim.params["physical"]["T"]       # the configuration
sim.scalars["k"]                  # a scalar written by the readouts, as an array over frames
sim.fem_data[-1]                  # the FE solution of the last frame (read on access)

io.read_csv("output_data.csv")    # a readout table
```

`DiFfRG.phasediagram.get_all_sims(folder)` reads all runs in a folder, see the [Python API](../python.md).

## Writing data from a model

A model writes data in its `readouts` method, which the assembler calls once per frame at the equation of motion
(the minimum of the potential). It receives a handle to the current frame:

```cpp
template <int dim, typename DataOut, typename Solutions>
void readouts(DataOut &output, const Point<dim> &x, const Solutions &sol) const
{
  const double m2Pi = get<"fe_functions">(sol)[idxf("u")];

  auto table = output.table("data.csv");       // <run>_data.csv, one row per frame
  table.value("k", k);
  table.value("m2Pi", m2Pi);

  auto hdf5 = output.hdf5();                   // <run>.h5
  hdf5.scalar("m2Pi", m2Pi);                   // /scalars/m2Pi
}
```

A table must receive the same columns in every frame. Tables get an RG-time column `t` and, if `/physical/Lambda` is set, a column `k [GeV]`
$= \Lambda e^{-t}$; `table.set_Lambda(L)` uses a different $\Lambda$. Pure-variable models write grid-valued data with
`hdf5.map("ZA", coordinates, data)`, see [Tutorial 4](../tutorials/tut4.md).

A model can evaluate its readouts at several points (e.g. different definitions of the EoM) by implementing
`readouts_multiple`; see `Examples/QuarkMesonLPAprime/model.hh`.

### CSV format

CSV tables have one header row and one row per frame, comma-separated, in scientific notation. The C++ reader
(`DiFfRG/common/csv.hh`) and the Python reader (`DiFfRG.file_io.csv`) skip `#` comment lines, blank lines and rows
with the wrong number of fields, and read empty or non-numeric cells as `NaN`.

## Verbosity

`/output/verbosity` controls what is printed to the console:

| level | console shows |
| --- | --- |
| 0 | warnings and errors only |
| 1 | + info messages and residual/timestep progress |
| 2 | + Jacobian and linear-solver progress, debug messages |
| 3 | + factorization and output timings (also `<run>_output_timings.csv`) |
| 4 | as 3 |
| 5 | every single event; slows the run down, for debugging only |

At levels 1–4, progress of each kind is summarized at most once per second. Every progress line carries `wall=`,
the wall time since the start of the run, and `avg=`, the mean cost of the operation it reports.

The `.log` file is controlled separately by `/output/log_level` (default 2 = info; 1 = debug, 0 = trace) and
always receives every message at or above that level, independent of the verbosity.

## Advanced

### Custom field output

Inside a frame, additional finite-element fields can be written as their own series:

```cpp
data_out.write_frame(time, [&](auto &frame) {
  auto auxiliary = frame.fields("auxiliary");
  auxiliary.attach(dof_handler, auxiliary_solution, "auxiliary_u");
});
```

This writes `<run>_auxiliary.pvd` and the HDF5 group `/auxiliary`. The names `FE`, `potential`,
`eom_potential`, `scalars`, `maps` and `coordinates` are reserved.

### Diagnostics

Code that runs outside of output frames (a custom timestepper, a model hook) can write CSV records through a
`DiagnosticPort`, obtained from `data_out.diagnostic_port()`. It is thread-safe and writes each record atomically:

```cpp
diagnostics.record("jacobian_eigenvalues.csv", time, {{"lambda_min", lambda_min}, {"lambda_max", lambda_max}});
```

With `/timestepping/implicit/jacobian_diagnostics = true`, every implicit timestepper writes one record per
Jacobian to `<run>_jacobian_diagnostics.csv`: matrix size and norms, diagonal dominance, and for UMFPack-based
solvers the factorization time and a condition estimate. It costs a sweep over the Jacobian per record and is
off by default.

### Messages and progress from custom code

`data_out.report_port()` returns a cheap, copyable `ReportPort` with `info`, `warn`, `error` and `progress`
methods. A default-constructed `ReportPort` prints to the console only. Custom linear solvers derived from
`AbstractLinearSolver` receive the timestepper's port automatically. Structured progress records are built from
a `ProgressTopic` and a `ProgressEvent`; see their class documentation.

### Temporary output

A default-constructed `OutputSession` writes to a fresh temporary directory, deleted when the session is
destroyed; useful in tests. `OutputPath::temporary(TemporaryRetention::keep)` keeps the directory for inspection.

### Asynchronous writing

VTK and HDF5 output is written by a background thread, so the time stepping does not wait for the disk. Code that
opens the run's HDF5 file itself during the run must call `data_out.drain()` first. `Config::OutputSettings`
(constructed in C++, not from the parameter file) has `asynchronous = false` to write synchronously, e.g. for
debugging.
