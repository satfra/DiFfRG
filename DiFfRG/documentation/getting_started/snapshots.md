# Flow snapshots and restarts

A flow can write *snapshots* of its complete state at chosen RG scales, and a later run can continue
from one of them instead of starting at the initial scale Λ. The later run may use a changed
configuration. This pays off whenever a change only matters below some scale:

- **Parameter scans.** Temperature and chemical potential only affect the flow once k has come down
  to the order of T and μ. One seed run can be continued from a snapshot above that scale for
  every point of a phase diagram.
- **IR work.** When you are debugging an IR instability, tuning tolerances for the late flow,
  changing readouts or EoM settings, the UV part of the flow is recomputed for nothing.
- **Long runs.** A flow killed by a wall-time limit can be continued from its last snapshot.

## Is a restart with a changed configuration meaningful?

The flow above the snapshot scale k_s was computed with the snapshot's configuration. A restart
with a changed configuration is exact only if the change would not have affected the flow above
k_s. DiFfRG cannot decide that for you. It logs every entry that differs from the snapshot's
configuration as a warning, but it does not refuse to run.

For T and μ, the effects above k_s are suppressed but not zero:

- With a spatial (3d) Litim regulator all quark energies satisfy E ≥ k. At T = 0 the quark loops
  therefore do not depend on μ at all while k > μ, so a restart at k_s > μ is exact.
- At T > 0 the thermal distribution functions give corrections of order e^{-k/T} for bosons and
  e^{-(k-μ)/T} for fermions at scale k.

As a rule of thumb, choose k_s ≳ μ_max + 10 T_max. Then check one point of the scan against a full
run from Λ before trusting the rest. The corrections act like a shift of relevant couplings, such
as a mass, so near a critical point they are *not* washed out by the flow.

Also check how much there is to gain. The time saved per scan point is the wall time the flow
needs to get from Λ down to k_s. The `wall=` column of a run's progress log at that k shows it
directly. This can be more than you might expect: IDA starts with small steps, and Matsubara sums
are largest for k ≫ T. For the quark-meson example (Λ = 0.65 GeV, T = 0.02 GeV), about half of the
wall time is spent above k = 0.45 GeV. On the other hand, if T and μ already matter close to Λ,
e.g. T ≳ 0.05 GeV in that example, there is nothing to reuse.

As an illustration: restarting that example from a μ_q = 0 snapshot at k = 0.45 GeV with
μ_q = 0.2 GeV, T = 0.02 GeV reproduces σ(k_IR) of a full run to 1e-6. The pion mass differs by
3e-4, although the μ-dependence above k_s is only of order e^{-12}; the small IR pion mass
amplifies it.

## Writing snapshots

```json
"timestepping": {
  "snapshots": {
    "k": [2.0, 1.0],
    "t": [],
    "snap_to_output_grid": true,
    "stop_after_last": false
  }
}
```

or, in a `parameter.toml`:

```toml
[timestepping.snapshots]
k = [2.0, 1.0]
t = []
snap_to_output_grid = true
stop_after_last = false
```

- `k`: RG scales, converted to t = ln(Λ/k) using `/physical/Lambda`. `t`: RG times directly. Both
  lists are merged.
- `snap_to_output_grid` (default `true`): moves each snapshot to the nearest output time
  t_start + n·`output_dt`, so the output grid of the run is unchanged. The file records the exact t
  and k it was taken at.
- `stop_after_last` (default `false`): ends the run right after its last snapshot. Use it for seed
  runs that exist only to produce the snapshot.

The same can be given on the command line, which also works when the parameter file has no
`snapshots` section:

```bash
./QuarkMesonLPAprime --snapshots-k 2.0,1.0 --stop-after-last-snapshot
```

Each snapshot is written to `<output name>_snapshot_<nnn>.h5` next to the run's other output. The
file is written under a temporary name and then renamed, so other runs never read a half-written
snapshot.

The flow is split into segments at the snapshot times, and the snapshot is taken at the end of a
segment. This makes the snapshot the solver's exact accepted state for every time stepper. At an
ordinary output step, the hybrid IDA + explicit steppers only hold interpolated variables. The
cost is one solver restart per snapshot, i.e. a few small steps.

## Restarting

```bash
./QuarkMesonLPAprime --restart output/seed_snapshot_000.h5 -sd /physical/T=0.1 -ss /output/name=T0.1
```

A restarted run takes its whole configuration from the snapshot, which carries the seed run's full
parameter tree, whether the seed read a JSON or a TOML parameter file. No parameter file is read (`-p` together with `--restart` is an error), and
`/restart` in a parameter file is an error too. Only the command line changes anything:
`-sd`, `-si`, `-sb` and `-ss` override entries of the snapshot's configuration, and, as usual, only
entries that exist there. Every override is printed at the start of the run, with its old and new
value.

What happens to the snapshot schedule depends on where the restart writes:

- **Continuing the same run.** If the snapshot is one of this run's own, i.e. `--restart` names
  `<folder>/<name>_snapshot_<nnn>.h5` and `/output` is not overridden, the restart continues that
  run. It keeps the snapshot schedule and numbers the later snapshots on from `nnn + 1`, exactly as
  the uninterrupted run would have. A run that died between its snapshots at 0.5 and 1.0 is
  continued with

  ```bash
  ./App --restart output/output_snapshot_000.h5
  ```

  If a later snapshot already exists, the run has been continued before, and the restart stops
  with an error instead of overwriting it: restart from the latest snapshot.
- **A new run.** With a different `/output/name` or `/output/folder`, e.g. every point of a
  phase-diagram scan, the restart starts without a snapshot schedule and writes snapshots only if
  its command line asks for them. Restarts can be chained this way as well.

In both cases `--snapshots-k`, `--snapshots-t` and `--stop-after-last-snapshot` on the command line
replace the schedule as a whole.

On its first call, `TimeStepper::run`:

1. reads the snapshot, rebuilds the mesh if it was adapted, and restores the spatial state, the
   variables, the model's history-dependent state (see below) and the adaptation schedule;
2. logs the differences between the configuration and the snapshot's, i.e. the overrides.
   `/output`, `/restart` and `/timestepping/snapshots` are not listed;
3. continues from the snapshot's time. The `t_start` passed to `run()` is ignored.

The application code does not change: set up the initial condition as usual, with
`initial_condition.interpolate(model)`, and call `time_stepper.run(initial_condition, 0., t_final)`.

A restarted run writes its output from the snapshot's time on. The part of the flow above k_s is in
the seed run's output. Without `-ss /output/name=...` or `-ss /output/folder=...` it writes to the
seed run's output name and replaces its `<name>.h5` and friends; the output above the snapshot's
scale is then lost. Snapshot files are never overwritten: a run that would write one that exists
stops before it starts stepping.

Differences that make the data impossible to load are errors, not warnings. These are a different
coarse grid, finite element, number of variables or dimension.

## History-dependent model state

The snapshot holds the discrete state. Most models are a function of that state and of t, and
nothing more is needed. A model that *latches* something must also carry that value across a
restart, because nothing in the state records it. An example is a model that freezes its EoM once
it has jumped. Implement

```cpp
void save_state(DiFfRG::ModelState &state) const
{
  state.set("last_EoM", last_EoM);
  state.set("lock_EoM", lock_EoM);
}

void load_state(const DiFfRG::ModelState &state)
{
  last_EoM = state.get<double>("last_EoM");
  lock_EoM = state.get<bool>("lock_EoM");
}
```

Both are optional and detected at compile time. `ModelState` stores named doubles, integers,
booleans and `std::vector<double>`. If a snapshot carries model state but the model has no
`load_state`, the restart warns and continues. The quark-meson example
(`Examples/QuarkMesonLPAprime/model.hh`) shows the pattern.

## Python and Mathematica

```python
import DiFfRG.phasediagram as pd
import DiFfRG.file_io as io

seed_params = [["/physical/T", 0.005], ["/physical/muq", 0.0]]
seed = pd.make_seed(exe, seed_params, k_values=[2.0], folder="seeds/")
pd.run_point(exe, [["/physical/T", 0.1], ["/physical/muq", 0.2]], folder="pd/", restart=seed[0]["path"])

io.list_snapshots("seeds/")   # t, k, config, ... of every snapshot in a folder
io.find_snapshot("seeds/" + pd.get_name(seed_params), k=2.0, rel_tol=1e-2)
```

In Mathematica, `SnapshotInfo[file]` and `ListSnapshots[run]` return the same metadata.

## File format

HDF5, `format_version` 1:

| Location | Content |
|---|---|
| `/` attributes | `t`, `k` (NaN without Λ), `Lambda`, `dim`, `last_adaptation_time` |
| `/mesh` | attributes `fe_name`, `dofs_per_cell`, `n_coarse_cells`, `n_dofs`; dataset `active_cells` (deal.II `CellId`s) |
| `/state/spatial` | local dof values of every active cell, cell by cell |
| `/state/variables` | the variables |
| `/model/<name>` | the model's `ModelState` |
| `/config_json`, `/config` | the configuration of the writing run, verbatim and as a group tree |

The spatial state is stored per cell rather than per global dof. This makes a snapshot independent
of the dof numbering, so one written with N MPI ranks can be read with M ranks, and an adapted mesh
can be rebuilt from its cell ids.

## Limitations

- The coarse grid and the finite element must match the snapshot. The state is copied, not
  interpolated onto a different discretization.
- Restoring an adapted mesh works for every discretization. Adapting further in an MPI run with the
  IDA steppers, or with the hybrid steppers, is limited exactly as in a run without snapshots.
- The solver's internal history (IDA's order and step size, the ABM multistep history) is not part
  of the snapshot. Both the snapshotting run and the restarted run rebuild it at the snapshot time,
  which is why they agree.
