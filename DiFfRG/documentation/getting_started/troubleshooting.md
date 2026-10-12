# Troubleshooting

## Building

**`find_package(DiFfRG)` fails.** The application looks in the paths given to `find_package(... HINTS ...)`,
by default `$HOME/.local/share/DiFfRG/`. If the library was installed elsewhere, pass
`-DCMAKE_PREFIX_PATH=<prefix>` to `cmake` or adapt the hint.

**A segfault right at startup, or in code that looks correct, after updating the library.** Object files of the
application compiled against older headers do not match the new library. Delete the application's build directory
and build it again. The same applies to generated `flows/` libraries.

**The GPU code aborts at startup, or every run spends minutes in JIT compilation.** The code was compiled for a
different GPU architecture than the one it runs on; see
{ref}`choosing-the-gpu-architecture`.

**The build runs out of memory.** Generated flow kernels, in particular CUDA ones, can need several GB per
translation unit. Lower the parallelism (`make -j4`).

## Running

**`timestepping got stuck at t = ...` or `timestep failure`.** The adaptive stepper could not take a step larger
than `minimal_dt`, or failed repeatedly. In fRG flows the usual causes are:

- the potential develops a feature the grid does not resolve, typically near the minimum when it moves or when a
  flat region forms. Refine the grid there (`x_grid` with several blocks) or use mesh adaptivity;
- the flow runs into a singularity of the truncation, e.g. a propagator pole ($k^2 + m^2 \to 0$). This is physics,
  not numerics: check the initial conditions and the truncation;
- tolerances that are too loose for implicit stepping with symmetry breaking: use `rel_tol` ≤ 1e-7.

Run with `-si /output/verbosity=2` to see the step sizes and Newton iterations up to the failure. For badly
scaled Jacobians, try a [scaled linear solver](timesteppers.md#scaled-solvers).

**Very many Newton iterations or Jacobian evaluations.** The Jacobian is inaccurate. Check that `flux`, `source`
and `extract` do not convert to `double` internally (e.g. by storing intermediate values in `double` variables):
with `def::AD`, that silently drops derivatives. Extractors that change at every evaluation also slow IDA down,
see the warning in [Tutorial 5](../tutorials/tut5.md).

**The result depends on the tolerance or the grid.** For flows with kinks or shocks, the position of the
non-smooth feature is set by the time-stepping tolerance as much as by the grid. Converge both.

**An abort mentioning `map()` during assembly.** A model called a `map()` of a generated integrator from `flux`,
`source` or `evaluate_batch`. Use `map_points()` there, see [Tutorial 6](../tutorials/tut6.md) and
[MPI](mpi.md#rules-for-application-code).

**A warning that several thread counts disagree.** Environment variables, the batch scheduler and
`/discretization/threads` all set the CPU thread count; the warning names the one used. See
[CPU threads](mpi.md#cpu-threads).

**Warnings that a parameter uses its default.** The key is missing from the parameter file, and its default is a
poor guess for most flows. Set it explicitly; see the [parameter reference](parameters.md).

## Output

**The output of an earlier run is gone.** A run overwrites the files of any earlier run with the same
`/output/folder` and `/output/name`, including a restart from a snapshot without a new `-ss /output/name=...`.

**An HDF5 error when reading the run's file during the run.** Call `data_out.drain()` before opening the file
from within the application. From outside, e.g. in Python, the file can be read at any time; only the frame
currently being written is missing.
