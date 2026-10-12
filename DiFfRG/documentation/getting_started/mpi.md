# Running with MPI and several GPUs

## What is distributed

Running an application with `mpirun -n N` spreads the momentum integrals evaluated through `map()` over the `N`
ranks, and with one GPU per rank over `N` GPUs. No change to the model, the generated flows or `main` is needed.

- **Only the external grid of a `map()` is split.** Each rank evaluates a slice of the grid points; every single
  integral is still summed on one rank, in the original order. The result is therefore bitwise identical to a
  serial run at any rank count. Grids of any dimension are split.
- **FE assembly is not distributed.** Every rank assembles the whole field-space mesh, redundantly. For a model
  with FE functions this part of the work does not get faster with more ranks. With several ranks on one node it
  even gets slower, since the ranks share the node's CPU threads. For such models, use one rank per node.
- Pure-variable systems built on `map()` (e.g. [Tutorial 4](../tutorials/tut4.md)) benefit most.

## Building

MPI is a single switch for the whole dependency superbuild: deal.II, SUNDIALS, PETSc and DiFfRG must all agree,
and a mixed configuration does not work. Enable it when installing:

```bash
bash install_diffrg.sh --mode source --mpi        # or the pre-built bundle with --mpi
cmake <DiFfRG source> -DMPI=ON -DCMAKE_INSTALL_PREFIX=<prefix>    # superbuild directly
```

A library build configured from `DiFfRG/` against an existing bundle inherits the bundle's MPI setting. See
[Installation](installation.md#build-options).

Some timesteppers have no distributed implementation (`TimeStepperBoostABM`, `TimeStepperBoostRK*`, implicit
Euler, TR-BDF2); see [Timesteppers](timesteppers.md). For a pure-variable system with an explicit stepper, use
`RectangularMeshSerial` or no mesh at all.

## Launching

```bash
mpirun -n 4 ./app -p parameter.toml
srun --ntasks=4 --gpus-per-task=1 ./app -p parameter.toml     # Slurm, one GPU per rank
```

Each rank picks a GPU from the devices visible to it (round robin over node-local ranks). If two ranks on a node
end up on the same physical GPU, every affected rank prints a warning to stderr; the run stays correct but will
not scale. Give each rank its own GPU, e.g. with `--gpus-per-task=1` or a per-rank `CUDA_VISIBLE_DEVICES`.

Only rank 0 writes output files.

### Tuning knobs

None of these needs to be set for a correct run.

| knob | where | meaning |
| --- | --- | --- |
| `DIFFRG_MAP_VERBOSE=1` | environment | Print the distribution plan once: per flow, its split width and owning ranks, and the load per rank. |
| `DIFFRG_MAP_QUANTUM` | environment | Minimum amount of work (kernel evaluations) per rank below which a flow is not split further. Overrides the hardware-derived default. |
| `DIFFRG_MAP_VERIFY_EVERY` | environment | How often the ranks check that they agree on the plan (default every 64 batches). Set to `1` when debugging a hang or a wrong distributed result. |
| `/integration/map_quantum`, `/integration/map_verbose` | parameter file | As the environment variables, which take precedence. |

## CPU threads

Each rank sizes its CPU thread pool (used by FE assembly and CPU integrators) from the first of these that is set:

| # | source | effect |
| --- | --- | --- |
| 1 | `DiFfRG_NUM_THREADS` | Threads per rank, verbatim. |
| 2 | the launcher's allocation: CPU affinity mask (`srun --cpus-per-task`, `mpirun --bind-to`, `taskset`), else `SLURM_CPUS_PER_TASK` etc. | Threads per rank, verbatim. |
| 3 | `/discretization/threads` | Threads per rank, verbatim. |
| 4 | `OMP_NUM_THREADS`, `DEAL_II_NUM_THREADS`, `KOKKOS_NUM_THREADS` | Used only if nothing above is set. |
| 5 | nothing | The CPUs this rank can see, split evenly among the node-local ranks that share them. |

If several sources disagree, the run prints a warning naming each one and the value it used.

## Rules for application code

- **Never call `map()` from `flux`, `source` or `evaluate_batch`.** `map()` is collective over all ranks, while
  assembly is not. The call aborts with an error message. Use `map_points()` inside assembly (see
  [Tutorial 6](../tutorials/tut6.md)), or compute the value once in `extract()` or `dt_variables`.
- **Do not branch on data only one rank has.** Only rank 0 writes the output files, so an application that reads
  a value back from its own output and decides, say, whether to start another run, must broadcast that value
  (`DiFfRG::MPI::bcast`, `DiFfRG::MPI::all_of` in `DiFfRG/common/mpi.hh`). Otherwise the ranks diverge and the
  next collective call hangs.
- **An exception inside a deferred batch of maps is not recoverable under MPI**; the run aborts with a diagnosis
  instead of hanging.

## Checking a distributed run

Run the same short configuration with 1 and with several ranks and compare the HDF5 output exactly, e.g. with
`h5diff -d 0`. The results must agree bit for bit. To measure scaling, time the stepping itself (e.g. the
`[res ]` summary in the log) rather than the whole process, which includes MPI and GPU start-up.
