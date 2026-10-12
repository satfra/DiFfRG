# Parameter reference

Every key the library reads from the parameter file, with its default. Keys are addressed by their path, both in
C++ (`json.get_double("/timestepping/output_dt")`) and on the command line (`-sd /timestepping/output_dt=0.05`).
The `/physical` section belongs to the model and is not listed here.

A missing key falls back to its default. For keys marked *(warns)* the default is a poor guess for most flows, and
the run warns once when it uses it. `./app --generate-parameter-file` writes a `parameter.toml` containing the
commonly used keys at their defaults (`--gen-json` for JSON).

## Command line

| flag | effect |
| --- | --- |
| `-p <file>` | Parameter file. `.toml`/`.tml` is read as TOML, anything else as JSON. Default: `parameter.json`, else `parameter.toml`. |
| `-sd`, `-si`, `-sb`, `-ss` `<path>=<value>` | Override a double, integer, boolean or string parameter. An existing key keeps its type; a missing one is created with a warning. |
| `--generate-parameter-file`, `--gen-toml`, `--gen-json` | Write a default parameter file named like the parameter file and exit. An existing file is never replaced. |
| `--restart <snapshot.h5>` | Continue from a snapshot, taking the configuration from it; see [Snapshots](snapshots.md). Cannot be combined with `-p`. |
| `--snapshots-k <k1,k2,...>`, `--snapshots-t <t1,t2,...>`, `--stop-after-last-snapshot` | Set the snapshot schedule from the command line. |
| `--help` | Print the usage. |

## `/discretization`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `fe_order` | uint | 3 *(warns)*; 0 for FV | Polynomial degree of the finite elements. |
| `overintegration` | uint | 0 | Extra quadrature points per direction in the assembly. |
| `output_subdivisions` | uint | 1 | Sub-cells per cell in the VTK output. |
| `threads` | uint | 0 | CPU threads for assembly and CPU integrators; 0 = automatic. See [CPU threads](mpi.md#cpu-threads) for what overrides it. |
| `EoM_abs_tol` | double | 1e-12 | Tolerance of the search for the equation of motion (EoM), the minimum of the potential. |
| `EoM_max_iter` | uint | 100 | Iteration limit of the EoM search within a cell. |
| `EoM_smoothing_length` | double | -1 | Smoothing of the reconstructed potential used by the EoM search: -1 automatic (twice the smallest initial cell width), 0 off, > 0 a length in field units. |
| `EoM_bound_tolerance` | double | 1e-12 | When a point counts as lying on a cell boundary in the EoM search. |
| `EoM_armijo_coefficient` | double | 1e-4 | Sufficient-decrease coefficient of the EoM line search, in (0, 1). |
| `EoM_max_backtracks` | uint | 20 | Step halvings allowed in the EoM line search. |
| `raw_potential_order` | uint | 2 | Polynomial order (2 or 3) of the potential reconstructed for output and extractors. |

### `/discretization/grid`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `x_grid` | string | *(warns)* | Cells along axis 0: comma-separated blocks `start:step:stop`, e.g. `"0:1e-4:1e-2, 1e-2:1e-3:1"`. |
| `y_grid`, `z_grid` | string | *(warns)* | The same for axes 1 and 2; only read in 2D / 3D. |
| `refine` | uint | 0 | Global refinements of the initial mesh. |

### `/discretization/adaptivity`

Only used if `main` passes an `HAdaptivity` to the timestepper (see [Tutorial 7](../tutorials/tut7.md)).

| key | type | default | meaning |
| --- | --- | --- | --- |
| `start_adapt_at` | double | 0 | RG time of the first adaptation. |
| `adapt_dt` | double | 0.2 | RG time between adaptations. |
| `level` | uint | 0 | Maximum refinement level of a cell, counted from the coarse grid (so with `grid/refine` = r, only `level` > r refines); 0 disables adaptation. |
| `indicator` | string | `"auto"` | `"curvature"`: $h_K^2\,\lvert\partial^2 u\rvert/(\max u - \min u)$ per cell, about 8 times the relative resolution error; `"model"`: the model's `cell_indicator`/`face_indicator`; `"auto"`: the model's if it defines one, else curvature. |
| `predict` | bool | true | Add each cell's indicator growth since the previous adaptation, so that forming structures are refined one adaptation earlier. |
| `marking` | string | `"tolerance"` | `"tolerance"`: refine cells above `tolerance`, coarsen below `coarsen_tolerance`; `"fraction"`: refine/coarsen fixed fractions of the total indicator at every adaptation. |
| `tolerance` | double | 3e-4 | `"tolerance"`: refine cells with a larger indicator. |
| `coarsen_tolerance` | double | `tolerance` / 8 | `"tolerance"`: coarsen cells with a smaller indicator. |
| `refine_percent` | double | 0.1 | `"fraction"`: fraction of the total indicator whose cells are refined. |
| `coarsen_percent` | double | 0.05 | `"fraction"`: fraction of the total indicator whose cells are coarsened. |
| `coarsen_buffer` | uint | 2 | Layers of cells around a refined cell that are not coarsened. |
| `coarsen_age` | uint | 10 | Cells created by the last that many mesh changes are not coarsened. |

### `/discretization/batched`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `max_stacked_points` | uint | ≈ 256 MB of AD data (FEM), 16 MB (KT) | Points per stacked evaluation when computing Jacobians with `evaluate_batch`, see [Tutorial 6](../tutorials/tut6.md). |

## `/timestepping`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `output_dt` | double | 0.1 | RG-time interval between output frames. |
| `final_time` | double | — | Not read by the library; by convention the end time applications pass to `run()`. |

### `/timestepping/implicit`

Read by IDA, implicit Euler, TR-BDF2 and the FE part of the IDA hybrids.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `abs_tol` | double | 1e-13 *(warns)* | Absolute local error tolerance. |
| `rel_tol` | double | 1e-7 *(warns)* | Relative local error tolerance. |
| `dt` | double | 1e-4 | Initial step. |
| `minimal_dt` | double | 1e-14 | Smallest step; a stepper stuck below it aborts the run. |
| `maximal_dt` | double | 1 | Largest step. |
| `max_non_linear_iterations` | uint | 3 | Newton iterations per IDA step attempt. |
| `iterative_refinement` | bool | true | Refine every `UMFPack` solve iteratively. |
| `local_tolerance_refresh` | double | 4 | For models with per-dof tolerances: IDA adopts new tolerances when one would be tighter than the current one by more than this factor. |
| `jacobian_diagnostics` | bool | false | Write `<run>_jacobian_diagnostics.csv`, see [Output](output.md#diagnostics). |

Debugging aids for IDA, all off by default: `ida_callback_trace` (print every failed callback at verbosity ≥ 5,
with `ida_callback_trace_min_t`, `ida_callback_trace_max_lines`, `ida_callback_trace_successes`) and
`ida_error_dof_diagnostics` (report the `ida_error_dof_diagnostics_top_n` dofs with the largest error on rejected
steps).

### `/timestepping/explicit`

Read by the explicit steppers and the variable part of the IDA hybrids.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `dt` | double | 0.01 *(warns)* | Fixed-step steppers: the step. BoostRK: the initial step. ABM: the largest step. |
| `minimal_dt` | double | 1e-16 | Smallest step of the ABM error control. |
| `maximal_dt` | double | 1e16 | Largest explicit step. |
| `abs_tol` | double | 1e-3 *(warns)* | Absolute local error tolerance. |
| `rel_tol` | double | 1e-3 *(warns)* | Relative local error tolerance. |
| `error_control` | bool | true | ABM: adapt the step to the tolerances; `false` uses `dt` throughout. |
| `detect_stuck` | bool | true | BoostRK: abort when the step size collapses. |
| `coupling_mode` | int | 1 | IDA + ABM hybrid: how IDA sees the variables, 0 lag, 1 stagger, 2 predict. |

`TimeStepperSUNDIALS_IDA_BoostABM` throws unless `minimal_dt <= dt <= maximal_dt` in this section.

### `/timestepping/snapshots`

See [Snapshots and restarts](snapshots.md).

| key | type | default | meaning |
| --- | --- | --- | --- |
| `k` | list of doubles | none | RG scales at which to write snapshots; needs `/physical/Lambda`. |
| `t` | list of doubles | none | RG times at which to write snapshots. |
| `snap_to_output_grid` | bool | true | Move snapshots onto the nearest output time. |
| `stop_after_last` | bool | false | End the run after the last snapshot. |

## `/integration`

Read by the momentum integrators when they are constructed.

| key | type | default | meaning |
| --- | --- | --- | --- |
| `x_order` | uint | 32 *(warns)* | Quadrature order of the radial loop-momentum integral. |
| `cos1_order`, `cos2_order`, `phi_order` | uint | 8 *(warns)* | Quadrature orders of the angular integrals, for integrators that have them. |
| `x_extent_tolerance` | double | 1e-5 *(warns)* | Relative tolerance for choosing the upper end of the radial integral, beyond which the regulator insertion is negligible. |
| `jacobian_quadrature_factor` | double | 0.8 | Factor on all orders for the AD (Jacobian) evaluations. |
| `vacuum_quad_size` | uint | 64 | Size of the frequency quadrature in vacuum integrals. |
| `matsubara_precision_factor` | double | 1 | Accuracy factor for numerical Matsubara sums. |
| `min_matsubara_size`, `max_matsubara_size` | uint | 8, 128 | Bounds on the number of Matsubara terms summed explicitly. |
| `force_exact_matsubara_sum` | bool | kernel's choice | Allow (`true`) or forbid (`false`) the exact frequency sum, overriding the generated kernel. |
| `matsubara_extent_margin` | double | 1 | Factor on the frequency cutoff of the exact sum. |
| `map_quantum`, `map_verbose` | double, bool | 0, false | Distribution of `map()` over MPI ranks, see [MPI](mpi.md). |

Lattice integrators additionally read `/integration/lattice/{N_t, N_s, a_t, a_s, q0_symmetric}`.

## `/output`

See [Output](output.md).

| key | type | default | meaning |
| --- | --- | --- | --- |
| `folder` | string | `"./"` | Directory of all output. |
| `name` | string | `"output"` | Prefix of all output files. |
| `field_directory` | string | value of `name` | Subdirectory of `folder` for the VTK frames. |
| `vtk` | bool | true | Write VTK output. |
| `hdf5` | bool | true | Write HDF5 output. |
| `json` | bool | false | Also write `<name>.log.json` with the parameters when HDF5 is on. |
| `verbosity` | int | 0 | Console progress, 0–5, see [Output](output.md#verbosity). |
| `log_level` | int | 2 | Lowest severity written to `<name>.log`: 0 trace, 1 debug, 2 info, 3 warn, 4 error. |
| `max_pending_frames` | uint | 2 | VTK frames that may be queued for writing. |
| `log_queue_size` | uint | 8192 | Capacity of the asynchronous log queue. |

## Environment variables

| variable | meaning |
| --- | --- |
| `DiFfRG_NUM_THREADS` | CPU thread count; overrides everything else, see [CPU threads](mpi.md#cpu-threads). |
| `DIFFRG_MAP_VERBOSE`, `DIFFRG_MAP_QUANTUM`, `DIFFRG_MAP_VERIFY_EVERY` | Distribution of `map()` over MPI ranks, see [MPI](mpi.md). |

## Ignored keys

`/discretization/mesh_workers`, `/discretization/batch_size` and `/discretization/kokkos_threads` are no longer
used; the run warns if it finds them. A `/restart` section in a parameter file is an error: use `--restart`.
