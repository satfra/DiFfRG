# Changelog

## Version 2.0

### Changed

- **Breaking:** every assembler (CG, DG, LDG, KT-FV) evaluates the model in batches: it gathers the
  solution at all quadrature points (or face traces), calls the model's `evaluate_batch(out, batch)` once for
  all of them, and scatters the results; no `MeshWorker::mesh_loop` remains. The default `evaluate_batch` calls
  the per-point `flux`, `source` and `diffusion_flux` in a parallel loop, so a model that does not override it
  keeps working. A model that overrides it evaluates its momentum integrals for all points with one
  `map_points()` call, on the CPU or the GPU (Tutorial 6). Jacobians of a `def::AD` model are the forward AD of
  `evaluate_batch`, with all seed directions stacked into one batch.
  **Migration:** none for most models. Changes a model can notice:
  - The dDG assembler is removed: DG now hands the model the same inputs as CG. CG and DG drop `"fe_hessians"`
    (and `"fe_derivatives"`) when the model declares `batch_reads_hessians = false`
    (`batch_reads_derivatives = false`); reading an absent entry does not compile (`tuple_has<name, Solution>`
    tests for one). **Migration:** replace `dDG::Assembler` by `DG::Assembler`; a DG model that reads only the
    values should declare `static constexpr bool batch_reads_derivatives = false`, otherwise its jacobian
    carries unused derivative seeds.
  - `def::AD_dual` is removed, and with it the per-point AD jacobian classes (`def::ADjacobian_flux`,
    `_source`, `_flux_source`, `_numflux`, `_boundary_numflux`). `def::AD` and `def::FE_AD` keep working; a
    model with hand-written per-point jacobians and without `def::AD` is evaluated per point as before.
  - `/discretization/mesh_workers` and `/discretization/batch_size` are ignored, with a warning.
  - `ldg_flux_source_batch<k>` is named `ldg_evaluate_batch<k>`.
- **Added:** `map_points(dest, args...)` on the vacuum and finite-temperature quadrature integrators: the
  integral at `dest.size()` points in one launch, each argument either shared or one value per point (a
  `PointSpan` or `std::vector`). Rank-local, unlike `map()`. `MakeKernel[..., "MapPoints" -> True]` emits the
  wrappers (`CT_map_points.cc`, `AD_map_points.cc`).
- **Added:** `/discretization/batched/max_stacked_points` bounds the points of one stacked AD evaluation.
- The four assemblers share one implementation of the EoM search, extractors, readouts, variables, constraints,
  mass matrix and timings (`internal::AssemblerCore`). Visible changes: the KT readouts see the unlimited
  3-point `"fe_hessians"` the KT extractors already saw (they were zero); KT reports the variables' residual time
  and `jac eval` in its summary; `reinit()` forgets the EoM and extractor cells of the previous mesh; LDG agrees
  on an extractor-cell change across ranks as CG and DG do.
- The batched assemblers report the wall time of their stages (`residual_phase_times()`,
  `jacobian_phase_times()`, an `AssemblyPhaseTimes`: extract, gather, evaluate, scatter).
- **Breaking** for custom timesteppers: `AbstractTimestepper::run(ic, t_start, t_stop)` is now a
  non-virtual driver (it implements snapshots and restarts), and the time stepping itself moves to
  the pure virtual `run_segment(ic, t_start, t_stop)`, which must return the accepted state at
  exactly `t_stop`. **Migration:** rename the `run(AbstractFlowingVariables&, ...)` override to
  `run_segment` and add `using Base::run;` if the class declares other `run` overloads. Calling
  `time_stepper.run(...)` is unchanged.
- `OutputSession::write_frame` drops a frame whose time equals that of the previous frame, so a
  flow split into segments does not write its boundary frames twice.
- CSV reading and writing are unified in `DiFfRG/common/csv.hh`, which documents the one dialect
  DiFfRG uses and which `CsvOutput`, `CSVReader` and `ExternalDataInterpolator` all go through. The
  Julia and Python analysis readers follow the same rules, so a file written by a run loads
  identically in all three languages: `#` comment lines and blank lines are skipped, a cell that is
  not a finite number reads back as `NaN` instead of raising, rows whose field count does not match
  the header are skipped, and quoted header names, CRLF endings and a UTF-8 BOM are accepted.
  Numbers are parsed with `std::from_chars`, so read-in no longer depends on the global locale.
- **Removed:** the third-party `rapidcsv` dependency, together with the CPM fetch that installed it
  into the bundle. It was also the last third-party header in the public include path pulled in by
  `csv_reader.hh`, so every translation unit including `DiFfRG.hh` compiles less.
- **Removed:** `string_to_double_array` from `DiFfRG/common/utils.hh`, a fourth ad-hoc CSV line
  parser with no callers. Use `split_csv_line` and `parse_csv_cell` from `DiFfRG/common/csv.hh`.
- **Breaking:** the application-facing discretization, assembler, timestepper and output-session
  types are keyed on one another instead of repeating the linear algebra:
  `CG::Discretization<Model, RectangularMesh<dim>>`, `CG::Assembler<Discretization>`,
  `TimeStepperSUNDIALS_IDA<Assembler>`, `OutputSession<Assembler>`. The spelled-out parameter
  lists live on the `_impl` classes, whose instantiations stay closed in the library.
  **Migration:** collapse the aliases, e.g.
  `TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, UMFPack>` becomes
  `TimeStepperSUNDIALS_IDA<Assembler>`; `OutputSession<dim, VectorType>` becomes
  `OutputSession<Assembler>` (anything exposing `dim` and `VectorType` works, e.g. the
  discretization).
- Every progress line of the run reporter carries a `wall=` column with the total wall time elapsed
  in the run, restoring the cumulative timing that the old `calc_t` column reported.
- The `QuadratureProvider`'s quadrature inventory is written into the run log, `<output name>.log`, rather than into a
  separate `<output name>_quadrature.log`.
- **Breaking:** Kokkos' host execution space is now `Kokkos::Serial`. DiFfRG's CPU parallelism is
  TBB -- the assemblers' loops are `tbb::parallel_for`, and every production flow
  instantiates its CPU integrators with `TBB_exec` -- so the Kokkos `std::threads` backend was a
  second pool of spinning workers contending for the same cores, which deal.II also dispatched its
  own vector kernels into. Build with `-DKOKKOS_THREADS=ON` to restore it.
- **Breaking:** `Threads_exec` / `Threads_memory` are renamed to `KokkosHost_exec` /
  `KokkosHost_memory`, which is what they are now that the backend behind them is Serial.
  **Migration:** rename the alias; nothing else changes. Generated flows follow automatically --
  codegen maps a `"Device" -> "Threads"` kernel to `KokkosHost_exec` (via `DeviceExecSpace[]` in
  the `TemplateParameterGeneration` package); `TBB` and `GPU` kernels are unchanged.
- **Breaking:** `/discretization/kokkos_threads` is removed. There is one CPU thread pool and
  `/discretization/threads` sizes it. DiFfRG warns if the key is still present.
- The CPU thread count is resolved from a documented precedence order, and DiFfRG warns loudly on
  stderr when several settings disagree: `DiFfRG_NUM_THREADS` > the launcher's allocation (the CPU
  affinity mask, else `SLURM_CPUS_PER_TASK` / `SLURM_JOB_CPUS_PER_NODE` / `SLURM_CPUS_ON_NODE`) >
  `/discretization/threads` > `OMP_NUM_THREADS` / `DEAL_II_NUM_THREADS` / `KOKKOS_NUM_THREADS` >
  automatic. `DEAL_II_NUM_THREADS` previously capped every one of these silently, from inside
  deal.II; it is now removed from the environment where it would undercut a higher-priority setting.
- The resolved budget is published as `DiFfRG::n_threads()` (`DiFfRG/common/threads.hh`) and is what
  the map scheduler's host/device split sizes itself against. Prefer it
  to `dealii::MultithreadInfo::n_threads()`, which is a mutable static that any `set_thread_limit()`
  call rewrites.
- **Breaking:** the interpolators (`LinearInterpolator1D/2D/3D`, `LinearInterpolatorND`,
  `SplineInterpolator1D`, `SplineInterpolator1DStack`) no longer take a memory-space template
  parameter. One object now holds its data on both host and device, `update()` leaves both current,
  and the call operator evaluates on whichever side it is invoked from. **Migration**: delete the third
  template argument, e.g. `SplineInterpolator1D<double, Coordinates1D, GPU_memory>` becomes
  `SplineInterpolator1D<double, Coordinates1D>`. A single generated flow set can now be driven
  through `GPU_exec` and `TBB_exec` without changing the interpolator declarations.
- **Breaking:** `CPU()`, `GPU()` and `get_on<MemorySpace>()` are removed, along with the lazily
  allocated twin object they returned. Call the interpolator directly instead; a host-side
  `interpolator(x)` now reads the host mirror rather than requiring `interpolator.CPU()(x)`.
  The `is_interpolator` concept no longer requires `get_on<>`.
- **Breaking:** Interpolator's `data()` is now `const` and returns `const NT *`. Writing through it would have left
  the device buffers (and, for the splines, the coefficients) stale, and there is no longer a public
  way to push such an edit. Route mutations through `update()`.
- `update()` is now a single overload taking a raw pointer; the `Kokkos::View` overload had no
  callers outside the removed twin refresh. It performs one host-side fill and one device copy
  behind a single fence, instead of the previous two fences per call.
- `operator[]` on the rank-2 and rank-3 interpolators is now explicitly row-major, matching the
  order `update()` takes its input in, independently of the mirror's Kokkos layout.
- `other_memory_space_t` is removed from `common/kokkos.hh`; the interpolators were its only users.

### Added

- Flow snapshots and restarts. `/timestepping/snapshots/k` (or `/t`) makes a run write its complete
  state, including the full parameter tree, at those RG scales to `<run>_snapshot_<nnn>.h5`, and
  `--restart <file>` makes a later run continue from such a snapshot instead of starting at Λ. A
  restarted run takes its configuration from the snapshot and reads no parameter file; `-sd` & co.
  override it, and every override is printed at the start. Restarting from one of the run's own
  snapshots continues that run with its snapshot schedule, e.g. after a wall-time limit; a restart
  into another output starts without one. No run ever overwrites an existing snapshot. This is meant for parameter scans
  whose parameters only matter below some scale (T and μ in a phase diagram), for IR debugging, and
  for continuing runs cut off by a wall-time limit. A snapshot holds the spatial state cell by cell,
  so it is independent of the dof numbering and of the MPI rank count and can rebuild an adapted
  mesh; it also holds the variables, the adaptation schedule, and history-dependent model state
  through the new optional model methods `save_state(ModelState&)` / `load_state(const ModelState&)`.
  Snapshots are taken at segment boundaries -- `run()` splits the flow there -- because only there
  does every stepper, including the IDA + explicit hybrids, hold its exact accepted state. New CLI
  flags `--restart`, `--snapshots-k`, `--snapshots-t`, `--stop-after-last-snapshot`; Python helpers
  `DiFfRG.file_io.list_snapshots/find_snapshot/snapshot_info` and `phasediagram.make_seed` /
  `run_point(..., restart=)`; Mathematica `SnapshotInfo`, `ListSnapshots`. See
  `documentation/getting_started/snapshots.md`, including when a restart with a changed
  configuration is physically meaningful.
- `-DDiFfRG_CUDA_ARCH=<list>` selects the GPU compute capabilities the library and every
  application are compiled for (`90`, or `80;90` for one binary covering both), independently
  of the dependency bundle. It defaults to the GPUs found on the build machine, and the
  installer asks for it (`--cuda-arch`). Previously the architecture was whatever the bundle
  happened to be built with, so on any other GPU the driver JIT-compiled every kernel at
  startup — which DiFfRG's generated flow kernels cannot afford, being large enough to
  overflow the driver's JIT cache and be recompiled on every run. The shipped CUDA bundle's
  floor moved to sm_75 accordingly: it now only sets the architecture of Kokkos' and deal.II's
  own kernels, not of anything the user compiles.
  `DiFfRG::Init()` reports the resolved architecture and suppresses Kokkos' now-misleading
  startup warning about it -- that one line only; other Kokkos warnings still appear.
- **Breaking:** the old `install.sh` is removed (its former curl one-liner URL now 404s).
  `install_diffrg.sh` replaces it entirely: the wizard for easy installs, its
  `--mode source` path (or driving the superbuild directly with CMake) for everything
  `install.sh` did, including the `THREADS`/`FOLDER`/`MPI` knobs as flags.
- An interactive installer, `install_diffrg.sh`: a curl-able wizard covering the typical
  choices (pre-built dependency bundle vs. full self-build, prefix, build folder, MPI/GPU/
  MUMPS/documentation features, optional copy of examples and tutorials, optional install
  of the Mathematica package into the Wolfram applications directory) with command-line
  flags for scripted use. The superbuild now forwards `DiFfRG_MATHEMATICA_INSTALL_DIR`
  to the library build so the package installs in one pass.
- The timestepper `LinearSolver` template-template parameters are variadic
  (`template <typename...>`) instead of two-argument: `GMRES` carries a defaulted third
  parameter, which GCC's relaxed matching (P0522) tolerated but AppleClang rejects.
  Fixes compilation on macOS.
- `std::format` replaced by `fmt::format` (spdlog's bundled fmt) in the two places it was
  used, restoring GCC 12 compatibility: the compiler floor is GCC >= 12 everywhere, except
  nvcc host compilation where GCC 13 is excluded by an nvcc/libstdc++-13 bug (12 and >= 14
  both validated against the CUDA bundle).
- A CUDA dependency bundle variant (`linux-x86_64-v3-cuda12`, sm_75/Turing floor with PTX
  forward-compatibility for newer GPUs) plus its `release-deps-linux-cuda` workflow;
  the wizard offers it when an NVIDIA GPU is detected. deal.II's nvcc-wrapper shim now
  installs into `bundled/bin` and self-relocates instead of recording a build-tree path.
- On-demand release workflows (`release-deps-linux`, `release-deps-macos`): build the
  pre-built dependency bundles on GitHub runners with the same scripts as a local release
  build, validate them, and upload workflow artifacts (optionally attached to a draft
  `deps-v*` release for manual publishing). The macOS/Apple-Silicon path is experimental.
- Pre-built binary dependency bundles: `install-diffrg-deps.sh` downloads a relocatable
  tarball of the full dependency superbuild (deal.II, Kokkos, Boost, TBB, SUNDIALS, HDF5, ...)
  from GitHub Releases (`deps-v*` tags), so only the DiFfRG library itself is compiled locally.
  Linux x86_64, CPU-only, compiled for `x86-64-v3` (AVX2+FMA), glibc ≥ 2.34. The release
  pipeline lives in `containers/release/`.
- A `MARCH` string option supersedes the binary `NATIVE` switch: `-DMARCH=x86-64-v3` (or any
  `-march=` value, or `none`) is threaded through every bundled dependency — including Boost,
  which previously received no architecture flag at all, so `NATIVE=ON` source builds now
  compile bundled Boost with `-march=native` too. `NATIVE=ON/OFF` keeps working unchanged.
- MPI support: FEM and FV flows can be distributed over ranks. In an MPI build a plain
  `RectangularMesh<dim>` is a partitioned triangulation and the linear algebra follows it (PETSc
  vectors and matrices); `RectangularMeshSerial<dim>` pins a mesh serial and
  `RectangularMeshParallel<dim>` names the partitioned one explicitly. Mesh and vector type must
  agree -- a mismatch would be silently wrong at runtime, so it is rejected at compile time with a
  `static_assert` naming the cause. LDG is structurally serial and requires
  `RectangularMeshSerial<dim>`. Output stays on rank 0: the timesteppers refresh ghosted
  `SolutionView` replicas outside `write_frame`, so no rank enters a collective the others never
  reach. Momentum integration distributes over ranks (and GPUs) through the SPMD `MapScheduler`,
  which splits the external grid plan-driven and bitwise-reproducibly.
- Structured runtime progress reporting: timesteppers, solvers and assemblers submit
  `ProgressEvent`s through their `ReportPort` instead of formatting console text.
  `/output/verbosity` selects what is shown -- 0 nothing, 1 residual/timestep work, 2 adds
  Jacobian, linear-solver and solver diagnostics, 3 adds factorization and output work. Levels
  1--4 are aggregated to at most one line per topic per second in both the console and the run
  log; 5 prints every event. Assemblers return a structured `SummaryEvent` from `summary()`,
  reported automatically when a run drains its output.
- Additional field series can be written next to the primary fields: `frame.fields("name")`
  returns a sink whose data goes to `<run>_name.pvd`/VTUs and, with HDF5 on, to
  `/name/<six-digit frame>` groups inside the single `<run>.h5`. Series names are one safe path
  component; `FE`, `potential`, `eom_potential`, `scalars`, `maps` and `coordinates` are reserved.
- FV runs can recover a continuous mass Hessian at nonanalytic interfaces:
  `/discretization/raw_potential_recover_mass_hessian` (default off) reconstructs it from the
  DG0 gradient models, with `/discretization/raw_potential_mass_hessian_jump_threshold` enabling
  one-sided recovery across Hessian jumps and `/discretization/raw_potential_order` selecting the
  reconstruction order. The order is also honored on the cached reconstruction path and
  invalidates the cached potential system when it changes (previously the cached path silently
  used order 2).
- The Mathematica tests run on AUMP (vendored under `Mathematica/AUMP`), which launches every test
  leaf in a fresh `wolframscript` kernel; the old `TestRunner.m` is gone. Run them via the
  `mathematica-test` build target (registered with ctest as `mathematica_tests`) or directly
  through `AUMP/Runner.wls` with `--init Mathematica/DiFfRG/Tests/init.m`.
- Every HDF5 file a run writes now carries its configuration as a browsable `/config` group -- one
  subgroup per configuration section, one attribute per leaf -- so a run's parameters can be read
  with the same tools that read its data (`h5ls -r out.h5/config`,
  `f["/config/physical"].attrs["Lambda"]`) instead of parsing JSON. This covers the run-level
  `<name>.h5` and any secondary file obtained from `OutputSession::hdf5(name)`. HDF5 has no boolean
  type, so booleans appear as 0/1 ints, arrays as their serialized JSON, and nulls are omitted.
- `/output/json` (default `false`) forces the `<name>.log.json` copy of the configuration to be
  written.
- `/timestepping/implicit/jacobian_diagnostics` (default `false`) switches the Jacobian diagnostics
  tables on. They used to be written unconditionally, which is not free: every Jacobian build paid a
  full sweep over the assembled matrix and, with a factorizing linear solver, a condition estimate
  costing several extra triangular solves. With the switch off none of that work happens and no
  `<run>_jacobian_diagnostics.csv` is created.
- HDF5 files record how their run ended, in two root attributes written when the output session
  closes. `finished` is 0 from the moment the file is created and becomes 1 once the session has
  flushed and closed everything, so a run that never got that far -- SIGKILL, OOM, node eviction --
  is still marked 0. `crashed` says whether the session closed because an exception was unwinding
  through it: a timestepper that gave up leaves `finished = 1, crashed = 1`, because its output
  *was* written out properly, it just stopped early. The three readable states are therefore
  `(0, 0)` killed or still running, `(1, 0)` ran to completion, `(1, 1)` stopped early but closed
  cleanly. `/scalars/time` tells you how far it got.
- `hdf5lib`'s `Group` gains `has_attribute`, `delete_attribute` and `overwrite_attribute`. The
  last is deliberately separate from `write_attribute`, which still fails on a name that is
  already taken -- for most attributes a second write is a bug, and that throw is what catches it.

### Removed

- **Breaking:** the `configuration_json` root attribute is no longer written to HDF5 files. The
  configuration is now in the `/config` group instead. **Migration:** replace
  `json.loads(f.attrs["configuration_json"])["physical"]["Lambda"]` with
  `f["/config/physical"].attrs["Lambda"]`.
- **Breaking:** `<name>.log.json` is no longer written when HDF5 output is on, since `/config`
  already records the configuration. **Migration:** read it from the `.h5` file, or set
  `/output/json` to `true` to get the file back. `SimulationData1D.params`
  (`DiFfRG/python/DiFfRG/file_io/vtk.py`) is `None` when the file is absent rather than raising.


### Fixed

- The EoM potential reconstruction keeps its cells, faces and per-thread FE scratch with the cached system instead
  of rebuilding them for every solve: 4 ms per solve on 32 threads for a 61-cell mesh before, 0.1 ms now.
- The timesteppers report the duration of every linear solve, also of a direct solver, which reports no iteration
  count; before, UMFPACK solves were not reported at all. Progress durations are fractional milliseconds (they were
  truncated to whole ones, so every sub-millisecond callback read 0).
- Starting a flow at `t_start > 0` now works with every stepper. SUNDIALS IDA labelled the initial
  frame t = 0 and applied its t = 0 special cases (stuck detection, error-dof monitor) to
  `t_start`; the IDA + BoostRK/ABM hybrids wrote no initial frame at all; TRBDF2 set the time to 0;
  and ImplicitEuler/TRBDF2 anchored their output grid at 0 instead of `t_start`.
- The explicit steppers end exactly at `t_stop`: ExplicitEuler overshot by up to one step and
  returned the state of the step *before* the last one, BoostABM overshot whenever `t_stop - t_start`
  was not a multiple of `dt` (it now uses the largest equal step `<= dt`), and BoostRK's
  variables-only path overshot and labelled its first post-step state as `t_start`.
- `HAdaptivity` read its uninitialised `last_adapt` on the first adaptation check; it now starts at
  -inf, so the first adaptation happens as soon as `start_adapt_at` is reached.
- `DiFfRG.phasediagram.get_command` without a folder passed the run name as `-o`, which the
  executables ignore; it now sets `/output/name`.

## Version 1.1.0

### Changed

- Fully rebuilt build system as a CMake superbuild. Git submodules and the `external/` build scripts are gone; dependencies are now orchestrated by the top-level `CMakeLists.txt` via CPM/`ExternalProject`.
- Boost (>= 1.80), TBB (>= 2021) and SUNDIALS (>= 5.4.0) now use a compatible system copy automatically if found, otherwise build the bundled one. Controllable via `build.sh -b/-t/-s <prefix>` (or `-DBOOST_DIR`/`-DTBB_DIR`/`-DSUNDIALS_DIR`) and the `BUILD_BOOST`/`BUILD_TBB`/`BUILD_SUNDIALS` env vars.
- Compiler is selected via `CC`/`CXX`/`FC` and propagated to all dependencies.
- `build.sh` now auto-detects a sensible `-j` thread count and gains `-f` (full, non-interactive build).
- `doxygen-awesome-css` is vendored directly instead of being a submodule.

### Added

- Containers (Debian 13, Fedora 41, Rocky Linux 9, Ubuntu 24.04) and scripts to cross-test the build on other environments.

### Fixed

- Forgotten dependency on CPM source data caused by accidentally `INTERFACE` include directories.
- Attempted assignment to a `const` member inside a copy constructor.

## Version 1.0.1

### Fixed

- There was a bug with TaskFlow internally in deal.ii. Fixed for now by simply disabling taskflow in deal.ii.
- deal.ii changed its interface for dealii::SolutionTransfer. Adapted the corresponding methods.

### Changed

- The FlowingVariables classes are now in separate namespaces. For finite elements, use DiFfRG::FE::FlowingVariables, for pure variable systems use DiFfRG::FlowingVariables.
