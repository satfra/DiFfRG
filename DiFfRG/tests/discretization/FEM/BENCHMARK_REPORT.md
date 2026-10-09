# FEM Assembler Benchmark Report

Historical: measured on the per-point `mesh_loop` assemblers, which the batched assemblers have replaced. For
those see `doc/batched_assembly/RESULTS.md` at the repository root.

## Baseline Results (pre-optimization)

### CG Burgers (1-component)
| Method       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|
| mass        | 190.8 us  | 209.7 us  | 227.5 us  |
| residual    | 214.8 us  | 241.0 us  | 269.5 us  |
| jacobian_mass| 193.1 us | 259.3 us  | 312.8 us  |
| jacobian    | 539.6 us  | 1350.9 us | 2728.4 us |

### CG Constant (3-component)
| Method       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|
| mass        | 227.1 us  | 267.9 us  | 304.6 us  |
| residual    | 258.4 us  | 313.2 us  | 375.2 us  |
| jacobian_mass| 278.6 us | 468.1 us  | 855.3 us  |
| jacobian    | 1496.8 us | 3428.9 us | 6108.5 us |

### DG Constant (1-component)
| Method       | p=0       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|-----------|
| mass        | 188.1 us  | 196.6 us  | 214.9 us  | 221.0 us  |
| residual    | 263.8 us  | 286.2 us  | 327.4 us  | 375.1 us  |
| jacobian_mass| 189.0 us | 231.7 us  | 263.2 us  | 311.7 us  |
| jacobian    | 331.2 us  | 377.6 us  | 499.6 us  | 692.5 us  |

### dDG Constant (1-component)
| Method       | p=0       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|-----------|
| mass        | 182.2 us  | 197.8 us  | 216.4 us  | 229.3 us  |
| residual    | 294.7 us  | 329.5 us  | 384.3 us  | 423.7 us  |
| jacobian_mass| 210.4 us | 227.7 us  | 265.4 us  | 320.0 us  |
| jacobian    | 401.7 us  | 485.9 us  | 717.5 us  | 1120.3 us |

### LDG Constant (1-component, 2 subsystems)
| Method       | p=0       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|-----------|
| mass        | 358.7 us  | 384.1 us  | 396.7 us  | 406.9 us  |
| residual    | 972.5 us  | 1026.5 us | 1161.2 us | 1248.1 us |
| jacobian_mass| 404.6 us | 429.9 us  | 470.2 us  | 535.6 us  |
| jacobian    | 1581.6 us | 1759.3 us | 2067.0 us | 2799.6 us |

## Optimized Results

### CG Burgers (1-component) — after Opts 1-5
| Method       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|
| mass        | 183.8 us  | 207.4 us  | 223.5 us  |
| residual    | 211.1 us  | 243.9 us  | 262.2 us  |
| jacobian_mass| 218.6 us | 218.2 us  | 304.5 us  |
| jacobian    | 339.9 us  | 449.7 us  | 599.0 us  |

### CG Constant (3-component) — after Opts 1-5
| Method       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|
| mass        | 229.0 us  | 262.6 us  | 310.2 us  |
| residual    | 254.6 us  | 302.9 us  | 372.1 us  |
| jacobian_mass| 279.3 us | 464.9 us  | 875.5 us  |
| jacobian    | 928.3 us  | 3211.4 us | 5707.1 us |

### DG Constant (1-component) — after Opts 7-9
| Method       | p=0       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|-----------|
| mass        | 195.9 us  | 210.1 us  | 215.2 us  | 226.2 us  |
| residual    | 261.5 us  | 285.1 us  | 327.1 us  | 357.6 us  |
| jacobian_mass| 221.9 us | 239.0 us  | 268.8 us  | 310.3 us  |
| jacobian    | 342.2 us  | 388.5 us  | 500.3 us  | 690.6 us  |

### dDG Constant (1-component) — after Opts 10-11
| Method       | p=0       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|-----------|
| mass        | 188.4 us  | 201.2 us  | 222.5 us  | 244.8 us  |
| residual    | 299.8 us  | 336.3 us  | 389.1 us  | 442.7 us  |
| jacobian_mass| 219.4 us | 237.6 us  | 268.7 us  | 329.2 us  |
| jacobian    | 401.7 us  | 487.1 us  | 640.9 us  | 831.5 us  |

### LDG Constant (1-component, 2 subsystems) — after Opts 12-13
| Method       | p=0       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|-----------|
| mass        | 369.2 us  | 394.1 us  | 400.7 us  | 413.3 us  |
| residual    | 966.6 us  | 1059.1 us | 1157.4 us | 1285.1 us |
| jacobian_mass| 408.2 us | 438.4 us  | 475.9 us  | 536.8 us  |
| jacobian    | 1588.8 us | 1808.1 us | 2189.0 us | 2837.3 us |

## AD-Optimized Results (after AD-Opts 1-4)

### CG Burgers (1-component) — after AD-Opts 1-4
| Method       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|
| mass        | 183.2 us  | 211.6 us  | 220.3 us  |
| residual    | 214.5 us  | 240.2 us  | 270.6 us  |
| jacobian_mass| 217.7 us | 250.0 us  | 306.5 us  |
| jacobian    | 252.5 us  | 283.1 us  | 319.0 us  |

### CG Constant (3-component) — after AD-Opts 1-4
| Method       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|
| mass        | 228.8 us  | 255.8 us  | 301.5 us  |
| residual    | 256.0 us  | 301.2 us  | 369.7 us  |
| jacobian_mass| 279.2 us | 453.5 us  | 871.9 us  |
| jacobian    | 298.6 us  | 2422.0 us | 4496.9 us |

### DG Constant (1-component) — after AD-Opts 1-4
| Method       | p=0       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|-----------|
| mass        | 194.9 us  | 204.1 us  | 217.2 us  | 226.0 us  |
| residual    | 261.9 us  | 281.7 us  | 319.3 us  | 354.1 us  |
| jacobian_mass| 222.5 us | 238.7 us  | 232.5 us  | 319.6 us  |
| jacobian    | 313.5 us  | 342.4 us  | 442.0 us  | 606.0 us  |

### dDG Constant (1-component) — after AD-Opts 1-4
| Method       | p=0       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|-----------|
| mass        | 188.1 us  | 203.9 us  | 220.9 us  | 238.0 us  |
| residual    | 300.4 us  | 330.2 us  | 375.4 us  | 429.7 us  |
| jacobian_mass| 213.1 us | 233.6 us  | 265.8 us  | 325.0 us  |
| jacobian    | 346.2 us  | 378.8 us  | 462.9 us  | 551.2 us  |

### LDG Constant (1-component, 2 subsystems) — after AD-Opts 1-4
| Method       | p=0       | p=1       | p=3       | p=5       |
|-------------|-----------|-----------|-----------|-----------|
| mass        | 377.7 us  | 375.4 us  | 411.4 us  | 412.6 us  |
| residual    | 969.5 us  | 983.7 us  | 1156.4 us | 1267.4 us |
| jacobian_mass| 410.8 us | 432.8 us  | 473.7 us  | 529.6 us  |
| jacobian    | 1534.1 us | 1716.0 us | 2050.0 us | 2684.2 us |

## Speedup Summary (jacobian method, most impacted)

| Assembler     | p=0   | p=1   | p=3   | p=5   |
|--------------|-------|-------|-------|-------|
| CG Burgers   | -     | 2.14x | 4.77x | 8.55x |
| CG Const3    | -     | 5.01x | 1.42x | 1.36x |
| DG           | 1.06x | 1.10x | 1.13x | 1.14x |
| dDG          | 1.16x | 1.28x | 1.55x | 2.03x |
| LDG          | 1.03x | 1.03x | 1.01x | 1.04x |

## Optimizations Applied

### Assembler-level optimizations
- **Opt 1**: Move `comp[]` to ScratchData (CG) — avoid per-cell allocation
- **Opt 2**: Pre-compute `comp[]` in CG boundary workers
- **Opt 3**: Cache shape values/gradients/hessians before `(i,j)` loop (CG jacobian)
- **Opt 4**: Consolidate 4 `+=` into 1 in CG jacobian inner loop
- **Opt 5**: TBB grain size threshold for CG jacobian (skip TBB for n_dofs*n_dofs < 64)
- **Opt 7**: Move `comp[]` to ScratchData + pre-compute in all DG workers + reserve face_data
- **Opt 8**: (Applied with Opt 7 as part of DG optimization)
- **Opt 10**: Move `comp[]` to ScratchData + pre-compute in all dDG workers + reserve face_data
- **Opt 11**: Cache shape values/grads/hessians + consolidate `+=` + grain size threshold in dDG jacobian
- **Opt 12**: Pre-compute component indices in all LDG workers
- **Opt 13**: Pre-allocate `face_data` in LDG CopyData

### AD-level optimizations (ad.hh)
- **AD-Opt 1**: Remove TBB `parallel_for` for small component counts + eliminate per-chunk `du` copy
- **AD-Opt 2**: Hoist `tuple_cat` outside component loops (build tuple once, not per iteration)
- **AD-Opt 3**: Fused `jacobian_flux_source` methods (compute flux+source jacobians in single seed/unseed pass)
- **AD-Opt 4**: Move `res` arrays outside inner loops in grad/hess methods (avoid repeated stack allocation)

---

## Assembly-schedule calibration (2026-08-21)

Superseded: `assembly_schedule.hh` and the `DIFFRG_BENCH_QUEUE` / `DIFFRG_BENCH_CHUNK` overrides are gone with
`mesh_loop`. Kept as the record of what the schedule was based on: the constants in `assembly_schedule.hh`, which
replaced the `/discretization/mesh_workers` and `/discretization/batch_size` config keys.

### Per-cell cost, measured

| loop | cost per cell |
|---|---|
| CG p=3 cheap loops (`mass`, `jacobian_mass`, and the Burgers `residual`/`jacobian`) | 0.2–0.3 µs |
| KT `residual` with real momentum integrals (ONfiniteT O(N), `x_order` 32, T=0.05) | 2.7 µs |
| KT `jacobian` | 8.9 µs |

### Chunk size

Integral-heavy loops are **indifferent**: sweeping chunk ∈ {1,…,128} at 10/40/150/600/2400 cells
put every value within 5 % of the best, for both `residual` and `jacobian`.

Cheap loops are not: chunk = 1 costs 2–4× the optimum, the plateau is 32–128, and chunk ≥ 512
regresses again — `queue_length * chunk_size` `CopyData` objects are allocated up front, so an
oversized chunk is pure allocation on a mesh that cannot fill it.

Both are reproduced by a single constant, `chunk_work_ns = 8e3`: 8 µs of work buys 32 cheap cells
or 3 integral ones, which are the two measured optima. This is why the schedule takes a per-cell
cost rather than a light/heavy flag — a loop between the two gets a chunk between the two.

### Worker count

Cheap loops on a small mesh are fastest with **one** worker (128 cells: 126 µs at queue 1 vs 219 µs
at queue 16), two at 512 cells, and only past a few thousand does the full budget win.
`worker_work_ns = 64e3` reproduces that progression. Validation against the measured optima: the
integral loops land within 0.5–6.8 % at every mesh size, cheap `mass` within 0–7.5 %.

### End-to-end, ONfiniteT KT (150 cells, 8 threads, 4 interleaved repeats)

Auto picks 1×32 for the cheap loops and 7×3 for the integral ones, against the config's former
hand-tuned 32×8 for everything:

| loop | hand-tuned 32×8 | automatic | gain |
|---|---|---|---|
| residual | 0.180 ms | 0.160 ms | 11 % |
| jacobian | 0.353 ms | 0.297 ms | 16 % |

Assembly counts were identical (87 residuals, 24 Jacobians), as they must be: `mesh_loop`'s copier
is serial and runs in iterator order, so no schedule can change an assembled number.

### Caveat

The 2-D (queue × chunk) grid for the cheap loops was taken while the machine ran an unrelated
28-core job; adjacent grid points differ by up to 2.5×, which is noise. `worker_work_ns` is
therefore provisional — re-run the sweep on an idle machine before treating it as final. The
integral-loop constant, the "chunk = 1 is much worse" result and the end-to-end A/B above are all
robust to that contention.

### Reproducing

The benchmarks honour `DIFFRG_BENCH_QUEUE`, `DIFFRG_BENCH_CHUNK`, `DIFFRG_BENCH_CELLS` and
`DIFFRG_BENCH_FE` (see `DiFfRG/tests/boilerplate/benchmark_sweep.hh`), so a sweep needs no rebuild.

## Assembly iterates only locally-owned cells (2026-08-21)

Every assembly `mesh_loop` takes `locally_owned_cells(dof_handler)` — a
`filter_iterators(..., IteratorFilters::LocallyOwnedCell())` range — rather than the raw
`begin_active(), end()` pair. `is_locally_owned()` is true for every cell of a serial
triangulation, so serial builds are unaffected.

This is what lets the schedule be sized from a plain owned-cell count: a work item now contains
only cells that do work, whatever the partition looks like in `active_cell_index` order.

Two defects closed along the way, both MPI-only:

- `internal::CopyData_I` had no member initializers, and `mesh_loop` assigns the sample
  `CopyData` before its ownership check while WorkStream's copier stage runs over every iterator
  in a chunk regardless of whether the worker did. Every foreign cell therefore reached
  `indicator[c.cell_index] += c.value` with indeterminate values — an out-of-bounds write into a
  `Vector<double>` sized `n_active_cells()`. Filtering keeps the copier away from unwritten
  `CopyData`, and the members are now initialized as well. `HAdaptivity transfers the solution
  identically under distribution` SIGSEGV'd at 2 and 4 ranks before this and passes after.
- deal.II's `dim == 1` refinement-edge branch calls `face_worker` with no ownership guard, unlike
  the equal-level and `dim > 1` branches. On a replicated mesh every rank reached it for every
  hanging-node face and `compress(add)` summed `n_ranks` copies. Affects DG/dDG under
  MPI + 1D + h-adaptivity.

Loops that must keep visiting every cell, and do: the KT topology, face-descriptor and
reconstruction caches (an owned cell reads its neighbours' slots, and the stencil reaches two cells
out) and `eom.hh::solve_potential`, which drives a deliberately serial potential handler.
