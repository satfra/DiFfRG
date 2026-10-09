# Running flow equations across several GPUs

This is the hand-over for the follow-up session that runs DiFfRG on a cluster with 1–4 GPUs and
produces a scaling curve. It describes what was built, how to build and launch it, what to measure,
and — importantly — what a *disappointing* curve would and would not mean.

---

## 1. What was built

`mpirun -n N ./YourApp` now spreads the flow-equation work over `N` ranks (and, with one GPU per
rank, over `N` GPUs). **No model, no generated flow file and no application source had to change.**
There is deliberately no user-facing API for this — no balancer to construct, no distribution to
register, nothing to tune per flow.

| Component | File | Role |
|---|---|---|
| `MapScheduler` | `include/DiFfRG/physics/integration/map_scheduler.hh`, `src/.../map_scheduler.cc` | Decides which rank computes which part of each `map()`, and performs the batched exchange |
| `MapCompletion` | `include/DiFfRG/physics/integration/map_completion.hh` | Lands deferred device results, then drives the exchange |
| `scheduled_map`, `staged_map` | `include/DiFfRG/physics/integration/map_distribution.hh` | The scheduler and staging half of every integrator's `map()`: vacuum and finite-T quadrature, and the lattice sums |
| Rank → GPU affinity | `src/DiFfRG/common/init.cc` | Gives each rank its own device and its share of the CPU threads |
| MPI utilities | `include/DiFfRG/common/mpi.hh`, `src/.../mpi.cc` | `allgatherv`, `agree`, `bcast`, `any_of`, `split_shared`, `abort` |

The previous sketch (`NodeDistribution`, `IntegrationLoadBalancer`,
`AbstractIntegrator::set_node_distribution`) has been deleted.

### 1.1 What gets split, and why that choice

**Only the external coordinate grid of a `map()` — never a quadrature reduction.**

Every individual integral is therefore still summed entirely on one rank, in the original order, so
the distributed result is **bitwise identical to a serial run at any rank count**. This is not
fastidiousness. The flow kernels feed a stiff DAE; `common/tbb.hh:75-85` spells out the consequence:
a last-bit change in one residual changes the accepted step sequence, and from there the entire
trajectory. A scheme that summed partial quadratures across ranks would give a *different physics
answer* for every rank count, and no scaling curve drawn from it would be comparable.

It also makes rebalancing free: moving a partition boundary changes *which* rank computes a grid
point, never *what* it computes.

Splitting applies to external grids of **any** dimension. `SubCoordinates` is a window into the
*linear* index range, which is exactly the shape `part(G, r, j)` hands out, so no per-axis structure
is involved. It used to derive a per-axis box by running `from_linear_index()` on both ends and
subtracting, which is the same set of points only for `dim == 1`; multi-dimensional grids were
therefore assigned whole to one rank. That was free in `QCD_Nf2`, where every `map()` is over a 1D
momentum grid, and expensive everywhere else — in `YangMills/Full` the three `coordinates3D` vertex
flows are both the costliest and the ones that could not be split.

The bitwise guarantee survives the change for the same reason as before, and more directly: the
window's `from_linear_index(s)` returns the **base** multi-index of global point `offset + s`, and
`forward()` is `Base::forward` unchanged, so a rank computes the same `double` for a point that a
serial run computes for it.

### 1.2 CPU backends, and mixed CPU/GPU models

`MapScheduler` is **execution-space agnostic**: `schedule()` runs before any space-specific code, in
all four `map()` overloads (the generic `QuadratureIntegrator`, its `TBB_exec` specialisation, and
the two finite-T mirrors). Generated flow wrappers such as `Integrator_p2_1ang` inherit `map()`
unchanged, so nothing is needed per integrator.

- **A CPU-only build distributes identically.** `GPU_exec` is `Kokkos::DefaultExecutionSpace`, which
  *is* the host space when CUDA is off, so the same flows route through the host branch of
  `map_dist()`. Same slices, same plan, same `Allgatherv`, same bitwise guarantee — only the pinned
  staging disappears, because there is no device to stage from.
- **The target is chosen per execution space, automatically.** The execution space is a compile-time
  property of the integrator, so `map()` passes `map_target<ExecutionSpace>()` into `schedule()`; no
  configuration is involved. That target carries two things: the fill threshold, which sets the split
  width, and the **resource class** — `device` or `host` — which selects the budget the slices are
  charged to. A CUDA-less build resolves both to `host` for free, because there `GPU_exec` *is* the
  host space and the selector keys on `memory_space == CPU_memory`, the same test `map_dist()`
  already uses to pick its staging path.
- **CPU-only MPI within a single node is expected to be a small net loss.** `Init` gives each rank
  the cores its affinity mask actually grants it, so N ranks on one node share the same T cores,
  minus MPI overhead and minus TBB's ability to work-steal across one arena. CPU-only MPI pays off
  *across* nodes, not within one.
- **A model mixing `GPU_exec` and `TBB_exec` integrators uses both at once, and is balanced on
  each.** A rank owns two independent resources — its GPU and its share of the node's cores — and
  they overlap in time: inside a `DeferredMaps` scope the device path launches asynchronously and
  returns, so a host map issued after it executes while the device is still working. This is the one
  case where the "97.5 % serialised" measurement in section 7 does not describe the run.

  `MapScheduler` models that with **one load budget per resource**. Both the split width and the
  owner choice are made against the budget of the space the calling integrator actually uses, so a
  rank that has just taken a large TBB flow is still the natural home for the next GPU flow. The
  budgets are never summed and never converted into each other: there is no exchange rate at which a
  device evaluation equals a host one, and assuming one is exactly what a single shared budget did.
  With each resource levelled independently, per-rank wall time is `max(device_i, host_i)` with both
  terms flat in `i` — which is the optimum available under this structure.

  Two limits remain, and both are outside the scheduler. Within one resource the weight per
  evaluation is still 1, so flows with equal `S` but different per-evaluation cost still mis-weigh
  (see §8). And the overlap only exists inside a deferral scope — a host map issued with deferral off
  flushes, and therefore synchronises every rank, before the device work behind it can catch up.

  **Call order inside a deferral scope does not matter.** It used to: a host `map()` is
  `tbb::parallel_for` or a host `Kokkos::parallel_for`, both of which return only once the work is
  done, so it blocks the *host thread* — not the GPU, which keeps running whatever was already
  launched, but the launch of any device map issued *after* it. Writing `GPU, CPU, GPU` therefore
  drained the stream whenever the host map outlasted the device work in flight, while `GPU, GPU, CPU`
  did not. A performance cliff that depends on statement order is exactly the kind of thing a user
  should not have to know, so it was removed rather than documented: inside a `DeferredMaps` scope a
  host map is **queued**, not run, and `flush()` drains the queue *before* it fences. Any mixed block
  therefore executes as "launch every device map, run the host maps while they are in flight, fence",
  however it was written. Wall time is the larger of the two totals, never their sum.

  The queue is `MapCompletion::record_work` (`map_completion.hh`), fed by `run_or_queue*()` in the
  four `map()` overloads. Three properties are worth knowing:

  - It is **compiled out** when there is no device backend (`internal::has_device_backend`), so a
    CUDA-less build keeps running host maps inline — there would be nothing to overlap with.
  - Jobs run **sequentially, in call order**, each still internally TBB-parallel. So the summation
    order inside every integral is untouched and results stay bitwise identical; and the per-integrator
    `m_dest_device` scratch is written and drained before the next job touches it.
  - On the exception path `~DeferredMaps` **drops** queued jobs without running them, exactly as it
    already drops unlanded copies — the destinations may be gone.

  `DIFFRG_MAP_QUANTUM` replaces the *threshold* for every space, but never the resource class, so
  pinning it does not collapse the two budgets back into one. Nothing in `QCD_Nf2` exercises any of
  this — every flow there is `GPU_exec` — but the mixed path is covered by
  `tests/physics/integration/map_scheduler.cc`.

### 1.3 How the split width is chosen

Per `map()` call, with `G` external grid points and quadrature volume `Q = prod(grid_size)`:

```
S = G * Q                                    kernel evaluations in this map
r = clamp(S / quantum, 1, min(n_ranks, G))   how many ranks to spread it over
owners = the r least-loaded ranks so far in this batch
```

`quantum` is a **fill threshold** — the work below which one rank's compute resource cannot be kept
busy, so that splitting further buys nothing and costs a launch plus a share of a gather. It is *not*
a launch-overhead threshold; deriving it from launch cost (order 10³ evaluations) would oversplit by
two orders of magnitude.

**It is derived from the hardware, not hardcoded**, via `Kokkos::…::concurrency()`:

| execution space | threshold | why |
|---|---|---|
| device | `concurrency()` | it counts *resident threads*, and each thread evaluates the kernel once, so the device is full at exactly that many evaluations |
| host | `concurrency() × 1024` | it counts *workers*, each looping over many evaluations, so the threshold is workers × a grain big enough to bury one `parallel_for` spawn (order 10 µs) |

The device rule is checkable and checks out: the value tuned empirically on the development GPU was
`5e4`, and that GPU reports `concurrency() == 55 296` (36 SMs × 1536). **The host grain of 1024 is a
judgement, not a measurement** — no CPU-backend flow run has been profiled. If a CPU run splits
badly, that constant is the first suspect, and `DIFFRG_MAP_QUANTUM` overrides both.

This is not cosmetic. An A100 has `concurrency() == 221 184`, four times the development GPU, so a
hardcoded `5e4` would have **over-split by 4×** on exactly the cluster this feature exists for: each
`4D_3ang` flow would go to 8 ranks holding 55 296 evaluations each, a quarter of what it takes to
fill an A100. With the derived threshold it goes to 2 ranks at full occupancy while the least-loaded
packing keeps the other ranks busy on different flows.

The consequence that matters for load balance: **a cheap flow is assigned whole** to the
least-loaded rank rather than chopped into slivers, so a batch of many small flows spreads *across*
ranks instead of every rank doing a sliver of every flow.

For `QCD_Nf2/no_mesons_AAqbq18` at `x_order=32, cos1=cos2=phi=6`, on the development GPU
(threshold 55 296) and on an A100 (threshold 221 184):

| class | count | Q | S = 64·Q | r on 4070 Laptop | r on A100 |
|---|---|---|---|---|---|
| `4D_3ang` (18 `ZAAqbq` + `ZA4`) | 19 | 6912 | 442 368 | 8 | 2 |
| `4D_2ang` (`ZA3`, `ZAcbc`, `ZAqbq1/4/7`) | 5 | 1152 | 73 728 | 1 | 1 |
| `p2_1ang` (`ZA`, `Zc`, `Zq`) | 3 | 192 | 12 288 | 1 | 1 |

Note what this means for the cluster run: at N ≤ 4 the split width is clamped by the rank count
anyway (`r = min(…, n_ranks)`), so 1–4 GPUs behave identically under either threshold. The
distinction only starts to matter at N > 4.

**The fill threshold is the right one only for a map inside a batch.** Holding back below it is an
argument about *opportunity cost*: the ranks this map leaves out are about to be taken by the other
maps in the same `DeferredMaps` scope, so slivering this one buys nothing and costs a launch plus a
gather slice. Outside such a scope that premise is false. `map()` flushes on return, so the map *is*
the batch — there are no other maps, and every rank that owns no slice sits in the `Allgatherv`
doing nothing until the owners finish. The only reason left not to split is that a slice must still
cover the launch that computes it.

So `schedule()` measures against `internal::launch_threshold` (1024 evaluations, the launch-cost
constant this section rejects for the batched case) whenever no deferral scope is open:

```
quantum = pinned override, if any
        : batched   -> target.fill_threshold
        : unbatched -> min(target.fill_threshold, launch_threshold)
```

`MapScheduler` learns which case it is from `MapCompletion::set_deferral()`, which forwards to
`set_batched()`. That keeps the dependency one-way (`map_completion.hh` includes
`map_scheduler.hh`, not the reverse) and keeps the plan a pure function of the call sequence, since
`DeferredMaps` is constructed by replicated model code and the flag therefore holds the same value
on every rank at every `schedule()` call. `DIFFRG_MAP_QUANTUM` wins over both: a user who pins a
width chose it.

This matters most for exactly the configuration section 1.2 warns is unmeasured. No model in
`Examples/` or `Tutorials/` opens a `DeferredMaps` scope, so the unbatched case is the one they all
take today. On a host backend with 8 workers (fill threshold `8 × 1024 = 8192`), a `p2_1ang` flow at
`S = 12 288` went from `r = 1` to `r = 12`.

### 1.4 How results come back

`MapCompletion::flush()` fences, lands the pinned staging buffers, and then performs **one**
`MPI_Allgatherv` covering every map in the open batch (~12.8 kB for the 25-map block). One collective
per `DeferredMaps` scope, not one per flow.

The rule that keeps this from deadlocking, and the single most important invariant in the design:

> The collective is driven by the **shared plan**, never by rank-local state. Every rank registers a
> plan entry for every `map()`, *including* the ranks that own nothing (`count = 0`). `flush()`
> decides "no-op or collective" purely from whether a plan is open.

Rank-local predicates (`MapCompletion::has_pending()`, "is my pending list empty") differ between
owners and non-owners, so any decision based on them would have some ranks enter a collective the
others skip — a silent hang. Before the exchange, the ranks agree on a hash of the plan
(`MPI::agree`, one tiny collective); a mismatch calls `MPI_Abort` with a diagnosis rather than
hanging in a malformed `Allgatherv`.

That agreement is an `MPI_Allreduce`, and it is *tiny in bytes, not in latency*: at several hundred
ranks it is comparable to the `Allgatherv` it guards, and with no deferral scope open it is paid
once per flow per residual evaluation. It therefore runs on the first 8 batches and every 64th
batch after that (`DIFFRG_MAP_VERIFY_EVERY`, `1` to restore it on every batch).

Be clear about what that gives up. The head window still catches immediately the failure that
actually happens — a model whose flow sequence is wrong from the start. A divergence appearing
later is only diagnosed at the next verified batch, and until then a malformed gather runs instead.
And because the same bug class can desynchronise the batch counter itself (a rank that skips a flow
also skips its batch), one rank can reach the `Allreduce` while another goes straight to the
gather — a hang, which is exactly what the check exists to prevent. So skipping degrades the
guarantee back to what it was before the check existed; it never makes things worse than having no
check at all. **When debugging a hang or a wrong distributed result, set `DIFFRG_MAP_VERIFY_EVERY=1`
first.**

`MPI_Allgatherv` rather than `MPI_Allreduce(SUM)` over zero-filled buffers: the latter moves ~25×
more bytes and is **not bit-safe**, since `(-0.0) + 0.0 == +0.0` destroys a signed zero.

There is deliberately **no** `MPI_Iallgatherv`. Every gather result is consumed on the very next host
statement (`dtZA.update()`, `zqF[0]`, the AAqbq basis rotation), and an explicit ABM stepper cannot
start the next RHS, so there is nothing to overlap with.

### 1.5 Rank → CPU and GPU affinity

`Init` now, after `MPI_Init`:

1. builds a node-local communicator (`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)`),
2. works out this rank's CPU thread budget from the **affinity masks** (below),
3. picks `device = local_rank % n_visible_devices`,
4. initialises Kokkos itself with that device — replicating everything deal.II's
   `ensure_kokkos_initialized()` would have done, **including** setting
   `dealii::internal::dealii_initialized_kokkos = true` and registering `std::atexit(Kokkos::finalize)`.
   Both of those live inside its `if (!Kokkos::is_initialized())` branch; skipping them reintroduces
   the exit segfault documented at `init.cc:59-92`. This path is taken only when there is more than
   one node-local rank; a single-rank run still goes through deal.II unchanged.

#### CPU threads per rank

The obvious rule — take the machine's core count and divide by the number of node-local ranks — is
wrong on the systems that matter, and wrong in *both* directions. Measured on a 32-CPU machine
pinned to 8 CPUs with `taskset`:

| what it reports | unpinned | pinned to 8 of 32 |
|---|---|---|
| `std::thread::hardware_concurrency()` — what deal.II's `MultithreadInfo::n_cores()` calls | 32 | **32** |
| `sysconf(_SC_NPROCESSORS_ONLN)` | 32 | **32** |
| `CPU_COUNT(sched_getaffinity)` | 32 | 8 |
| TBB `default_concurrency()` | 32 | 8 |

So:

- **`hardware_concurrency()` ignores the affinity mask.** A rank pinned to its own cores does not
  learn it from there. Split a 128-core node into two ranks and each would deduce 128 — exactly the
  failure you would expect.
- **But DiFfRG's auto path already reads TBB's value, which does see the mask.** So under Slurm
  `--cpus-per-task=64` each rank's limit is *already* 64 — and dividing that again by the node-local
  rank count gives **32**, halving it a second time. That is the more dangerous bug, because it only
  fires on a correctly configured cluster.

The divisor is therefore not "how many ranks are on this node" but **"how many node-local ranks are
competing for *my* CPUs"**, answered by intersecting the affinity masks — the same shape of test as
the GPU bus-id comparison below, and for the same reason. Disjoint masks ⇒ the launcher already
partitioned the node, divisor 1. Identical masks ⇒ the ranks share it, divisor = the number that
overlap. `Init` prints which case it took.

**The knobs, in precedence order.** Highest first; the first one that says anything wins outright,
and DiFfRG prints a loud warning on stderr whenever several of them disagree.

| # | knob | effect |
|---|---|---|
| 1 | `DiFfRG_NUM_THREADS` (or `DIFFRG_NUM_THREADS`) | Taken **verbatim, per rank**. The escape hatch: it beats even a cluster allocation, on the grounds that someone who sets it means it. |
| 2 | the launcher's allocation — `srun --cpus-per-task=N`, `mpirun --bind-to`, `taskset`, a cgroup | Taken verbatim, per rank. The **CPU affinity mask is authoritative** and is checked first: it is the only thing that describes the CPUs this process may actually run on, and it covers every launcher rather than just Slurm. It counts only when it is a *strict subset* of the machine — an unrestricted mask is what a workstation looks like, not an allocation. Where the job did not pin its tasks, `SLURM_CPUS_PER_TASK` is used, then `SLURM_JOB_CPUS_PER_NODE` / `SLURM_CPUS_ON_NODE` divided by `SLURM_NTASKS_PER_NODE`. |
| 3 | `/discretization/threads` | Taken **verbatim, per rank**, with no division. Silently dividing it would make the setting mean "threads per node", which is neither what it says nor what it means in a serial run. |
| 4 | `OMP_NUM_THREADS`, `DEAL_II_NUM_THREADS`, `KOKKOS_NUM_THREADS` | Thread counts meant for somebody else, honoured only when nothing above said anything. |
| 5 | nothing set, no pinning | The CPUs this rank can see, split evenly among the node-local ranks that share them. |

`DEAL_II_NUM_THREADS` needs one note. `dealii::MultithreadInfo::set_thread_limit()` takes the
minimum with it on *every* call, so left alone it would silently cap tiers 1–3 from below and make
the ordering above a lie. Where that would happen, `Init` removes it from the environment before the
first `set_thread_limit()` call and says so in the warning.

The resolved number is published as `DiFfRG::n_threads()` (`DiFfRG/common/threads.hh`), which is
what the map scheduler's host/device split sizes itself against. Read
it from there rather than from `dealii::MultithreadInfo::n_threads()`: the latter is a mutable
static that any `set_thread_limit()` call rewrites, so it is not necessarily the number DiFfRG
resolved. The two are kept in agreement by construction.

There is exactly **one CPU thread pool**, TBB's. Kokkos' host execution space is `Kokkos::Serial`
(build option `KOKKOS_THREADS`, default `OFF`), so nothing spins beside the TBB arena — see
`CMakeLists.txt` for why that matters even when no DiFfRG code uses the Kokkos host backend.

#### GPUs per rank

**Sharing is detected by PCI bus id, not by counting.** Under Slurm `--gpus-per-task=1` every rank
sees exactly one device and `local_rank % n_devices == 0` is the *correct* answer; a count-based
check would report sharing on every well-configured cluster and send you chasing a phantom. The
node-local ranks gather their `cudaDeviceGetPCIBusId` strings and look for duplicates.

On a genuine duplicate you get a **loud warning on stderr from every affected rank, and the run
continues** — a shared GPU is slower, not wrong, and a scaling study has to be able to measure that
case:

```
================================ WARNING ================================
 GPU 0000:65:00.0 is shared by 2 node-local ranks (0, 1).
 The run is still correct, but those ranks time-slice the device and each
 costs several hundred MB of extra context, so it will NOT scale.
 Launch with one GPU per rank (e.g. Slurm --gpus-per-task=1), or set
 CUDA_VISIBLE_DEVICES per rank.
=========================================================================
```

Timestepper progress is stdout-only, so this deliberately goes to stderr — check both streams, or
merge them, if your launcher separates them.

### 1.6 Finite-T integrators now get host run-ahead too

`QuadratureIntegrator_fT` never had the pinned-staging path that `QuadratureIntegrator` was given
for exactly this reason, so its device path copied straight from the device into `dest`. `dest` is
ordinary *pageable* caller memory — a `dealii::Vector` element range — and a device-to-pageable
`cudaMemcpyAsync` is not actually asynchronous: the driver has to stage it, so the call blocks until
the copy, and therefore the kernels feeding it, has completed.

The consequence was **zero host run-ahead at finite temperature**: the GPU drained after every
single `map()`, and all host preparation for the next flow was exposed as device idle time. On the
vacuum integrator that was measured at `cudaMemcpyAsync` totalling 15.933 s against 15.966 s of
kernel time — the host sat inside the copy for essentially the whole run. It cost a few percent on a
slow part and over half the wall clock on a fast one, which is why `MapCompletion` exists; the
finite-T mirror simply never received it.

It has it now: page-locked staging, `MapCompletion::record`, the same one-staging-buffer-per-
integrator guard that lands an outstanding result before a second `map()` from the same integrator
overwrites it, and the same participation in the batched `Allgatherv`. Nothing about the API or the
numbers changes — `tests/physics/integration/finiteT/quadrature_integrator_fT.cc` pins that the
deferred results are bitwise what the undeferred path produced, and that they really are still in
staging while the scope is open.

**This matters independently of MPI.** A single-GPU finite-T run gets the speed-up too; it is not
something the rank count unlocks. Expect the gain to be largest exactly where the kernels are
fastest relative to the host, i.e. on the cluster part rather than on a laptop GPU.

---

## 2. Building with MPI on the cluster

MPI is **a single switch for the whole superbuild**. deal.II declares the MPI types (real ones with
MPI, dummies in `namespace dealii` without), SUNDIALS and PETSc have their own switches, and a mixed
configuration miscompiles rather than failing cleanly. There is now a configure-time guard for the
deal.II/DiFfRG mismatch and a `#error` in `common/mpi.hh` as a backstop, but the fix is always the
same: turn MPI on for the superbuild.

```bash
bash install_diffrg.sh --mode source --mpi   # installer route
# or, configuring the superbuild directly:
cmake <src> -DMPI=ON -DCMAKE_INSTALL_PREFIX=<prefix>
```

There is also a `release-mpi` CMake preset in `DiFfRG/CMakePresets.json`.

**Build-parallelism rules for this workflow** (learned the hard way on a 30 GB machine):

- Bound the parallelism explicitly: `-j6`. Never bare `-j` or `-j$(nproc)`.
- To rebuild only the library and tests, build the **inner** target
  `build/DiFfRG/src/DiFfRG-build`, not the superbuild target — the latter rebuilds deal.II.
- **A header-layout change requires a clean rebuild of the application.** This increment changed the
  layout of `AbstractIntegrator` and deleted `distribution.hh`, so any app object file from before
  it must be rebuilt. A stale one gives a silent `SIGSEGV` that looks exactly like a logic bug.
  Also delete any stale `nsys` `.sqlite` next to the binary.

---

## 3. Launching

```bash
mpirun -n 4 ./QCD parameter.toml
# Slurm, one GPU per rank:
srun --ntasks=4 --gpus-per-task=1 ./QCD parameter.toml
```

Confirm each rank actually got a distinct device — by bus id, not by count:

```bash
mpirun -n 4 bash -c 'echo "rank $OMPI_COMM_WORLD_RANK sees: $CUDA_VISIBLE_DEVICES"; nvidia-smi -L'
```

and check that no sharing warning appeared on stderr.

Two knobs, both optional:

| knob | where | meaning |
|---|---|---|
| `DIFFRG_MAP_QUANTUM` | environment | overrides the split threshold for *every* execution space, replacing the hardware-derived defaults (wins over the parameter file). It does not merge the per-resource budgets — see §1.2 |
| `DIFFRG_MAP_VERBOSE=1` | environment | prints the plan once: per-flow resource and split width, owning ranks, and the per-rank load of each resource separately |
| `DIFFRG_MAP_VERIFY_EVERY` | environment | how often the ranks agree on the plan hash before an exchange. Default 64 (plus the first 8 batches unconditionally). **Set to `1` first when debugging a hang or a wrong distributed result** — it restores immediate, diagnosable detection of a schedule divergence |
| `/integration/map_quantum` | parameter file | same as above, lower precedence |
| `/integration/map_verbose` | parameter file | same as above |

Run once with `DIFFRG_MAP_VERBOSE=1` before any timing and keep the output: the plan is what any
imbalance in the curve has to be explained against.

---

## 4. The gate: bitwise equality, before any timing

**A scaling number from a run that fails this is meaningless.** Run the same short configuration at
`-n 1`, `-n 2`, `-n 3` (deliberately not a divisor of 64) and `-n 4`, and compare the outputs
*exactly*, on raw `double` bits — not to a tolerance:

```bash
for n in 1 2 3 4; do
  mpirun -n $n ./QCD short.toml            # writes output_n$n.h5
done
# then compare the HDF5 datasets bit-for-bit, e.g. with h5diff -d 0, or numpy .tobytes()
```

If `-n 1` and `-n 4` differ in the last bits, stop and investigate before measuring anything: the
most likely cause is that the Kokkos `AUTO` team size in the phase-2 reduction
(`quadrature_integrator.hh:305`) resolved differently at a different `league_size`. That is a
correctness question, not a performance one.

The library-level part of this contract is already covered by tests:

- `tests/physics/integration/map_scheduler.cc` — simulates 1…16 ranks in a single process and checks
  that the slices tile the grid exactly, that the split width follows the cost score, that cheap
  flows spread rather than fragment, and that a batch exchange reassembles the full result bitwise
  on every rank. Runs on any build, MPI or not.
- `tests/physics/integration/distributed_quadrature_integrator.cc` — compares `map()` against
  `map_dist()` bitwise, and is registered under `mpirun` at 2, 3 and 4 ranks when the build has MPI
  (`setup_mpi_test`, `MPI_TEST_RANKS`).
- `tests/physics/integration/lattice/distributed_lattice_integrator.cc` — the same check for
  `IntegratorLat<1..4>` on a 2D external grid, also inside a `DeferredMaps` scope; registered under
  `mpirun` the same way.

---

## 5. The measurement protocol

1. **Stop every background run first.** Any build, sweep or leftover job contaminates the numbers —
   this has produced misleading benchmarks before, and it looks like a machine problem rather than a
   self-inflicted one.
2. Use a short, fixed configuration: `/tuning/m2A/tune = false`, `/tuning/STI/tune = false`,
   `/timestepping/final_time` small enough for a few dozen ABM steps at `dt = 0.05`. Keep it
   identical across all rank counts.
3. Time **the timestepping loop**, not the process: process wall time includes MPI startup, Kokkos
   init and HDF5 setup, which do not scale and are not what is being measured. The assembler's own
   `average_time_residual_assembly()` × `num_residuals()` is the cleanest quantity; failing that,
   time from the first to the last output frame.
4. Repeat each rank count at least twice and report the *minimum*, not the mean — GPU clock ramp
   makes first runs slow and there is no ramp-down noise to average away.
5. Record, per rank count: wall time of the timed region, number of RHS evaluations (they must be
   identical across rank counts — if they are not, the bitwise gate failed), and the plan dump.

---

## 6. What to expect, and what would be surprising

### The ceiling is GPU-side, not communication-side

Two things do **not** scale, and they set the ceiling:

- **The η loop** (`model.hh:279-310`). Up to 50 iterations of 3 one-angle maps, each only
  64 × 192 = 12 288 kernel evaluations. Those do not fill a single GPU today, so the η loop **cannot
  be accelerated by more GPUs under any split**. It is a hard Amdahl floor of roughly
  `n_iter × (3 launches + 2 global fences + 2 spline uploads)`.
- **The phase-2 team reduction** (`quadrature_integrator.hh:304-306`).
  `TeamPolicy(space, integral_view.size(), AUTO, 32)` has `league_size == G_local`, so it shrinks
  64 → 16 → 8 → 4 as ranks grow. At 64 teams it is *already* below an A100's 108 SMs and
  latency-bound, so its wall time is roughly **constant in N** while phase 1 shrinks linearly. Its
  share `f₂` is documented as 0.3–4 %, which caps per-flow speedup at `1/(f₂ + (1−f₂)/N)`: at
  `f₂ = 4 %` that is 3.6× at N=4 but only 10× at N=16.

**Both of these were measured before any MPI code ran, and both turned out to be much smaller than
feared: the η loop is 0.24 % of wall time and the phase-2 reduction 0.16 % of kernel time.** See
**section 7** for the numbers and the resulting speedup table. Read them before interpreting the
curve; without them the curve cannot be interpreted at all.

### Things that are *not* the problem

- **Collective latency.** ~1 collective per main block plus 2 per η iteration, i.e. ~21–61 per RHS at
  realistic `n_iter`. At 5–40 µs each plus straggler skew that is ~5–15 ms against a multi-second
  RHS. If someone proposes non-blocking collectives to fix the curve, the arithmetic says no.
- **Redundant host work.** Every rank runs the whole host side, dominated by 29 `Interp::update()`
  at ~25 µs each, i.e. ~1.8 ms/RHS. Under 1 % even at N=16.
- **The position cache.** `SubCoordinates` *does* have a `to_string()`, so the MPI path keeps the
  cached `forward()` positions. (A comment in `quadrature_integrator.hh` used to claim otherwise;
  it has been corrected.) A schedule that moved a slice boundary would re-key and re-run a ~5 µs
  fill — the schedule is stable across batches, so this does not happen.

### If the curve is worse than section 7 predicts, suspect, in order

1. **GPU sharing.** Check stderr for the warning and confirm distinct bus ids.
2. **Grid quantization.** 64 points over N: exactly 0 % waste at N = 2, 4, 8, 16; ≤ 12.5 % elsewhere,
   with N = 7, 9 and 12 the awkward ones.
3. **Unequal flows within the `4D_3ang` class**, but *only* at N > 8 — see the end of section 7.
   This is the most likely explanation for a disappointing 16-GPU point and cannot explain a
   disappointing 2-, 3- or 4-GPU one.
4. **`quantum` mis-sized for this GPU.** Sweep `DIFFRG_MAP_QUANTUM` over, say, 1e4 … 2e5 and watch
   the plan dump — too small fragments the big flows past the fill threshold, too large leaves ranks
   idle.
5. **`n_iter` in the η loop.** It is data-dependent (≈ 14.7 per RHS in the Step-0 run); if it grew
   substantially, the serial fraction grew with it — but from a base of 0.24 % it would have to grow
   enormously to matter.

### The lever for scaling past ~8 GPUs

`p_grid_size` (`model.hh:80`) is `static constexpr uint p_grid_size = 64`. It is the external grid
being split, so it directly bounds the usable rank count — and raising it is physically desirable
anyway. It is compile-time, so raising it means rebuilding the app (see the clean-rebuild note in
section 2).

---

## 7. Step-0 measurements

Measured on the development workstation before any MPI code ran: one **RTX 4070 Laptop GPU (36 SMs,
sm_89)**, `nsys profile -t cuda` over a serial run of `no_mesons_AAqbq18` with tuning off,
`final_time = 0.5`, `dt = 0.05`. 34 RHS evaluations, 4735 kernel launches, 142.0 s of GPU-activity
span.

### How the per-kernel numbers were obtained

Straight from the nsys CUDA activity trace: each row of `CUPTI_ACTIVITY_KIND_KERNEL` carries a
device-side `start`/`end`, and every flow is a distinct template instantiation, so the demangled
kernel name identifies the flow exactly (`DiFfRG::<Flow>_kernel<...>`). Group by name, sum
`end - start`. Phase 1 and phase 2 are separated by whether the name contains
`KokkosNDLambdaWrapper`.

Summing durations is only legitimate if kernels do not overlap, so that was checked rather than
assumed:

| | |
|---|---|
| distinct CUDA streams | 9 |
| GPU-busy time with exactly **one** kernel resident | **97.5 %** |
| sum of kernel durations vs. union of intervals | 144.51 s vs 140.90 s, i.e. **2.56 %** inflation |

So despite nine streams the flows effectively execute one at a time — each phase-1 launch is 3456
blocks × 128 threads against 36 SMs, i.e. ~96 blocks per SM, which leaves nothing for a neighbour to
overlap with. The per-flow numbers are therefore additive to within ~2.5 % and are already
effectively isolated measurements. Per-launch spread within one flow is ~1.3× min-to-max over 34
launches (clock state), so quote the median, not a single launch.

`S = G · Q` is exactly the launched thread count: 64 × 6912 = 442 368 = 3456 × 128. That is the
sense in which the scheduler's score is exact — it is exact about *work issued*, and says nothing
about work *per thread*, which is where the 6–8× spread below comes from.

| quantity | value | share |
|---|---|---|
| GPU busy (union of kernel intervals) | 140.90 s | **99.25 %** of the span |
| main flow block, wall | 140.96 s | 99.29 % |
| **η loop, wall** | **0.341 s** | **0.24 %** |
| host-only gaps between GPU segments | 0.661 s | 0.47 % |
| **phase-2 team reduction, kernel time** | **0.226 s** | **0.16 %** of 144.5 s kernel time |
| η iterations per RHS | 501 / 34 | ≈ 14.7 |

**Both feared ceilings are far smaller than projected, and neither is worth acting on.**

- The η loop was expected to be a hard Amdahl floor, possibly over 10 % — it is **0.24 %**. There is
  no reason to loosen `eta_tol`, raise `eta_relax` or Anderson-accelerate the fixed point for
  scaling reasons.
- The phase-2 reduction was cited at 0.3–4 %; measured **0.16 %**, at or below the bottom of that
  range. Restructuring the league (`G_local × n_chunks`) is not warranted — and would have changed
  the reduction tree and therefore the bitwise baseline.

Non-scaling total: `0.341 + 0.661 + 0.226 ≈ 1.23 s` against 141.96 s, i.e. **f ≈ 0.87 %**. With
`T(N) = 140.7/N + 1.23`:

| N | predicted speedup | predicted efficiency |
|---|---|---|
| 2 | 1.98× | 99 % |
| 4 | 3.90× | 97 % |
| 8 | 7.54× | 94 % |
| 16 | 14.2× | 89 % |

Collective overhead is not in this model because it does not matter: ~21–61 collectives per RHS at
5–40 µs is 7–83 ms over the *entire* 142 s run.

**These fractions are hardware-specific, and they get worse on a faster GPU.** The η loop costs
0.68 ms per iteration, of which only ~0.41 ms is kernel time; the remaining ~0.27 ms is launch and
fence latency, which does not shrink when the device gets faster. The 0.661 s of host-only gaps is
likewise invariant. Scaling the *kernel* time down by 6× (roughly a 4070 Laptop → A100 step) while
holding the invariant terms fixed gives `f ≈ 3.6 %` rather than 0.87 %, and an N = 16 ceiling near
**10×** rather than 14×. The ranking of the terms does not change — collectives still do not matter
— but on a fast cluster GPU the dominant non-scaling term stops being the η loop and becomes the
per-RHS host work behind those 0.661 s. If the cluster curve flattens earlier than the table above,
re-measure the three invariant terms on the cluster's own GPU before looking for a bug.

**The one real imbalance risk, and it only appears at N > 8.** The 19 `4D_3ang` flows all score
`S = 442 368`, so the scheduler treats them as equally expensive — but they are not. Measured
per-flow kernel time:

- `ZAAqbq1`–`ZAAqbq15`: 8.1–9.4 s each (5.6–6.5 % of GPU time each)
- `ZAAqbq16`, `ZAAqbq17`: ~2.0 s each
- `ZAAqbq18`: 1.6 s
- `ZA4`: 1.2 s

i.e. a ~6–8× spread within one cost class. At N ≤ 8 this is harmless: `r = min(8, N)` means *every*
rank owns an equal slice of *every* big flow, so the per-rank load is identical whatever the flows
actually cost. At N = 16, `r` is still 8, so 19 × 8 = 152 slices are packed onto 16 ranks by a score
that cannot tell a 9.4 s flow from a 1.6 s one — and imbalance of tens of percent becomes possible.
**If a 16-GPU run underperforms the table above, this is the first thing to check** (dump the plan
with `DIFFRG_MAP_VERBOSE=1` and weigh the flows by the kernel times above). The fix is the
calibration listed as open in section 8, or simply raising `p_grid_size` so that `r` can exceed 8.

---

## 8. Known-open items

These were consciously left out of this increment:

- **FE work is not distributed — every rank runs the whole of it, redundantly.** This is worth
  stating precisely, because "not distributed" is easy to misread as "one rank does it". It is the
  opposite: under SPMD all ranks execute the same program, so all of them assemble the entire mesh.
  No FE assembler calls `map()` (verified by grep over `discretization/`); they reach flows through
  the scalar `get()` path, which the scheduler never sees. All ranks therefore compute the same FE
  residual from the same input and agree bitwise, which is why no gather is needed and why this is
  *correct* — just not *fast*.

  Two consequences for anyone running an FE + variables model, as opposed to the pure-variables
  `QCD_Nf2`:

  1. **FE assembly is a pure Amdahl term.** Its cost is constant in N while the flow cost falls as
     1/N, so it caps the speedup at `1/f_FE`.
  2. **Worse: with several ranks per node it actively grows.** FE assembly is TBB-parallel over
     cells, and `Init` divides the thread budget by the node-local rank count — so N ranks on one
     node each do the *full* mesh with T/N threads, and FE wall time rises roughly linearly in N.
     Total time then behaves like `F·N + S/N`, which has a minimum at `N ≈ sqrt(S/F)` and gets worse
     beyond it. One rank *per node* avoids this entirely (the thread division only triggers for
     node-local ranks > 1); several ranks per node does not. Until FE is distributed, prefer one rank
     per node for any model with FE functions, and measure before assuming more ranks help.

  A related rule while this stands: **never call `map()` during assembly** -- from a model's `flux`,
  `source` or `evaluate_batch`. `map()` is collective, while each rank assembles only its own cells,
  so the per-rank plans would diverge. The assemblers open a `NoMapsHere` scope around every model
  evaluation, and a `map()` inside it aborts with a message instead of hanging. Use `map_points()`
  there: it is rank-local and never touches the scheduler.

  The SPMD choice was made partly so that distributing FE later is an orthogonal addition rather than
  a rewrite: deal.II is SPMD, so the mesh partitioning it already supports drops into this model.
- **Cost calibration is not implemented.** `w_kernel` is fixed at 1, i.e. the score is exactly
  `G · Q`. The Step-0 profile shows this is *exactly* right for N ≤ 8 and *wrong by up to 6–8×
  per flow* at N > 8 — see the end of section 7 for why the first statement holds despite the
  second. If a 16-GPU run needs it, the cheapest version is: run one calibration batch with
  `r = 1, owners = all ranks` so every rank times the *whole* flow rather than its own slice, then
  one `Allreduce` yields both `w_kernel` and the per-rank speed ratios (the latter is what a
  heterogeneous cluster would additionally need). Keep calibration RHS evaluations out of any timed
  region.

  An earlier draft of this document warned that isolated per-flow timings overstate a block that in
  production overlaps on several streams. **That is measurably false for this workload** — see the
  concurrency table in section 7: 97.5 % of GPU-busy time has exactly one kernel resident, because a
  single phase-1 launch already saturates the device. Isolated timings and in-situ timings agree to
  ~2.5 % here, so the measured per-flow times may be used directly as weights. Re-check the
  concurrency table if the truncation, the quadrature orders, or the GPU change enough that a launch
  no longer fills the device.
- **The host fill grain (1024 evaluations per worker) is unmeasured.** The device threshold is
  `concurrency()` and is corroborated by the value tuned empirically on the development GPU; its host
  counterpart is `concurrency() × 1024`, where the 1024 comes from reasoning about `parallel_for`
  spawn cost rather than from a profile. No CPU-backend flow run has been measured. Anyone running a
  CPU or mixed backend at scale should profile one and correct `internal::host_grain`.
- **The application must not branch on rank-local data.** `QCD.cc` reads control-flow scalars back
  out of the HDF5 file that only rank 0 writes, at four sites: `:46` and `:84`
  (`time_of_divergence`), `:52` (`m2A`, `Zc`), `:57` (`sti_deviation_or_nan`). Those values drive the
  `tune_m2A` / `tune_STI` bisection, and each probe is a whole run — so if ranks disagree by one
  float about convergence they execute a *different number of runs*, and the first collective of the
  extra run hangs. **A barrier does not fix this.** The fix is rank-0-reads-and-broadcasts for every
  scalar that influences control flow, including `run()`'s boolean return
  (`MPI::bcast` / `MPI::all_of` are provided for exactly this). Until that is done, **run the scaling
  study with tuning switched off**, which is what section 5 prescribes anyway.
- **An exception thrown inside a `DeferredMaps` scope is not recoverable under MPI.** It would unwind
  one rank out of a batch the others are still filling. The scheduler detects this (`poison()`) and
  turns the next flush into an `MPI_Abort` with a diagnosis rather than a hang. `QCD`'s own
  `FlowAbort` is thrown *before* any map and outside any scope, deliberately, so it is unaffected.
- **Consider pinning the phase-2 team size** instead of `Kokkos::AUTO`. `AUTO` resolves from
  registers and shared memory rather than from `league_size`, so it is very likely invariant under
  the split — but the whole design is sold on bitwise identity, and the bitwise gate in section 4 is
  what actually verifies it.
