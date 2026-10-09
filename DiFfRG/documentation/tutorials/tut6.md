(tut6)=
# Tutorial 6: Batched evaluation with `evaluate_batch`

The models of the previous tutorials define their flux and source **per point**: `flux(F, x, sol)` receives the
solution at one quadrature point and computes the flux there, typically by one momentum integral per call. This
tutorial shows how a model evaluates them **for all points at once**, which lets a single call evaluate an integral
at thousands of points, on the CPU or on the GPU.

The code lives in `Tutorials/tut6`: the O(N) model at finite temperature of [Tutorial 3](tut3.md), once for the CG
assembler and once for the Kurganov-Tadmor (KT) finite-volume assembler, with the Matsubara sum done numerically
by the finite-temperature integrator.

## How the assemblers call the model

Every assembler (CG, DG, LDG and KT) assembles a residual or a jacobian in four stages:

| stage | what happens |
| --- | --- |
| extract | the model's `extract()` (extractors, e.g. the field at the EoM); LDG also builds its levels here |
| gather | the solution at all quadrature points (or KT face traces) is collected into one *batch* |
| evaluate | the model is evaluated at all points of the batch: **one** call of `model.evaluate_batch(out, batch)` |
| scatter | the results are contracted with the test functions and added into the global vector or matrix |

`evaluate_batch` is the only place where the model's physics is evaluated. Its default, inherited from
`def::AbstractModel`, calls the per-point `flux`, `source` (and, for KT, `diffusion_flux`) at every point of the
batch, in a parallel loop. That is why every model of the earlier tutorials works unchanged.

The assemblers record how long each stage takes. After a run,

```cpp
const AssemblyPhaseTimes &t = assembler.residual_phase_times(); // or jacobian_phase_times()
// t.extract, t.gather, t.evaluate, t.scatter: seconds summed over t.calls calls
```

and `tut6.cc` prints these averages at the end of the run. If `evaluate` dominates, the model's integrals are the
cost, and this tutorial is about making them cheaper.

## Do you need to override `evaluate_batch`?

On the CPU, often not. The default already runs all points in one flat parallel loop, which is most of the speedup
batching gives on a CPU. Overriding `evaluate_batch` pays off when

- the integrals should run on the **GPU**: a GPU integral evaluated per point is one kernel launch per point,
  while `map_points` evaluates all points in one launch;
- several terms share work, which one function can compute once for all of them;
- the per-point callbacks are not thread safe (a GPU integrator's `get()` must not be called from several threads
  at once, which is exactly what the default does).

## Kernels with `map_points`

`map_points` is the integrator's batched entry point. `MakeKernel` emits it with `"MapPoints" -> True`
(`Tutorials/tut6/ON.wl`):

```mathematica
MakeV[expr_,name_,parameters_,device_]:=MakeKernel[expr,
	"Name"->name,
	"Integrator"->"Integrator_fT_p2",
	"d"->4,
	"AD"->True,
	"Device"->device,
	"MapPoints"->True,
	"MatsubaraEven"->True,
	"Parameters"->parameters,
	"IntegrationVariables"->{"l1","l10"}
];
MakeV[flowV,"V",kernelParameterList,"TBB"];
MakeV[flowV,"V_GPU",kernelParameterList,"GPU"];
```

Unlike Tutorial 3, the flow is not summed over Matsubara frequencies analytically: the kernel depends on the
spatial momentum `l1` and the frequency `l10`, and `Integrator_fT_p2` sums over the frequencies numerically. Each
kernel is generated twice, for the CPU (`"TBB"`) and for the GPU.

Next to `get`, the generated integrator class (`flows/V/V.hh`) now has

```cpp
void map_points(const DiFfRG::PointSpan<double> dest, const DiFfRG::PointArg<double> &k,
                const DiFfRG::PointArg<double> &N, const DiFfRG::PointArg<double> &T,
                const DiFfRG::PointArg<double> &m2Pi, const DiFfRG::PointArg<double> &m2Sigma);
```

plus the same for the AD number types. `dest` is an array of results, one per point. Every argument is a
`PointArg`: either one value, shared by all points, or one value per point -- a `PointSpan` (e.g. a column of the
batch) or a `std::vector`, with exactly `dest.size()` entries. `dest[i]` is the integral with every per-point
argument taken at index `i`. A shared value or a column of another number type is converted, e.g. a `double` into
an AD argument.

`map_points` is available for the vacuum integrators (`Integrator_p2*`) and the finite-temperature ones
(`Integrator_fT_p2*`), but not for the lattice integrators. It is rank-local: unlike `map()`, it may be called
from inside assembly.

## Writing `evaluate_batch` (CG)

The CG model of `model.hh` has a flux $F = V(m_\pi^2, m_\sigma^2)$ with $m_\pi^2 = m^2$ and
$m_\sigma^2 = m^2 + 2\rho\, \partial_\rho m^2$, and no source:

```cpp
template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
{
  // Cell points ask for flux and source, faces for the flux only; this model has no source.
  if (!out.requested(Term::flux)) return;
  // double, or an AD number when the assembler builds the jacobian.
  using NT = typename Batch::number_type;
  const auto m2Pi = batch.values(idxf("m2"));
  const auto dm2Pi = batch.derivatives(idxf("m2"), 0);
  const auto rho = batch.coordinates(0);
  std::vector<NT> m2Sigma(batch.size());
  for (size_t i = 0; i < batch.size(); ++i)
    m2Sigma[i] = m2Pi[i] + 2. * rho[i] * dm2Pi[i];
  // k, N and T are shared by all points, m2Pi and m2Sigma have one value per point.
  flow_equations.V.map_points(out.flux(idxf("m2"), 0), k, prm.N, prm.T, m2Pi, m2Sigma);
}
```

Step by step:

- **`out`** holds the results. The assembler asks only for the terms it needs: `Term::flux | Term::source` at
  cell points, `Term::flux` at faces (for the boundary and the numerical flux). `out.requested(Term::source)`
  tells the model whether to compute a term at all; `out.source(c)` for a term that was not requested throws.
  Everything in `out` is zero on entry, so a term that is identically zero needs no code.
- **Columns.** `batch.values(c)`, `batch.derivatives(c, d)`, `batch.hessians(c, d1, d2)` and
  `batch.coordinates(d)` are the inputs of component `c` (direction `d`) at all points, and `out.flux(c, d)`,
  `out.source(c)` the outputs; all of them are `PointSpan`s, indexed by the point `i`. A column can be passed to
  `map_points` directly, as `m2Pi` is here.
- **Intermediate quantities** such as $m_\sigma^2$ go into a `std::vector<NT>` of `batch.size()` entries, which
  `map_points` also accepts directly.
- **One `map_points` call per integral**, which on the GPU is one kernel launch.

`batch.extractors()`, `batch.variables()` (shared by all points), `batch.x(i)` and `batch.cell_width(i)` are also
available.

`model.hh` keeps the per-point `flux` as well. It is not needed by the batched path, but with
`/tutorial/backend = per_point` the model calls the default `evaluate_batch` instead, which uses it -- a direct
comparison of the two.

## Jacobians come for free

The model derives from `def::AD`, so its jacobian is the forward-mode automatic derivative of `evaluate_batch`. The
assembler calls `evaluate_batch` with `Batch::number_type = autodiff::real` (KT also uses `autodiff::Real<2,
double>`), and each copy of the points carries a different seed direction: the derivative with respect to
$m^2$, to $\partial_\rho m^2$, and so on. All directions are stacked into one batch, so the jacobian takes
**one** `evaluate_batch` call per group of directions rather than one per point and direction.

Two consequences for `evaluate_batch`:

- It must work for any number type: write it in terms of `typename Batch::number_type`, as above. The generated
  integrators have `map_points` overloads for the AD types when the kernel is generated with `"AD" -> True`.
- `batch.size()` is not the number of quadrature points: the AD and the LLF numerical flux evaluate stacked copies
  of the points. Size everything from `batch.size()`, and never keep per-point state across calls.

`/discretization/batched/max_stacked_points` bounds the number of points in one stacked evaluation; the default
is a few hundred MB of AD data for CG, DG and LDG and 16 MB for KT.

A model also declares which inputs its flux and source read:

```cpp
static constexpr bool batch_reads_hessians = false;
```

The CG and DG assemblers then neither compute the hessians nor seed them in the jacobian.
(`batch_reads_derivatives = false` does the same for the first derivatives.) Which inputs a batch holds is
fixed at compile time: `batch.hessians()` of a batch without them does not compile, nor does
`get<"fe_hessians">(sol)` in a per-point callback. A model used with several assemblers can ask
`if constexpr (Batch::has_hessians)` (or, per point, `if constexpr (tuple_has<"fe_hessians", Solution>)`).

## Faces and boundaries

`def::LLFFlux` (the numerical flux of DG) and `def::FlowBoundaries` evaluate the model at the faces
through the same `evaluate_batch`, with only `Term::flux` requested. `LLFFlux` needs the flux on both sides of the
face plus a finite-difference wave speed, and evaluates all of them in one call over $2 + 2N_f$ stacked copies of
the face points. A model with its own numerical flux can define `numflux_batch(out, normals, batch_s, batch_n)`
(and `boundary_numflux_batch(out, normals, batch)`); without them, the per-point `numflux` and `boundary_numflux`
are used.

## The KT model

The KT assembler splits the flux into an advection part (`flux`) and a diffusion part (`diffusion_flux`), see
`Examples/ONfiniteT/model_KT.hh`. It calls `evaluate_batch` three times per residual, each time for one term on
different states: `Term::flux` on the face traces, `Term::diffusion_flux` on the face traces with their corrected
gradients, `Term::source` at the cell points.

```cpp
template <typename Out, typename Batch> void evaluate_batch(Out &out, const Batch &batch) const
{
  using NT = typename Batch::number_type;
  const auto m2 = batch.values(idxf("m2"));
  if (out.requested(Term::flux)) {
    const auto F = out.flux(idxf("m2"), 0);
    pion.map_points(F, k, prm.N, prm.T, m2);
    for (auto &f : F)
      f *= prm.N - 1.;
  }
  if (out.requested(Term::diffusion_flux)) {
    const auto dm2 = batch.derivatives(idxf("m2"), 0);
    const auto rho = batch.coordinates(0);
    std::vector<NT> m2Sigma(batch.size());
    for (size_t i = 0; i < batch.size(); ++i)
      m2Sigma[i] = m2[i] + 2. * rho[i] * dm2[i];
    sigma.map_points(out.diffusion_flux(idxf("m2"), 0), k, prm.N, prm.T, m2Sigma);
  }
}
```

(`pion` and `sigma` are the CPU or GPU integrators, see `model.hh`.) Differences to the FEM assemblers:

- KT evaluates the flux with AD numbers **even for the residual** (`autodiff::Real<1, double>`): the wave speed
  needs $\partial F/\partial u$.
- The three calls see three different batch types, and `evaluate_batch` is compiled for each. Only the
  diffusion traces carry third derivatives (`batch.third_derivatives(c, d0, d1, d2)`, 1D only), so code that reads
  them must be guarded by `if constexpr (Batch::with_third)`, not by `out.requested(...)`.
- The extractors stay plain numbers under AD: KT keeps them frozen within a Newton step.
- The source sees the hessians only if the model declares `static constexpr bool source_uses_hessians = true`.

## Per assembler

| assembler | inputs at a point | terms per call | number types |
| --- | --- | --- | --- |
| CG | values, derivatives, hessians (unless `batch_reads_*` is false) | flux and source (cells), flux (boundary) | `double`, `autodiff::real` |
| DG | as CG | flux and source (cells), flux (faces) | `double`, `autodiff::real` |
| LDG | values of the FE functions and of every level (`batch.ldg_values(k, c)`) | as DG | `double`, `autodiff::real` |
| KT | values, reconstructed derivatives; third derivatives (diffusion, 1D); hessians (source, if requested) | one term per call | `Real<1>` (flux, residual), `Real<2>` (flux, jacobian), `double` and `autodiff::real` (= `Real<1>`; diffusion flux and source) |

The LDG levels themselves are built by `ldg_evaluate_batch<k>(out, batch)`, whose default calls the per-point
`ldg_flux<k>` / `ldg_source<k>`.

## CPU or GPU

`model.hh` picks the integrator at compile time, and `tut6.cc` selects it from the parameter file:

```toml
[tutorial]
  assembler = "cg"     # or "kt", with parameter_KT.toml
  backend = "cpu"      # per_point, cpu or gpu
```

Things to know about the GPU:

- A GPU integrator gains only if there is enough work per call: many points times many quadrature nodes. On a
  consumer GPU (double precision at 1/64 of single) the CPU is often as fast or faster in double precision;
  data-centre GPUs do much better.
- Single precision (`"ComputeType" -> "float"`) is fast on any GPU, but a residual accurate to $10^{-7}$
  relative is usually too coarse for an implicit time stepper at the tolerances of these flows: Newton stops
  converging.
- `integrator.set_map_points_policy(MapPointsPolicy::team_per_point)` (or `thread_per_point`) overrides how the
  GPU distributes the points; the default chooses from the number of points and nodes.

## Building and running

Generate the flows once (`wolfram -script ON.wl` in `Tutorials/tut6`; the generated `flows/` are checked in), then

```bash
cd Tutorials/tut6
./build.sh -j8
cd build
./tut6                          # CG, batched on the CPU
./tut6 -ss /tutorial/backend=gpu
./tut6 -p parameter_KT.toml     # KT
```

At the end of the run, `tut6` prints the average time per stage, here for CG on 61 cells with the CPU backend:

```
residual (711 calls), ms per call: extract 0.002, gather 0.019, evaluate 0.153, scatter 0.021
jacobian (33 calls), ms per call: extract 0.001, gather 0.015, evaluate 0.199, scatter 0.021
```

The model evaluation dominates, as it should. On this small 1D problem the three backends are close: per point
and batched on the CPU take the same time per call, and the GPU (a laptop RTX 4070, double precision) is about
1.5 times slower -- there are only a few hundred points per call. `Examples/ONfiniteT/bench_batched` measures
single calls on larger meshes.

## Pitfalls

- **Skipping a requested term** leaves it zero, silently. Check `out.requested(...)` for every term the model has.
- **Keying state on the point index** breaks under AD and LLF, where the batch holds stacked copies.
- **Calling `map()` in `flux`, `source` or `evaluate_batch`** aborts: `map()` is collective over MPI ranks, and each
  rank assembles only its own points. Use `map_points()`, or compute the value once in `extract()`.
- **A GPU integrator in a per-point callback** is called from many threads at once; keep the per-point callbacks
  on CPU integrators.
- **A member defined in two bases** (a shared base class and `def::AbstractModel`, which has defaults for
  `evaluate_batch`, `initial_condition`, `readouts`, ...) is ambiguous: say which one with
  `using Base::member;` in the model, as `model.hh` does.

## What to take away

* The assemblers call `evaluate_batch(out, batch)` once for all points; its default runs the per-point callbacks
  in parallel, so existing models work unchanged.
* Overriding it means reading columns from `batch`, writing columns of `out`, and evaluating each integral with
  one `map_points()` call -- which also runs on the GPU.
* With `def::AD`, the jacobian is the AD of the same function: write it for any `Batch::number_type`.
* `AssemblyPhaseTimes` shows whether the model evaluation is worth optimising at all.

## Next steps

* `Examples/ONfiniteT` (`-DONFINITET_BATCHED=ON`) has the same model for CG, DG, LDG and KT, and `bench_batched`,
  which times single residual and jacobian calls stage by stage.
* [Models](../getting_started/models.md) lists what each assembler hands the per-point callbacks.
