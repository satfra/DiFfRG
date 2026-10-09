(tut4)=
# Tutorial 4: Momentum-dependent truncations

[Tutorial 3](tut3.md) generated a *scalar* flow kernel and integrated it on a finite-element field-space grid. The other major class of fRG truncations is **momentum-dependent**: the flowing objects are dressing functions and couplings tabulated on a **momentum grid**, evaluated through **interpolators**, with no field-space discretization. This tutorial shows how to generate and wire such a truncation, using the **Yang-Mills** system in the *single-point* (SP) approximation as the example. The full code is in `Tutorials/tut4`.

We assume familiarity with the FunKit workflow from [Tutorial 3](tut3.md); only the differences are spelled out here.

## The physics

We flow the SU(N) Yang-Mills correlation functions in a vertex expansion: the gluon and ghost propagator dressings $Z_A(p)$, $Z_c(p)$ and the three couplings $Z_{A^3}(p)$, $Z_{A^4}(p)$, $Z_{A\bar c c}(p)$, each a function of a single momentum $p$. "Single-point" means the vertices are projected onto one momentum configuration, so every flowing object is a function of one momentum only. The gluon mass parameter is read off from $Z_A$ at the lowest grid point.

## Variables instead of FE functions

These objects do not live on a field-space grid, so they are declared as **variables** rather than FE functions. A variable is a value (or array of values) carried and evolved by the model without any FEM discretization. Grid-valued variables use `FunctionND<name, sizes...>`; a single value uses `Scalar<name>`:

```cpp
static constexpr uint p_grid_size = 96;

using VariableDesc =
    VariableDescriptor<FunctionND<"ZA3", p_grid_size>, FunctionND<"ZAcbc", p_grid_size>,
                       FunctionND<"ZA4", p_grid_size>, FunctionND<"ZA", p_grid_size>,
                       FunctionND<"Zc", p_grid_size>>;
using Components = ComponentDescriptor<FEFunctionDescriptor<>, VariableDesc, ExtractorDescriptor<>>;

constexpr auto idxv = VariableDesc{};
```

Note the **empty** `FEFunctionDescriptor<>`: this is a pure variable system. `idxv("ZA")` returns the offset of `ZA`'s block in the flat variable vector, so `idxv("ZA") + i` indexes grid point `i`. (Variables and extractors are also described in the [Numerical Models](../getting_started/models.md) guide.)

The model derives from `def::AbstractModel` and `def::fRG` as before, but uses `def::NoJacobians` instead of `def::AD` — the couplings are evolved explicitly, so no Jacobian is needed.

## Momentum grid and interpolators

The momentum grid is a coordinate system; here a logarithmic grid between `p_grid_min` and `p_grid_max`:

```cpp
using Coordinates1D = LogarithmicCoordinates1D<double>;
const Coordinates1D coordinates1D;   // constructed as (p_grid_size, p_grid_min, p_grid_max, p_grid_bias)
```

`coordinates1D.forward(i)` returns the physical momentum at grid point `i`. To evaluate a dressing at an arbitrary loop momentum inside a kernel, the model holds an **interpolator** per dressing:

```cpp
mutable SplineInterpolator1D<double, Coordinates1D> dtZc, dtZA, ZA, Zc;
mutable SplineInterpolator1D<double, Coordinates1D> ZA4, ZAcbc, ZA3;
```

`dtZA`/`dtZc` hold the anomalous dimensions (the propagator flows) and are needed inside the loop integrands — see the self-consistency loop below.

## Generating grid kernels with FunKit

The Mathematica side is the same `MakeKernel` / `UpdateFlows` pipeline as [Tutorial 3](tut3.md), with three additional options that turn a scalar kernel into a **grid** kernel. The gluon-propagator kernel is generated (verified from `Tutorials/tut4/Yang-Mills.nb`) as:

```mathematica
MakeKernel[ FlowAA / p^2,
  "Name"                 -> "ZA",
  "Integrator"           -> "Integrator_p2_1ang",   (* loop magnitude + 1 angle *)
  "d"                    -> 4,
  "AD"                   -> False,                   (* explicit evolution, no Jacobian *)
  "Device"               -> "GPU",
  "Type"                 -> "double",
  "Parameters"           -> kernelParameterList,     (* k and the interpolators, with C++ Types *)
  "CoordinateArguments"  -> {"p"},                   (* the external momentum *)
  "IntegrationVariables" -> {"l1", "cos1"},          (* the loop variables *)
  "Coordinates"          -> {"LogarithmicCoordinates1D<double>"} ];
```

The new ingredients relative to the scalar case:

- **`"Coordinates"`** is what makes this a grid kernel. In addition to the single-point `get(dest, p, ...)`, the generated integrator gains a `map(dest, coordinates, ...)` method that evaluates the kernel at **every** point of the coordinate grid in parallel and writes the results into `dest`.
- **`"CoordinateArguments" -> {"p"}`** names the external momentum that labels the grid.
- **`"IntegrationVariables" -> {"l1", "cos1"}`** are the loop integration variables. `Integrator_p2_1ang` integrates over the loop magnitude and one angle; the vertices (`ZA3`, `ZA4`, `ZAcbc`) use `Integrator_p2_4D_2ang` (two angles).
- **`"Parameters"`** entries carry explicit C++ `Type`s, including the interpolator type, so the kernel receives the dressings `ZA3, Zc, ...` by const reference and can evaluate them at the loop momentum.

`UpdateFlows["YangMillsFlows"]` then assembles the `YangMillsFlows` class. The generated integrator (`flows/ZA/ZA.hh`) holds a GPU integrator and exposes both methods:

```cpp
class ZA_integrator {
public:
  using Regulator = DiFfRG::PolynomialExpRegulator<>;
  Integrator_p2_1ang<4, double, ZA_kernel<Regulator>, DiFfRG::GPU_exec> integrator;

  // tabulate the flow over the whole momentum grid
  DiFfRG::GPU_exec map(double *dest, const LogarithmicCoordinates1D<double> &coordinates,
                       const double &k, const SplineInterpolator1D<...> &ZA3, /* ... */);
  // evaluate at a single momentum p
  void get(double &dest, const double &p, const double &k, const SplineInterpolator1D<...> &ZA3, /* ... */);
};
```

## Wiring the model

Three methods drive a variable system (`model.hh`).

**Initial condition** — fill each grid block from the physical momentum:

```cpp
template <typename Vector> void initial_condition_variables(Vector &values) const
{
  for (uint i = 0; i < p_grid_size; ++i) {
    const double p = coordinates1D.forward(i);
    values[idxv("ZA") + i] = (powr<2>(p) + prm.m2A) / powr<2>(p);
    values[idxv("Zc") + i] = 1.;
    // ... couplings initialised from alpha and a logarithmic tilt ...
  }
}
```

**Scale update** — forward the RG scale to the kernels each step:

```cpp
void set_time(double t_) { t = t_; k = std::exp(-t) * Lambda; flow_equations.set_k(k); }
```

**The flow** — `dt_variables` is the heart of the model. It loads the current dressings into the interpolators, bundles the kernel arguments, and `map`s each flow onto the grid:

```cpp
template <typename Vector, typename Solution> void dt_variables(Vector &residual, const Solution &data) const
{
  const auto &variables = get<"variables">(data);

  // load the interpolators from the current variable buffer
  ZA3.update(&variables.data()[idxv("ZA3")]);   ZAcbc.update(&variables.data()[idxv("ZAcbc")]);
  ZA4.update(&variables.data()[idxv("ZA4")]);   ZA.update(&variables.data()[idxv("ZA")]);
  Zc.update(&variables.data()[idxv("Zc")]);

  const auto arguments = device::tie(k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);

  // propagators: iterate until the anomalous dimensions are self-consistent
  bool converged = false; int n_iter = 0;
  while (!converged) {
    flow_equations.ZA.map(&residual[idxv("ZA")], coordinates1D, arguments);
    flow_equations.Zc.map(&residual[idxv("Zc")], coordinates1D, arguments);
    dtZA.update(&residual[idxv("ZA")]);   // feed the just-computed flows back in
    dtZc.update(&residual[idxv("Zc")]);
    // ... stop when the relative change is below eta_tol or n_iter > eta_iter_max ...
  }

  // vertices: a single pass with the converged propagator flows
  flow_equations.ZA4.map(&residual[idxv("ZA4")], coordinates1D, arguments);
  flow_equations.ZAcbc.map(&residual[idxv("ZAcbc")], coordinates1D, arguments);
  flow_equations.ZA3.map(&residual[idxv("ZA3")], coordinates1D, arguments);
}
```

The **self-consistency loop** is the only conceptually new piece: the propagator flows $\partial_t Z_A$, $\partial_t Z_c$ appear (as anomalous dimensions) inside their own integrands through the `dtZA`/`dtZc` interpolators, so they are iterated to `eta_tol` before the vertex flows are evaluated once.

Output is written in `readouts` straight to HDF5 over the grid:

```cpp
auto &hdf = output.hdf5();
hdf.map("ZA", coordinates1D, &variables.data()[idxv("ZA")]);   // ... and Zc, ZA3, ZA4, ZAcbc ...
hdf.scalar("k", k);
```

## Wiring main

A pure variable system uses the `Variables::Assembler` (spatial dimension `0`, no FE functions) and is carried by a default-constructed `FlowingVariables` — there is no mesh, discretization or VTK output:

```cpp
using Model = YangMills;
using Assembler = Variables::Assembler<Model>;
using TimeStepper = TimeStepperBoostABM<Assembler>;

int main(int argc, char *argv[])
{
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  const auto json = config_helper.get_json();

  OutputSession<Assembler> output(json);
  Model model(json);
  Assembler assembler(model, json);
  TimeStepper time_stepper(json, assembler, output);

  FlowingVariables initial_condition;          // no discretization argument
  initial_condition.interpolate(model);        // calls initial_condition_variables

  time_stepper.run(initial_condition, 0., json.get_double("/timestepping/final_time"));
}
```

The couplings are evolved explicitly, so an explicit Adams-Bashforth-Moulton stepper (`TimeStepperBoostABM`) is appropriate instead of the implicit IDA stepper used for the FEM tutorials.

## Building and running

```bash
$ cd Tutorials/tut4
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j8
$ ./tut4
```

```{important}
The kernels are generated with `"Device" -> "GPU"`, so this tutorial requires a CUDA build of DiFfRG **and a GPU at runtime**. To run on the CPU instead, regenerate the kernels with `"Device" -> "TBB"`. The interpolators need no change: they hold their data on both sides and evaluate on whichever one the kernel runs on.
```

The run flows from the UV cutoff $\Lambda$ to the IR and writes `output.h5`, containing the momentum-dependent dressings $Z_A(p)$, $Z_c(p)$ and couplings, plus the gluon mass $m_A^2$. Read these with the Python HDF5 utilities (`DiFfRG.file_io`).

## Towards full momentum dependence

The SP truncation keeps only the external-momentum dependence of the vertices. A **fully momentum-dependent** truncation additionally resolves the angle between the external legs, so a three-point vertex becomes a function of two momenta and the enclosed angle, `(|p1|, |p2|, cos(p1,p2))`. The infrastructure scales up directly:

- **Coordinates** become a product grid, `LogLogLinCoordinates` (= `CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>`): two logarithmic momentum axes and one linear angle axis.
- **Variables** become multi-dimensional, e.g. `FunctionND<"ZA3", vertex_grid_size>` with `vertex_grid_size = vertex_p_grid_size * vertex_p_grid_size * n_angles` laid out row-major over `(p1, p2, cos)`.
- **Interpolators** become `LinearInterpolatorND<double, LogLogLinCoordinates>`.
- **Kernels** use multi-angle integrators — `Integrator_p2_4D_2ang` / `_3ang` — generated with the same `MakeKernel` options, plus `"Coordinates" -> {"LogLogLinCoordinates"}`, `"CoordinateArguments" -> {"p1","p2","cosp1p2"}`, and a `"KernelBody" -> DeclareAnglesP34Dpqr[l1,p1,p2]` that declares the loop–external cosines in terms of the integration angles. In the model, the residual is filled with `flow_equations.ZA3.map(&residual[idxv("ZA3")], coordinates3D, arguments)`, exactly the same `.map` call as the 1D dressings but over the 3D coordinate pack.

The angle-resolved Yang-Mills example lives in `Examples/YangMills/Full` and is the worked reference for this: `Yang-Mills.m`/`Yang-Mills.nb` generate the six kernels (`ZA`, `Zc`, `ZA3`, `ZAcbc`, `ZA4tadpole`, `ZA4SP`) and `model.hh` wires them with the `.map` pattern. Like SP, it must be tuned in the initial gluon mass `m2A` — too negative lands on the Higgs branch (the flow diverges), too small on the massive branch, and the scaling regime lies in between; set `/tuning/tune_m2A` to bisect onto it.
