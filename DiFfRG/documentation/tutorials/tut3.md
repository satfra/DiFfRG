(tut3)=
# Tutorial 3: Generating flow equations with FunKit

In the first two tutorials we wrote the right-hand side of the flow equations directly in C++ - either as a simple analytic flux (Tutorial 1) or in terms of pre-implemented threshold functions (Tutorial 2). For realistic truncations this quickly becomes unwieldy: the flow of an n-point function is a sum of many one-loop diagrams, each a trace over field-space indices, propagators and regulator insertions, followed by a loop-momentum integral.

DiFfRG therefore ships a Mathematica toolchain that automates the whole path from a truncation of the effective action to optimized, GPU-ready C++ integration kernels. This toolchain is built on top of [**FunKit**](https://github.com/satfra/FunKit), which performs the symbolic functional derivation and traces (using [FormTracer](https://github.com/FormTracer/FormTracer) under the hood), and DiFfRG's own `CodeTools` package, which turns the resulting symbolic expressions into C++.

In this tutorial we walk through the complete pipeline for the **O(N) model at finite temperature**, deriving the flow of the effective potential and using the generated kernels in a numerical model. The full code, including the Mathematica notebook, lives in `Tutorials/tut3` (and, with several discretizations, in `Examples/ONfiniteT`).

## The physics: O(N) effective potential

We treat a purely bosonic O(N) theory in the local-potential approximation (LPA),

```{math}
\Gamma_k[\phi] = \int_x \bigg( \frac{1}{2}(\partial_\mu\phi_i)^2 + V_k(\rho) \bigg)\,,\qquad
\rho = \frac{\phi_i\phi_i}{2} = \frac{\sigma^2 + \vec\Pi^2}{2}\,,
```

where the $N$ fields split into one radial mode $\sigma$ and $N-1$ Goldstone modes $\Pi_a$. The single object we flow is the effective potential $V_k(\rho)$, represented through its $\rho$-derivative

```{math}
m^2(\rho) = \partial_\rho V_k(\rho)\,,
```

which is exactly the FE function `m2` of the numerical model. The pion and sigma curvature masses are

```{math}
m_\pi^2 = \partial_\rho V = m^2\,,\qquad
m_\sigma^2 = \partial_\rho V + 2\rho\,\partial_\rho^2 V = m^2 + 2\rho\,\partial_\rho m^2\,.
```

The flow of $V_k$ follows from the Wetterich equation,

```{math}
\partial_t \Gamma_k = \frac{1}{2}\,\mathrm{Tr}\, G_k\, \partial_t R_k\,,
```

and our task is to derive it symbolically, perform the Matsubara sum (finite $T$) and the loop integral, and generate C++ for it.

## Setting up the notebook

Open `Tutorials/tut3/ON.nb` (or create your own notebook in the application directory). The first cell loads the package and sets the working directory to the notebook location:

```mathematica
Get["DiFfRG`"]
SetDirectory[GetDirectory[]];
```

`Get["DiFfRG`"]` loads DiFfRG's Mathematica package, which in turn loads `FunKit` and `FormTracer`. If FunKit is not yet installed, DiFfRG offers to download and install it from [its GitHub repository](https://github.com/satfra/FunKit) (this requires an internet connection on first use). `GetDirectory[]` returns the directory of the current notebook, so all generated files end up next to it.

```{note}
On this system, run notebooks headless with `wolfram -script ON.nb` (the `wolframscript` front-end is not available). FormTracer reads its configuration from the `form.set` file that sits next to the notebook.
```

## Defining the field space and truncation

FunKit needs to know the field content and which n-point functions are part of the truncation. Because $\sigma$ singles out one direction, the $N-1$ pions transform under $O(N-1)$, so we first register the corresponding group tensors:

```mathematica
DefineGroupTensors[{
  {SONfund, {SONm1, N - 1}, deltaadjSONm1[a, b], SONm1F[a, b, c],
   deltaFundSONm1[a, b], SONm1T[a, l, j], adjEpsSONm1[a, b, c], epsSONm1[a, b, c]}
}]
```

Next we declare the fields and the truncation - the propagators and the scalar vertices we keep - and register the global setup with FunKit:

```mathematica
fields = <|"Commuting" -> {σ[p], Π[p, {a}]}|>;

truncation = <|
  GammaN -> {
    {σ, σ}, {Π, Π},                          (* propagators *)
    {σ, σ, σ}, {σ, Π, Π},
    {σ, σ, σ, σ}, {σ, σ, Π, Π}, {Π, Π, Π, Π} (* scalar scatterings *)
  },
  Propagator -> {{σ, σ}, {Π, Π}},
  Rdot       -> {{σ, σ}, {Π, Π}}
|>;

Setup = <|"FieldSpace" -> fields, "Truncation" -> truncation|>;
FSetGlobalSetup[Setup];
```

Here `σ` is the radial field and `Π[p, {a}]` carries the $O(N-1)$ flavour index `a`. `Propagator` and `Rdot` list which two-point functions appear as full propagators and as regulator insertions $\partial_t R_k$ in the loop.

## Feynman rules from the potential

The vertices of the O(N) model are all derivatives of $V(\rho)$. A small helper, `λmΠnσ[pions, sigmas]`, extracts the vertex with a given number of pion and sigma legs by Taylor-expanding the potential around the equation of motion and differentiating. Its precise form is given in the notebook; the important point is that it expresses every vertex through the potential derivatives, with the first derivative identified as the pion mass, `d1V -> mΠ^2`.

With that helper, `FeynmanRules` collects the regulator-dot two-point functions, the propagators (carrying the masses `m2Pi`, `m2Sigma` and the regulator `RB`), and the scalar vertices:

```mathematica
FeynmanRules = {
  (* Regulator derivatives *)
  Rdot[{Π, Π}, {{p1_, {a1_}}, {p2_, {a2_}}}]  ->  deltaFundSONm1[a1, a2] RBdot[k^2, sps[p2, p2]],
  Rdot[{σ, σ}, {{p1_}, {p2_}}]                ->  RBdot[k^2, sps[p2, p2]],

  (* Propagators *)
  Propagator[{Π, Π}, {{p1_, {a1_}}, {p2_, {a2_}}}] ->
     deltaFundSONm1[a1, a2] / (sp[p1, p1] + m2Pi    + RB[k^2, sps[p1, p1]]),
  Propagator[{σ, σ}, {{p1_}, {p2_}}] ->
                       1    / (sp[p1, p1] + m2Sigma + RB[k^2, sps[p1, p1]]),

  (* Scalar vertices, all expressed through the potential *)
  Γσσσ[{p1_, p2_, p3_}]                         :>  λmΠnσ[{},          3],
  ΓΠΠσ[{p1_, a1_, p2_, a2_, p3_}]               :>  λmΠnσ[{a1, a2},    1],
  ΓΠΠσσ[{p1_, a1_, p2_, a2_, p3_, p4_}]         :>  λmΠnσ[{a1, a2},    2],
  ΓΠΠΠΠ[{p1_, a1_, p2_, a2_, p3_, a3_, p4_, a4_}] :>  λmΠnσ[{a1, a2, a3, a4}, 0],
  Γσσσσ[{p1_, p2_, p3_, p4_}]                   :>  λmΠnσ[{},          4]
};
```

## Deriving the flow

Now the actual derivation. FunKit builds the one-loop diagrams from the Wetterich equation, inserts the Feynman rules, and performs the field-space trace with FormTracer:

```mathematica
flowV = WetterichEquation // FTruncate // FPlot // FRoute // FPrint;
flowV = flowV /. FeynmanRules;
flowV = FormTrace[ flowV["1-Loop"]["Expression"] ];
```

Reading the pipeline stage by stage:

- `WetterichEquation` is the symbolic master equation $\tfrac12\,\mathrm{Tr}\,G\,\partial_t R$.
- `FTruncate` restricts it to the declared truncation, `FPlot` generates the contributing diagrams, `FRoute` assigns and routes the loop momenta, and `FPrint` returns the result in a form ready for substitution.
- `/. FeynmanRules` replaces every propagator, regulator insertion and vertex by its explicit expression.
- `FormTrace[...]` contracts the remaining field-space indices (the $O(N-1)$ tensors), producing a scalar integrand. We pick the `"1-Loop"` part and its `"Expression"`.

### Finite temperature

At finite temperature the loop integral becomes a Matsubara sum over the temporal momentum component plus a spatial integral. FunKit provides the machinery to expand the scalar products accordingly and perform the sum analytically:

```mathematica
flowV = ExpandScalarProductsFiniteT[flowV] // (SimplifyAllMomenta[l1, #] &);

flowV = Assuming[
  -m2Pi    - l1^2 - RB[k^2, l1^2] < 0 &&
  -m2Sigma - l1^2 - RB[k^2, l1^2] < 0,
  Map[MatsubaraSum[#, l10, T] &, flowV] // FullSimplify
];
```

`ExpandScalarProductsFiniteT` splits four-momenta into a temporal Matsubara component (`l10`) and the spatial loop momentum `l1`; `SimplifyAllMomenta` rewrites everything in terms of `l1`. `MatsubaraSum[..., l10, T]` then carries out the temporal sum, turning the bosonic occupation into the $\coth$ functions familiar from the threshold functions of [Tutorial 2](tut2.md). The `Assuming[...]` block tells Mathematica that the dispersions are positive, which is needed for the sum to converge to a closed form.

## Generating C++ kernels with `MakeKernel`

The symbolic integrand is now ready to be turned into C++. We first describe the runtime parameters the kernel depends on:

```mathematica
kernelParameterList = {
  <|"Name" -> "k"|>,
  <|"Name" -> "N"|>,
  <|"Name" -> "T"|>,
  <|"Name" -> "m2Pi",    "AD" -> True|>,
  <|"Name" -> "m2Sigma", "AD" -> True|>
};
```

These become the arguments of the generated `get(...)` method. Marking `m2Pi` and `m2Sigma` with `"AD" -> True` tells the code generator that the kernel must be differentiable with respect to them - DiFfRG needs $\partial(\text{flow})/\partial m^2$ for the Jacobian of the FEM system (these are the FE-function-dependent arguments, while `k`, `N`, `T` are plain parameters).

Then `MakeKernel` performs the loop integral symbolically where possible, optimizes the remaining integrand (common-subexpression elimination, etc.) and emits the C++:

```mathematica
MakeKernel[
  SafeFiniteTFunctions[flowV, T],
  "Name"                 -> "V",
  "Integrator"           -> "Integrator_p2",
  "d"                    -> 3,
  "AD"                   -> True,
  "Device"               -> "TBB",
  "Parameters"           -> kernelParameterList,
  "IntegrationVariables" -> {"l1"}
];
```

`SafeFiniteTFunctions[flowV, T]` rewrites the $\coth$ expressions into numerically safe finite-$T$ functions (e.g. `CothFiniteT`) that behave well in the $T \to 0$ limit. The options control code generation:

- **`"Name" -> "V"`** — names everything after the potential: it produces a kernel class `V_kernel`, an integrator class `V_integrator`, and writes them under `flows/V/`.
- **`"Integrator" -> "Integrator_p2"`** together with **`"d" -> 3`** selects DiFfRG's `DiFfRG::Integrator_p2` for a 3-dimensional spatial loop integral (the `_p2` integrator integrates over the magnitude `l1` and angles).
- **`"AD" -> True`** additionally generates an `autodiff::real` instantiation of the integrator, used for the Jacobian.
- **`"Device" -> "TBB"`** selects the execution space. `TBB` runs the integral on the CPU, and it is what a model's per-point `flux` needs: the assemblers call it from many threads at once, and a GPU integrator's `get` must not be. `"GPU"` and `"Threads"` work for `map()` and for `map_points()`, which a batched model calls once for all points ([Tutorial 6](tut6.md)).
- **`"Parameters"`** and **`"IntegrationVariables"`** supply the parameter list above and name the loop variable `l1`.

`MakeKernel` writes the following into `flows/V/`:

```
flows/V/
├── kernel.hh          # V_kernel<Regulator>: the integrand
├── V.hh               # V_integrator: holds the Integrator_p2 instances + get(...)
├── sources.m          # list of generated source files (for CMake)
└── src/
    ├── constructor.cc # V_integrator::V_integrator(...)
    ├── CT_get.cc      # double evaluation
    └── AD_get.cc      # autodiff::real evaluation (Jacobian)
```

Splitting the `get` methods into separate translation units keeps compile times manageable, since the integrand expressions can be large.

## Aggregating the kernels with `UpdateFlows`

A realistic model has many kernels (one per flowing function). `UpdateFlows` collects every kernel generated in the working directory and wires them into a single class plus a `CMakeLists.txt`:

```mathematica
UpdateFlows["ONFiniteTFlows"]
```

This writes `flows/flows.hh`, `flows/flows.cc` and `flows/CMakeLists.txt`. The generated class bundles a shared `QuadratureProvider` with all integrators and exposes setters for the global scales:

```cpp
class ONFiniteTFlows
{
public:
  ONFiniteTFlows(const DiFfRG::ConfigTree &json);

  void set_k(const double k);   // set the RG scale on all kernels
  void set_T(const double T);   // set the temperature on all kernels

  DiFfRG::QuadratureProvider quadrature_provider;
  V_integrator V;               // one member per generated kernel
};
```

```{note}
`MakeKernel` prints *"Please run UpdateFlows[] to export an up-to-date CMakeLists.txt"* as a reminder: whenever you add, remove or rename a kernel, re-run `UpdateFlows[...]` so the build picks up the change.
```

## A look at the generated code

The integrand lives in `flows/V/kernel.hh` as a static, inlineable function templated on the regulator:

```cpp
template <typename _Regulator> class V_kernel
{
public:
  using Regulator = _Regulator;

  static KOKKOS_INLINE_FUNCTION auto
  kernel(const double &l1, const auto &k, const auto &N, const auto &T,
         const auto &m2Pi, const auto &m2Sigma)
  {
    using namespace DiFfRG;
    using namespace DiFfRG::compute;
    const auto _interp2 = CothFiniteT(sqrt(powr<2>(l1) + m2Pi    + RB(powr<2>(k), powr<2>(l1))), T);
    const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
    const auto _interp4 = CothFiniteT(sqrt(powr<2>(l1) + m2Sigma + RB(powr<2>(k), powr<2>(l1))), T);
    // 0.25 * RBdot * [ (N-1) coth(ε_π)/ε_π + coth(ε_σ)/ε_σ ]
    ...
  }

  static KOKKOS_INLINE_FUNCTION auto constant(...) { return 0.; }
};
```

This is exactly the bosonic O(N) threshold structure: $N-1$ Goldstone (pion) contributions plus one radial (sigma) contribution, weighted by the regulator-derivative $\partial_t R_k$ and the finite-$T$ $\coth$ factors. The `KOKKOS_INLINE_FUNCTION` markers make the kernel usable on both CPU and GPU backends while leaving the final inlining decision to the compiler.

`flows/V/V.hh` wraps the kernel in an integrator that owns both the plain-`double` and the `autodiff::real` instantiations of `Integrator_p2`:

```cpp
class V_integrator
{
public:
  V_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json);

  using Regulator = DiFfRG::PolynomialExpRegulator<>;

  Integrator_p2<3, double,         V_kernel<Regulator>, DiFfRG::TBB_exec> integrator;
  Integrator_p2<3, autodiff::real, V_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD;

  void get(double &dest, const double &k, const double &N, const double &T,
           const double &m2Pi, const double &m2Sigma);
  void get(autodiff::real &dest, const double &k, const double &N, const double &T,
           const autodiff::real &m2Pi, const autodiff::real &m2Sigma);
};
```

The two `get` overloads share the same kernel; the assembler picks the `double` one when computing residuals and the `autodiff::real` one when computing Jacobians.

## Using the flows in the model

The numerical model in `model.hh` is structurally identical to the ones from Tutorials 1 and 2 - the only new ingredient is that it holds an `ONFiniteTFlows` member and calls the generated integrator inside the flux. First, we include the generated header and keep a (mutable) instance of the flow class:

```cpp
#include "flows/flows.hh"

class ON_finiteT : public def::AbstractModel<ON_finiteT, Components>,
                   public def::fRG, public def::LLFFlux<ON_finiteT>,
                   public def::FlowBoundaries<ON_finiteT>, public def::AD<ON_finiteT>
{
protected:
  const Parameters prm;
  mutable ONFiniteTFlows flow_equations;

public:
  ON_finiteT(const ConfigTree &json)
    : def::fRG(json.get_double("/physical/Lambda")), prm(json), flow_equations(json)
  {
    flow_equations.set_k(Lambda);
    flow_equations.set_T(prm.T);
  }
```

Whenever the RG scale changes, we forward it to the kernels:

```cpp
  void set_time(double t_)
  {
    t = t_;
    k = std::exp(-t) * prm.Lambda;
    flow_equations.set_k(k);
  }
```

Finally, the flux function computes the two curvature masses from the FE function and its derivative, and calls the generated integrator to evaluate $\partial_t V'$:

```cpp
  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &flux,
            const Point<dim> &x, const Solution &sol) const
  {
    const auto rho           = x[0];
    const auto &fe_functions  = get<"fe_functions">(sol);
    const auto &fe_derivatives = get<"fe_derivatives">(sol);

    const auto m2Pi    = fe_functions[idxf("m2")];
    const auto m2Sigma = fe_functions[idxf("m2")] + 2. * rho * fe_derivatives[idxf("m2")][0];

    flow_equations.V.get(flux[idxf("m2")][0], k, prm.N, prm.T, m2Pi, m2Sigma);
  }
```

The arguments passed to `V.get(...)` line up exactly with the `kernelParameterList` we declared in Mathematica: `k, N, T, m2Pi, m2Sigma`. The result is written into the flux of the `m2` equation, which the CG assembler then integrates against the test functions just as in the previous tutorials.

## Building and running

The `Tutorials/tut3/CMakeLists.txt` adds the generated flow library as a subdirectory and links it into the executable:

```cmake
find_package(DiFfRG REQUIRED HINTS ${DiFfRG_DIR} $ENV{HOME}/.local/share/DiFfRG/)

add_subdirectory(flows)               # builds the ONFiniteTFlows library

add_executable(tut3 tut3.cc)
target_link_libraries(tut3 PRIVATE DiFfRG::DiFfRG ONFiniteTFlows)
```

Build and run it just like the previous tutorials:

```bash
$ cd Tutorials/tut3
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j8
$ ./tut3
```

The simulation flows the effective potential from the UV cutoff $\Lambda$ down to the IR. Besides the usual VTK/HDF5 output of the FE function `m2`, the model's `readouts` method writes a `data.csv` with physical observables along the flow - the condensate $\sigma$ and the pion and sigma masses $m_\pi$, $m_\sigma$. You can post-process these with the Python utilities shown in [Tutorial 2](tut2.md); the `Tutorials/tut3/phasediagram.ipynb` notebook demonstrates scanning the model in temperature.

## Regenerating the flows yourself

The `flows/` directory shipped with the tutorial was produced by exactly the steps above. To regenerate it - for instance after changing the truncation - run the notebook again:

```bash
$ cd Tutorials/tut3
$ wolfram -script ON.nb
```

This re-runs the derivation, `MakeKernel` and `UpdateFlows`, overwriting the `flows/` tree. Re-running CMake then picks up any changes to `flows/CMakeLists.txt`.
