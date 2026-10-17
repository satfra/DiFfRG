(tut9)=
# Tutorial 9: Momentum-dependent flows with NumTracer

Tutorials 3, 4 and 8 trace their diagrams with FormTracer and emit the kernels with `MakeKernel`. This tutorial uses
[NumTracer](https://github.com/satfra/NumTracer) instead. NumTracer contracts the Dirac, colour and flavour structure
of every diagram numerically and writes straight-line C++ kernels; it pays off for large truncations, where symbolic
traces get slow, and its DiFfRG front end `MakeNTKernelDiFfRG` has options that make the generated kernels
considerably faster. Those options are the second half of this tutorial.

The model is the quark-meson model of [Tutorial 8](tut8.md) at finite temperature, with the meson wave functions now
depending on the frequency and the spatial momentum, $Z_\pi(p_0, |\vec p|)$ and $Z_\sigma(p_0, |\vec p|)$. The code
is in `Tutorials/tut9`: the derivation `QM.wl`, the model `model.hh`, `tut9.cc`, and `bench.cc`, which times the
codegen options. The generated flows are shipped in `flows/` and `bench_flows/`, so the tutorial builds without
Mathematica, but not without NumTracer: the flows link against it.

## Installing NumTracer

NumTracer is a C++20 library with a Mathematica front end. The library is needed to build any project with
NumTracer flows, the front end only to generate them. Build and install it with

```bash
$ git clone https://github.com/satfra/NumTracer.git
$ cmake -S NumTracer/numtracer -B NumTracer/build -DCMAKE_BUILD_TYPE=Release
$ cmake --build NumTracer/build -j4
$ cmake --install NumTracer/build
```

This installs to `~/.local/share/NumTracer`, which is where generated flows look for it
(`find_package(NumTracer HINTS ~/.local/share/NumTracer)`); pass `-DCMAKE_INSTALL_PREFIX=...` for another place and
`-DNumTracer_DIR=<prefix>/lib/cmake/NumTracer` to the CMake call of your project. If a Wolfram kernel is found at
configure time, the install also puts the Mathematica front end on Wolfram's application path, so that
`Get["NumTracer`"]` works from any notebook or script. `cmake --install` runs NumTracer's test suite first; pass
`-DNUMTRACER_TEST_ON_INSTALL=OFF` to skip it.

To generate flows you additionally need [FunKit](https://github.com/satfra/FunKit) and
[TensorBases](https://github.com/satfra/TensorBases), as for the other derivation tutorials. FORM is not needed:
NumTracer does its own traces.

## The model

The fields are those of Tutorial 8: $N_f = 2$ quarks with $N_c = 3$ colours and the mesons
$\phi = (\sigma, \vec\pi)$, $\rho = \phi^2/2$. With $p^2 = p_0^2 + \vec p^{\,2}$ the meson two-point functions are

```{math}
\Gamma^{(2)}_{\pi\pi}(p) = Z_\pi(p_0, |\vec p|)\, p^2 + V'(\rho)\,, \qquad
\Gamma^{(2)}_{\sigma\sigma}(p) = Z_\sigma(p_0, |\vec p|)\, p^2 + V'(\rho) + 2\rho\, V''(\rho)\,.
```

At finite temperature the frequency $p_0$ takes the bosonic Matsubara values $2\pi T n$, so $Z_\pi$ and $Z_\sigma$
are functions of an integer and a momentum. Unlike in Tutorial 8, $Z_\sigma$ is independent of $Z_\pi$. To keep the
tutorial small, the quarks keep their classical wave function, $Z_q = 1$, and the Yukawa coupling $h$ of the bare
fields does not flow. The quark mass is $m_q = h\sigma/2$. The term $-c\,\sigma$ breaks chiral symmetry explicitly.

All regulators are three-dimensional, $R_k(p) = R_B(k^2, \vec p^{\,2})$ for the mesons, and act on the spatial
momentum only.

**Renormalisation.** The meson field is rescaled, $\bar\phi = \bar Z^{1/2}\phi$, such that $Z_\pi(0, p_\text{min}) =
1$ at every scale, where $p_\text{min}$ is the smallest momentum of the grid. $\bar Z$ then enters only through
$\eta = -k\partial_k \bar Z / \bar Z$: in the regulator insertions, $k\partial_k R_k - \eta R_k$, and in the
rescaling of every coupling, $\eta Z_{\pi,\sigma}$ for the wave functions and $\eta(u + \rho u')$ for $u = V'$. The
couplings of the bare fields enter as $h/\sqrt{\bar Z}$ and $c/\sqrt{\bar Z}$; $\bar Z$ is a variable of the model.

**The wave-function flows.** With $\dot\Gamma_{\pi\pi}(p)$ the one-loop flow $k\partial_k \Gamma^{(2)}_{\pi\pi}(p)$,
symmetrised in $\vec p \to -\vec p$,

```{math}
k\partial_k Z_\pi(p_0, |\vec p|) = \frac{\dot\Gamma_{\pi\pi}(p) - \dot\Gamma_{\pi\pi}(0)}{p^2} + \eta\, Z_\pi(p_0, |\vec p|)\,,
```

and the same for $Z_\sigma$. The flow is linear in $\eta$, so the condition $k\partial_k Z_\pi(0, p_\text{min}) = 0$
fixes $\eta$ from two evaluations of the flow at that point.

The system is therefore

- the PDE for $u(\rho)$, discretised with CG as in Tutorial 8;
- $Z_\pi$ and $Z_\sigma$ on a grid of 4 Matsubara frequencies times 24 spatial momenta, and $\bar Z$, as Variables;
- the derivatives of $V$ at the EoM as extractors.

## The derivation file

`QM.wl` follows Tutorial 8 in its physics; this section covers what changes with NumTracer. Run it headless with
`wolfram -script QM.wl`; it takes about 45 s.

**Packages.** NumTracer is loaded after DiFfRG:

```mathematica
Get["DiFfRG`"];
Get["NumTracer`"];
```

**Feynman rules.** FunKit derives the diagrams as before. The quark rules come from the `qbqFT` basis of TensorBases,
whose three elements $\{i\gamma_0,\ i\vec\gamma\cdot\vec p,\ 1\}$ separate the temporal and spatial structures. The
meson rules are written out by hand, and the wave functions are functions of the frequency and the spatial momentum:

```mathematica
ZPiOf[p_] := ZPi[vec[p, 0], Sqrt[sps[p, p]]];
GPhi[m2_, p_, Z_] := 1/(Z*sp[p, p] + m2 + RB[k^2, sps[p, p]]);
...
Propagator[{pion, pion}, {{p1_, {f1_}}, {p2_, {f2_}}}] :> deltaAdjFlav[f1, f2]*GPhi[d1V, p2, ZPiOf[p2]],
Rdot[{sig, sig}, {{p1_}, {p2_}}] :> RBdot[k^2, sps[p2, p2]] - etaPi0*RB[k^2, sps[p2, p2]],
```

`sp` is the four-dimensional scalar product, `sps` the spatial one and `vec[p, 0]` the frequency.

**Frames.** NumTracer evaluates the diagrams in an explicit momentum frame. `propFrameFT` puts the external momentum
along the first spatial axis and the loop momentum at the angle $\cos_1$; `frameShiftedLoop` adds the fermionic loop
momentum, whose frequency is shifted by $\pi T$:

```mathematica
frameProp = NumTracer`frameShiftedLoop[NumTracer`propFrameFT[p0, p, f0, l1, cos1, p1, l1], l1, lf1, Pi T];
argsProp = {l1, cos1, f0, p0, p, k, T};
```

`ftParam` then replaces every scalar product by its value in the frame, and `NumTrace[FromFunKit[...]]` contracts
the diagrams:

```mathematica
traceMeson[list_] := NumTrace[
   FromFunKit[ftParam[dressingRules[FEx @@ FTerm /@ list], frameProp, $loops, {p1}, {cos1}]],
   "Frame" -> frameProp, "Args" -> argsProp];
```

`NumTrace` is fast: the expensive step, running the generated trace programs, is deferred to the build (see below).

**Kernel parameters.** As in every DiFfRG derivation, the parameter list is the C++ signature of the kernels, and
`model.hh` has to pass the arguments in the same order. The two dressings are spline interpolators on a stack of
Matsubara frequencies:

```mathematica
bosCoords = "BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>";
bosInterp = "SplineInterpolator1DStack<double, " <> bosCoords <> ">";
kernelParameterList = {
   <|"Name" -> "k", "Type" -> "double"|>,
   ...
   <|"Name" -> "d1V", "Type" -> "double", "AD" -> True|>,
   ...
   <|"Name" -> "ZPi", "Type" -> bosInterp, "Const" -> True, "Reference" -> True|>,
   <|"Name" -> "ZSig", "Type" -> bosInterp, "Const" -> True, "Reference" -> True|>};
```

**Emitting a kernel.** `MakeNTKernelDiFfRG` takes the options of `MakeKernel` that describe the integral:

```mathematica
MakeNTKernelDiFfRG[ntk, etaPi0*ZPi[p0, p],     (* the traced kernel, and the constant term *)
  "Name" -> "ZPi",
  "Integrator" -> "Integrator_fT_p2_1ang",
  "Parameters" -> kernelParameterList,
  "Coordinates" -> {bosCoords},
  "CoordinateArguments" -> {"p0", "p"},
  "IntegrationVariables" -> {"l1", "cos1", "f0"}]
```

The Matsubara frequency is the last integration variable. NumTracer checks whether the kernel is even in it and, if
so, declares it `matsubara_even`, which halves the number of evaluations. The defaults are `"Device" -> "GPU"` and
`"Coordinates" -> {"LogarithmicCoordinates1D<double>"}`, so a kernel on another grid has to name its coordinates.

**Selecting flows.** Each flow is wrapped in `NTFlowGate`. Setting `NT_FLOWS=V wolfram -script QM.wl` then emits only
the potential, which saves time while working on one flow. At the end,

```mathematica
NTFlowSelectionReport[];
If[!NTFlowSliceQ[], UpdateNTFlows["QMFlows"]];
```

writes `flows/CMakeLists.txt` and `flows.{hh,cc}`. Use `UpdateNTFlows`, not `UpdateFlows`: it also adds NumTracer to
the flows library. The log prints a note after each flow that `flows/CMakeLists.txt` needs refreshing; the
`UpdateNTFlows` call at the end takes care of it.

**Generation happens at build time.** `MakeNTKernelDiFfRG` writes the kernel and, into `gen/`, the C++ programs that
compute its traces. The first build of the flows library compiles and runs them (the `numtrace` target) and stores
the result in `flows/<Name>/kernels.hh`; `flows/<Name>/numtrace.json` records that this is done, so later builds skip
it. The shipped flows are already generated, which is why `gen/` is not part of the tutorial.

## Making the kernels fast

The flows of this model are cheap enough that the whole flow runs in about 20 s. In larger truncations the momentum
integrals dominate, and the options below decide how expensive they are. Every number here is from `bench.cc` on a
laptop RTX 4070 (see [Building and running](#building-and-running)); the variants it compares are generated into
`bench_flows/`.

### Kernel groups

$Z_\pi$ and $Z_\sigma$ are two projections of one loop: the same propagators, the same regulator insertions, the same
lookups of $Z_\pi$ and $Z_\sigma$ at the loop momenta. As two kernels, each computes all of that itself.
`MakeNTKernelGroupDiFfRG` emits them as one kernel with two outputs, which computes it once:

```mathematica
MakeNTKernelGroupDiFfRG[{ntkPi, ntkSig},
  "Name" -> "Mesons",
  "Outputs" -> {"ZPi", "ZSig"},
  "Constants" -> {etaPi0*ZPi[p0, p], etaPi0*ZSig[p0, p]},
  "SharedGrids" -> {{ZPi, ZSig}},
  mesonOpts]
```

`"Outputs"` names the outputs and `"Constants"` gives one constant term per output; the other options are those of
`MakeNTKernelDiFfRG`. All outputs share the parameters, the integrator and the coordinates. The group's `map()` takes
one destination per output, and `get()` returns a `DiFfRG::MultiValue`:

```cpp
flow_equations.Mesons.map({&residual[idxv("ZPi")], &residual[idxv("ZSig")]}, coordinates, arguments);

MultiValue<double, 2> g;
flow_equations.Mesons.get(g, p0, p, arguments);   // g[0]: ZPi, g[1]: ZSig
```

Groups are not available with `"AD" -> True`. On the 96 grid points of this model the group takes 8.6 ms, the two
separate kernels 13.7 ms (T = 100 MeV); at T = 10 MeV it is 22.6 ms against 36.3 ms.

### Shared lookups

A lookup in an interpolator first maps the momentum to grid coordinates, which costs a logarithm per coordinate on a
logarithmic grid, and then interpolates. NumTracer computes each such transform once per kernel evaluation and reuses
it for every lookup at the same momentum (`"ShareInterpolatorIndex"`, on by default for DiFfRG's interpolators).

Lookups of *different* interpolators can share the transform only if they live on the same grid, which the code
generator cannot know. `"SharedGrids" -> {{ZPi, ZSig}}` declares it. A wrong declaration gives wrong numbers, so
`map()` and `map_points()` check it at run time in builds without `NDEBUG` or with `-DDIFFRG_CHECK_SHARED_GRIDS`.

### Hoisting

The integrator of a finite-temperature flow evaluates the kernel at every combination of external momentum, loop
momentum $l$, angle and Matsubara frequency $l_0$. Much of the kernel depends only on $l$ and $l_0$: the regulators
$R_B(k^2, l^2)$, the dressings at the loop momentum $Z_\pi(l_0, l)$, the propagators built from them. The hoisted form
splits the kernel into

- `pre(l, l0, ...)`: everything that depends only on the radial loop momentum, the frequency and the arguments shared
  by all points;
- `post(P, ...)`: the rest, taking the values of `pre()` from `P`.

The integrator tabulates `pre()` once per pair of nodes $(l, l_0)$ and sign of $l_0$ and evaluates `post()` at every
quadrature point. The table is reused across calls as long as the scale, the arguments `pre()` reads and the data of
the interpolators are unchanged.

`"Hoist"` is on by default; it applies to finite-temperature flows and does nothing for vacuum ones. Here it makes
the group 1.4 times faster at every temperature tested (8.6 against 11.9 ms at T = 100 MeV, 22.6 against 31.4 ms at
T = 10 MeV). The results agree to rounding.

Two related switches exist on the C++ side. `integrator.set_pre_table_reuse(false)` rebuilds the table on every call,
for instance to rule it out when results look suspicious; the integrator is the member `integrator` of each flow,
e.g. `flow_equations.Mesons.integrator`. The compile definition `DIFFRG_MATSUBARA_SIGN_LOOP` makes the integrator
evaluate $l_0$ and $-l_0$ in a loop instead of two inlined copies of the kernel, which halves the code size of large
kernels. On the laptop it saves up to 2 %; it is meant for data-centre GPUs, where large kernels can stall on
instruction fetch, and is off until it has been measured there.

### Point arguments

The potential is evaluated by `evaluate_batch` at all quadrature points of the field grid in one `map_points()` call
([Tutorial 6](tut6.md)), with $\rho$, $u = V'$ and $u'$ different at every point. `pre()` must not read such a
per-point argument, since its table is shared by all points. `"PointArguments"` lists them:

```mathematica
MakeNTKernelDiFfRG[ntk, "Name" -> "V",
  "PointArguments" -> {"rhoPhi", "d1V", "d2V", "d3V"},
  "Device" -> "GPU", "AD" -> True, "MapPoints" -> True, ...]
```

For a kernel with `"MapPoints" -> True` the default is the AD parameters, here `d1V`, `d2V`, `d3V`; for other
kernels it is empty. **Plain parameters that vary per point, like `rhoPhi` here, have to be added by hand.** If one
is missing, the results are still right: the integrator sees that `pre()` reads an argument that was passed per point
and falls back to the plain kernel. It is only slower. With `rhoPhi` in the list the potential takes 0.25 ms for 400
points, with the default list (`VAuto` in `bench_flows/`) 2.4 ms.

### Other options

- `"HoistLoopConstLookups"` (automatic): lookups whose arguments depend only on $k$ are evaluated once on the host and
  passed into the kernel. There are none in this model.
- `"MatsubaraFiniteExtent"` (automatic): if every term of the summand decays with the regulator, the Matsubara sum
  is cut off where the regulator vanishes instead of being approximated by a quadrature.
- `"ComputeType" -> "float"`: evaluate the kernel in single precision. Only for flows that tolerate it; two-point
  flows of the form $(\dot\Gamma(p) - \dot\Gamma(0))/p^2$ usually do not.
- `"RealOutput" -> True, "ComplexEndProjection" -> True`: for a chemical potential, which makes the integrand
  complex while the flow stays real. See `"RealOutput"` in the NumTracer documentation before using it.

## The model class

The dressings live on a stack of Matsubara frequencies, each with the same logarithmic momentum grid:

```cpp
using Coordinates = BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>;
using Interpolator = SplineInterpolator1DStack<double, Coordinates>;

using VariableDesc = VariableDescriptor<FunctionND<"ZPi", n_stack>, FunctionND<"ZSig", n_stack>, Scalar<"Zbar">>;
```

`coordinates(0, n_matsubara, T, p_grid_size, p_min, p_max, bias)` covers the frequencies $2\pi T n$,
$n = 0, \dots, 3$. Entry `m * p_grid_size + i` of a variable is the value at frequency $m$ and momentum $i$.

**$\eta$.** `solve_eta` evaluates the group at $(0, p_\text{min})$ with $\eta = 0$ and $\eta = 1$ and solves the
linear condition. `device::tie` stores references, so the second call sees the changed `eta`:

```cpp
double eta = 0.;
const auto arguments = device::tie(k, prm.T, hPhi, eta, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
MultiValue<double, 2> g0, g1;
flow_equations.Mesons.get(g0, 0., prm.p_grid_min, arguments);
eta = 1.;
flow_equations.Mesons.get(g1, 0., prm.p_grid_min, arguments);
return -g0[0] / (g1[0] - g0[0]);
```

`extract` calls it at the EoM, for the potential flow; `dt_variables` calls it again at the state it is evaluated at
and then maps the group over the whole grid.

**The potential.** `evaluate_batch` computes the flux with one `map_points()` call and the rescaling as the source:

```cpp
if (out.requested(Term::flux)) {
  const NT d3V(0.);
  flow_equations.V.map_points(out.flux(idxf("u"), 0), k, prm.T, hPhi, etaPi, rho, u, du, d3V, ZPi, ZSig);
}
if (out.requested(Term::source)) {
  const auto s = out.source(idxf("u"));
  for (size_t i = 0; i < batch.size(); ++i)
    s[i] = etaPi * (u[i] + rho[i] * du[i]);
}
```

`rho`, `u` and `du` are columns of the batch, one value per point; the other arguments are shared. The jacobian
comes from the AD overloads of `map_points`, so the model derives from `def::FE_AD` and needs no `flux()`.

## Building and running

```bash
$ cd Tutorials/tut9
$ ./build.sh -j4
$ cd build
$ ./tut9
```

The first build runs NumTracer's trace programs for every flow and takes a few minutes; on a laptop with 16 cores
and an RTX 4070 the whole build took 3.7 minutes with `-j4`. The run takes 22 s on the same machine. At
$k \approx 12$ MeV, with $\Lambda = 650$ MeV and $T = 100$ MeV:

| $\sigma$ | $m_\pi$ | $m_\sigma$ | $m_q$ | $Z_\sigma(0, p_\text{min})$ | $Z_\pi(2\pi T, p_\text{min})$ |
| --- | --- | --- | --- | --- | --- |
| 91.2 | 153 | 371 | 254 | 1.34 | 1.11 |

(masses in MeV; $m_\sigma$ is the curvature mass divided by $Z_\sigma(0, p_\text{min})$). $Z_\pi(2\pi T,
p_\text{min}) \ne 1$ is the difference between temporal and spatial momenta in the heat bath, which a single
momentum-independent wave function cannot describe. $\eta$ starts at 1.4 and vanishes in the IR.

### The benchmark

```bash
$ cmake -DTUT9_BENCH=ON .
$ make bench
$ ./bench -sd /physical/T=0.01
```

`bench` evaluates every flow of `bench_flows/` once against its counterpart in `flows/`, at fixed, smooth inputs, and
prints the times and the largest relative difference. On the RTX 4070:

| | T = 100 MeV | T = 10 MeV |
| --- | --- | --- |
| `Mesons`: group, hoisted | 8.6 ms | 22.6 ms |
| `ZPi` + `ZSig`: two kernels | 13.7 ms | 36.3 ms |
| `MesonsPlain`: group, `"Hoist" -> False` | 11.9 ms | 31.4 ms |
| `V`: `"PointArguments"` with `rhoPhi`, 400 points | 0.25 ms | 0.64 ms |
| `VAuto`: default `"PointArguments"` | 2.4 ms | 5.3 ms |

The variants differ from each other by less than $10^{-13}$. This GPU computes in double precision at 1/64 of its
single-precision rate, so it gains from anything that saves arithmetic. The relative gains on an A100 or H100, where
double precision is fast, can be different; measure them there with `bench` before relying on these numbers.

## Regenerating the flows

```bash
$ cd Tutorials/tut9
$ wolfram -script QM.wl
```

regenerates `flows/` and `bench_flows/`; the next build runs the trace programs again. `NT_FLOWS=Mesons,V` restricts
the run to the named flows and skips the final `UpdateNTFlows`, so run it once without `NT_FLOWS` after adding or
removing a flow.

## What to take away

* `MakeNTKernelDiFfRG` takes `MakeKernel`'s description of the integral; finish a derivation with `UpdateNTFlows`.
  The traces are computed when the flows library is first built.
* Projections of one loop belong in one `MakeNTKernelGroupDiFfRG` group, and interpolators on one grid in one
  `"SharedGrids"` entry.
* Hoisting is on by default. For a kernel evaluated with `map_points()`, add every plain parameter that varies per
  point to `"PointArguments"`; forgetting one costs speed, not correctness.
* Measure the options on the GPU you run on.
