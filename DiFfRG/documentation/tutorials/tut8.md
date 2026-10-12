(tut8)=
# Tutorial 8: Quark-meson model in LPA′

The earlier tutorials flowed either a field-dependent potential (Tutorials 1–3, 5, 7) or momentum-dependent
couplings (Tutorial 4). This tutorial couples the two. The quark-meson model in LPA′ has an effective potential on
a field-space grid and, flowing alongside it, a wave-function renormalization and a Yukawa coupling, and both feed
on each other at every step. It is also the first tutorial with fermions, at finite temperature and chemical
potential.

The code is in `Tutorials/tut8`; `Examples/QuarkMesonLPAprime` is the same model. The derivation is `QM.wl`, the
generated flows are shipped in `flows/`, so the tutorial builds without Mathematica.

## The model

The fields are $N_f = 2$ flavors of quarks $q$ with $N_c$ colors and the mesons $\phi = (\sigma, \vec\pi)$, with
$\rho = \phi^2/2$. The effective action in LPA′ is

```{math}
\Gamma_k = \int_x \Big[ Z_q\, \bar q\, \gamma_\mu \partial_\mu q
  + h\, \bar q \big(T^0 \sigma + i\gamma_5\, T^a \pi^a\big) q
  + \frac{Z_\phi}{2} (\partial_\mu \phi)^2 + V_k(\rho) - c\,\sigma \Big] ,
```

at temperature $T$ and quark chemical potential $\mu$, which shifts the quark frequencies, $p_0 \to p_0 + i\mu$.
$T^a$ are the generators of $SU(N_f)$ with
$\operatorname{tr} T^a T^b = \delta^{ab}/2$, and $T^0 = \mathbb{1}/\sqrt{2N_f}$ has the same normalization; for
$N_f = 2$ the Yukawa term is $\frac{h}{2}\,\bar q(\sigma + i\gamma_5\,\vec\tau\cdot\vec\pi)q$. In a constant
background $\sigma$ the quarks have the mass $m_q = h\sigma/\sqrt{2N_f}$. The term $-c\sigma$ breaks chiral
symmetry explicitly and gives the pions a mass.

The wave-function renormalizations $Z_q$ and $Z_\phi$ are not used directly. All fields are renormalized,
$\bar\phi = Z_\phi^{1/2}\phi$ and $\bar q = Z_q^{1/2} q$, so that their kinetic terms keep unit coefficients, and
the $Z$ only enter through the anomalous dimensions

```{math}
\eta_\phi = -\frac{k\,\partial_k Z_\phi}{Z_\phi}\,, \qquad \eta_q = -\frac{k\,\partial_k Z_q}{Z_q}\,.
```

They appear in three places:

- the regulator insertions, $k\partial_k (Z R_k) = Z\,(k\partial_k R_k - \eta R_k)$;
- the rescaling of every coupling with the field it multiplies: the potential, $u = \partial_\rho V$, gets
  $\eta_\phi\,(u + \rho\,\partial_\rho u)$, and the Yukawa coupling gets $(\eta_\phi/2 + \eta_q)\,h$;
- the explicit breaking, which is fixed for the bare field: $\bar c = Z_\phi^{-1/2} c$. This is the only reason to
  keep $Z_\phi$ itself.

The anomalous dimensions are defined by the momentum dependence of the two-point functions at vanishing momentum:
$\eta_\phi$ from the $\vec p^{\,2}$ coefficient of the pion two-point function at $p_0 = 0$, and $\eta_q$ from the
$i\vec\gamma\cdot\vec p$ coefficient of the quark two-point function at the lowest fermionic Matsubara frequency.
The spatial direction is the natural choice at finite $T$, where the temporal direction is discrete. The
regulators are spatial too ("3d" regulators), so that the Matsubara sums can be done analytically.

The system to solve is therefore

- the PDE for $u(\rho)$, discretized with CG as in Tutorial 1;
- the ODEs for $Z_\phi$ and $h$, as Variables (Tutorial 4);
- $\eta_\phi$, $\eta_q$ and the derivatives of $V$ at the minimum, which all of the above need, as extractors
  (Tutorial 5).

## The derivation file

`QM.wl` follows Tutorial 3; this section only covers what is new. Run it headless with `wolfram -script QM.wl`.

**Fields and truncation.** Quarks are declared as a Grassmann pair, mesons as commuting fields:

```mathematica
fields = <|"Commuting" -> {σ[p], Π[p, {a}]},
           "Grassmann" -> {{qb[p, {d, c, f}], q[p, {d, c, f}]}}|>;
```

The truncation lists the quark and meson two-point functions, the two Yukawa vertices and the meson vertices up to
four legs. The meson vertices are derivatives of $V$ at the minimum, e.g. $\Gamma_{\sigma\sigma\sigma} =
3\sigma\,V'' + \sigma^3 V'''$, written in terms of the kernel parameters `d1V`, `d2V`, `d3V` and `rhoPhi`.

**Sanity checks.** For every propagator, `QM.wl` multiplies $\Gamma^{(2)} + R$ with the propagator and traces the
product; the result must be the trace of the unit matrix, up to FunKit's sign convention for fermions ($-4 N_c N_f$
for the quarks). The projector of each coupling is checked the same way on the tree-level vertex. They catch most
errors in the dressing rules, and are worth keeping in your own derivations.

**Momentum routing of fermion bases.** TensorBases writes every basis element in all-incoming momenta. The
momentum-odd element $i\vec\gamma\cdot\vec p_1$ of the quark two-point basis is built on the antiquark leg $p_1$,
while the dressings are written in terms of the quark momentum $p_2 = -p_1$. The spatial dressings of $\Gamma$,
$R$ and $\dot R$ therefore carry an explicit minus sign:

```mathematica
dressing[GammaN, {qb, q}, 2, {p1_, p2_}] :> -1,
dressing[R, {qb, q}, 2, {p1_, p2_}] :> -RQ[p2]/Sqrt[sps[p2, p2]],
```

A wrong sign here goes unnoticed until the sanity check of the quark propagator fails.

**Yukawa normalization.** The pion basis element is $i\gamma_5 T^a$, the sigma one the unit matrix. To match the
action, the sigma vertex carries the factor $T^0$ in its dressing:

```mathematica
dressing[GammaN, {σ, qb, q}, 1, {p1_, p2_, p3_}] :> hPhi/Sqrt[2 Nf],
dressing[GammaN, {Π, qb, q}, 1, {p1_, p2_, p3_}] :> hPhi
```

Two checks fix it: the sigma vertex is the $\sigma$-derivative of the quark mass term, and at $\rho = 0$ the quark
loops in the sigma and pion two-point functions agree, as chiral symmetry requires.

**Grassmann ordering of projectors.** The flow of the quark two-point function is taken as
`FTakeDerivatives[WetterichEquation, {qb[i1], q[i2]}]`, but the projector expects the quark index first. Swapping
the two Grassmann legs costs a sign:

```mathematica
traceExprEtaq = FTerm[-TBGetProjector["qbqFT", 2, {i2, i1} /. fRGqbq["1-Loop"]["ExternalIndices"]]] ** (...);
```

The Yukawa flow projects onto the unit matrix with the legs in derivative order, `{i1, i2}`, so there is no sign.

**External momenta.** The external frequency of the quark legs is the lowest Matsubara frequency, shifted by the
chemical potential, `p0 -> p0f - I muq`. With this shift the results do not depend on $\mu$ at $T \to 0$ for
$\mu$ below the quark mass (the Silver Blaze property), which is a good check of a finite-density derivation.

**Kernels.** The potential, `V`, is a three-dimensional loop integral with the Matsubara sum done analytically
(`Integrator_p2`). `etaQ`, `etaPhi` and `hPhi0` keep the external momentum and sum numerically
(`Integrator_fT_p2_1ang`). The Yukawa kernel carries its rescaling term as the constant part of `MakeKernel`:

```mathematica
MakeKernel[Re[FlowPhiqbq], (etaPhi/2 + etaQ) hPhi, kernelhPhi]
```

All kernels take the same parameter list: the scale, the external momentum, $N_f$, $N_c$, $T$, $\mu$, the
anomalous dimensions, $h$ and the potential at the minimum.

## The model class

```cpp
using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
using VariableDesc = VariableDescriptor<Scalar<"ZPhi">, Scalar<"hPhi">>;
using ExtractorDesc = ExtractorDescriptor<Scalar<"etaQ">, Scalar<"etaPhi">, Scalar<"rhoPhi">, Scalar<"d1V">,
                                          Scalar<"d2V">, Scalar<"d3V">>;
```

The model derives from `def::FE_AD`, so the Jacobian of the FE part comes from automatic differentiation, with the
extractors held fixed.

### Extractors: self-consistent anomalous dimensions

`extract` is called at the EoM (see below). It stores the derivatives of $V$ there and solves for the anomalous
dimensions. Both appear in their own flows through $\dot R - \eta R$, so it iterates, starting from the values of
the previous evaluation:

```cpp
do {
  double new_etaQ = 0., new_etaPhi = 0.;
  flow_equations.etaQ.get(new_etaQ, arguments);
  flow_equations.etaPhi.get(new_etaPhi, arguments);
  eps = std::max(std::abs((new_etaQ - etaQ) / new_etaQ), std::abs((new_etaPhi - etaPhi) / new_etaPhi));
  etaQ = new_etaQ;
  etaPhi = new_etaPhi;
} while (eps > 1e-5 && ++it < 8);
```

`arguments` is a `device::tie` of the kernel parameters. `device::tie` stores references, so every argument must be
a named variable; temporaries like `variables[idxv("hPhi")]` do not compile.

### Variables

```cpp
residual[idxv("ZPhi")] = -etaPhi * ZPhi;
flow_equations.hPhi0.get(residual[idxv("hPhi")], arguments);
cSigma = std::pow(ZPhi, -0.5) * prm.cSigma;
```

The residual convention is $\partial_t v = -\text{residual}$ with $t = \ln(\Lambda/k)$, so the first line is
the definition of $\eta_\phi$. The last line updates the renormalized explicit breaking, which the EoM uses.

### Potential

`flux` evaluates the `V` kernel; its $\rho$-derivative is the loop part of the flow of $u$. `source` adds the
rescaling $\eta_\phi(u + \rho\,\partial_\rho u)$.

### Two EoMs and two readouts

With explicit breaking, the physical minimum is that of $V(\rho) - c\sigma$, i.e. $u = c/\sigma$:

```cpp
template <int dim_, typename Vector> std::array<double, 1> EoM(const Point<dim_> &x, const Vector &u_i) const
{
  const double sigma = std::sqrt(2. * x[0]);
  return {{u_i[idxf("u")] - cSigma / (sigma + 1e-14)}};
}
```

The extractors, and therefore the flow, are evaluated there. `readouts_multiple` writes two tables: `data.csv` at
the physical minimum and `data_chiral.csv` at the minimum of $V$ alone, the chiral-limit value of the condensate
for the same flow. `raw_potential_gradient` tells the potential output to reconstruct $V$ without the $-c\sigma$
term.

$\bar c$ depends on $Z_\phi$, which is a Variable, but the EoM only sees the FE functions; the model caches $\bar c$
in `dt_variables`. That makes it history the snapshot does not hold, so `save_state` and `load_state` carry it
across a restart (see [Snapshots](../getting_started/snapshots.md)).

## Timestepping

The kernels with an external momentum are the expensive part. With a fully implicit stepper they would run in every
Newton iteration. `TimeStepperSUNDIALS_IDA_BoostABM` instead steps the potential implicitly with IDA and the
Variables explicitly with an Adams–Bashforth–Moulton method, which evaluates `dt_variables` once per step:

```cpp
using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<Assembler>;
```

Each part has its own tolerances, under `[timestepping.implicit]` and `[timestepping.explicit]`; see
[Timesteppers](../getting_started/timesteppers.md).

## Building and running

```bash
$ cd Tutorials/tut8
$ ./build.sh -j4
$ cd build
$ ./tut8
```

The run takes about 20 s on 32 CPU threads. The default parameters are $\Lambda = 650$ MeV, $T = \mu = 100$ MeV.
In the IR, at $k \approx 12$ MeV:

| | $\sigma$ | $m_\pi$ | $m_\sigma$ | $m_q$ | $h$ | $Z_\phi$ |
| --- | --- | --- | --- | --- | --- | --- |
| $T = \mu = 100$ MeV | 84.6 | 162 | 423 | 250 | 5.90 | 1.26 |
| $T = 5$ MeV, $\mu = 0$ | 103.0 | 141 | 491 | 287 | 5.57 | 1.50 |

(masses in MeV). $\eta_q$ starts at about 0.1 and vanishes once $k$ drops below the quark mass: gapped modes stop
contributing. Run the second line with

```bash
$ ./tut8 -sd /physical/T=0.005 -sd /physical/muq=0 -ss /output/name=vacuum
```

and again with `-sd /physical/muq=0.1`: all columns agree to six digits, as Silver Blaze requires.

## Limitations

At low temperature, the results do not depend on $\mu$ as long as $\mu$ stays below the quark mass ($287$ MeV at
$T = 5$ MeV). Above it a Fermi surface forms, and at $\mu = 300$ MeV the flow breaks down around $k \approx 180$
MeV: $\eta_\phi$ turns large and negative, and the timestepper stops. A likely cause: at $T \to 0$ the Fermi surface is
sharp, and the quark contribution to $\eta_\phi$, a derivative at vanishing external momentum, does not resolve it.
That part of the phase diagram calls for momentum-dependent wave-function renormalizations.

`Examples/QuarkMesonLPAprime/phasediagram.ipynb` scans $T$ and $\mu$ with the `adaptive` Python package and plots
the condensate.

## What to take away

* FE functions, Variables and extractors combine freely: extractors carry the potential at the EoM into the
  momentum integrals, Variables carry couplings back into the flux.
* With renormalized fields, a wave-function renormalization enters only through its anomalous dimension: in the
  regulator insertions and in the rescaling of every coupling.
* Anomalous dimensions that appear in their own flow are solved by iteration in `extract`.
* In fermionic derivations, check the momentum routing of momentum-odd basis elements and the Grassmann sign of
  swapped projector indices; the propagator sanity check and Silver Blaze catch most mistakes.
* `TimeStepperSUNDIALS_IDA_BoostABM` keeps expensive Variable kernels out of the Newton iterations.
