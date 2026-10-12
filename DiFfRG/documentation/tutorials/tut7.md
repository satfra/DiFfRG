(tut7)=
# Tutorial 7: Discretizations and mesh adaptivity

The previous tutorials each picked one field-space discretization. This tutorial solves one model with three of
them, continuous Galerkin (CG), discontinuous Galerkin (DG) and the Kurganov–Tadmor finite-volume scheme (KT),
shows what each needs from the model, and turns on adaptive mesh refinement.

The code is in `Tutorials/tut7`. As in [Tutorial 5](tut5.md), the flow is written by hand, so there is no
Mathematica step.

## The model

We flow the effective potential of the $O(N)$ model in the local potential approximation, $d = 3$, with a Litim
regulator. The FE function is $u = \partial_\rho V$, $\rho = \phi^2/2$, and the curvature masses are
$m_\pi^2 = u$ and $m_\sigma^2 = u + 2\rho\,\partial_\rho u$. With the Litim regulator the loop integral is
elementary, and each mode of mass $m^2$ contributes

```{math}
\ell(m^2) = \frac{k^5}{6\pi^2\,(k^2 + m^2)}
```

to the flow of $V$. Differentiating in $\rho$ gives the flow of $u$ as a conservation law,

```{math}
\partial_t u = \partial_\rho\Big[(N-1)\,\ell(m_\pi^2) + \ell(m_\sigma^2)\Big]\,,
```

where, as in [Numerical models](../getting_started/models.md#the-equations), $t = \ln(\Lambda/k)$. The initial
condition is $u = m^2 + \lambda\rho/2$ with $m^2 < 0$, a potential with its minimum away from the origin. For the
parameters in `parameter.toml` ($N = 4$, $m^2 = -0.12$, $\lambda = 2$ in units of $\Lambda$), fluctuations shrink
the condensate from $\sigma = 0.49$ at $\Lambda$ to $\sigma \approx 0.313$ in the IR: the flow ends in the broken
phase. Inside the minimum the potential becomes flat, $u \to 0$, which is the familiar approach to convexity; this
flat region is where the discretizations differ.

## One model, two forms

The schemes need the right-hand side in different forms, so `model.hh` has a shared base and two model classes.

`ONCommon` holds what does not depend on the scheme: parameters, the RG scale from `def::fRG`, the initial
condition, the readouts and the loop $\ell$.

**CG and DG** take the whole flux in one function:

```cpp
class ON_FEM : public def::AbstractModel<ON_FEM, Components>,
               public ONCommon,
               public def::LLFFlux<ON_FEM>,
               public def::FlowBoundaries<ON_FEM>,
               public def::AD<ON_FEM>
{
  ...
  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> &x, const Solution &sol) const
  {
    const auto m2Pi = get<"fe_functions">(sol)[idxf("u")];
    const auto m2Sigma = m2Pi + 2. * x[0] * get<"fe_derivatives">(sol)[idxf("u")][0];
    F[idxf("u")][0] = (prm.N - 1.) * loop(m2Pi) + loop(m2Sigma);
  }
};
```

In DG the solution may jump between cells, so the flux across a cell face needs a rule that combines both sides:
the *numerical flux*. `def::LLFFlux` supplies the local Lax–Friedrichs flux, which averages the two sides and adds
dissipation proportional to the local wave speed. CG ignores it, so the same class serves both. `def::FlowBoundaries`
treats both ends of the $\rho$ interval as inflow/outflow boundaries.

**KT** splits the flux into an advective part, which may only depend on $u$, and a diffusive part, which may also
depend on $\partial_\rho u$:

```cpp
class ON_KT : public def::AbstractModel<ON_KT, Components>,
              public ONCommon,
              public def::RhoSymmetricLinearExtrapolationBoundaries<ON_KT>,
              public def::AD<ON_KT>
{
  ...
  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> &, const Solution &sol) const
  {
    F[idxf("u")][0] = (prm.N - 1.) * loop(get<"fe_functions">(sol)[idxf("u")]);
  }

  template <typename NT, typename Solution>
  void diffusion_flux(std::array<Tensor<1, dim, NT>, 1> &F, const Point<dim> &x, const Solution &sol) const
  {
    const auto m2Sigma = get<"fe_functions">(sol)[idxf("u")] + 2. * x[0] * get<"fe_derivatives">(sol)[idxf("u")][0];
    F[idxf("u")][0] = loop(m2Sigma);
  }
};
```

The Goldstone modes give the advection, the radial mode the diffusion. KT stores one value per cell
and handles boundaries with ghost cells; `def::RhoSymmetricLinearExtrapolationBoundaries` makes $u$ even in
$\sigma$ at $\rho = 0$ and extrapolates linearly at the outer boundary.

Both classes inherit `initial_condition` and `readouts` from two bases, `ONCommon` and `def::AbstractModel`, which
has defaults for them. The `using ONCommon::...;` lines say which one is meant.

## Choosing at run time

`tut7.cc` instantiates all three and picks one from `/tutorial/discretization`:

```cpp
using CGDiscretization = CG::Discretization<ON_FEM, RectangularMesh<1>>;
using DGDiscretization = DG::Discretization<ON_FEM, RectangularMesh<1>>;
using KTDiscretization = FV::Discretization<ON_KT, RectangularMesh<1>>;
using Reconstructor = def::TVDReconstructor<1, def::MinModLimiter, double>;

if (discretization == "cg") return run<ON_FEM, CGDiscretization, CG::Assembler<CGDiscretization>>(config);
if (discretization == "dg") return run<ON_FEM, DGDiscretization, DG::Assembler<DGDiscretization>>(config);
if (discretization == "kt")
  return run<ON_KT, KTDiscretization, FV::KurganovTadmor::Assembler<KTDiscretization, ON_KT, Reconstructor>>(config);
```

The KT assembler takes the reconstruction of face values from cell averages as a template argument: here a
second-order TVD reconstruction with a minmod slope limiter, which falls back to first order at kinks so the
scheme does not oscillate. The other limiters are listed on the {doc}`C++ API main page
</doxygen/DiFfRG/html/index>`.

The function `run` is the `main` of the earlier tutorials, with two differences: it constructs an `HAdaptivity`
and passes it to the timestepper, and it chooses the container for the initial condition, since finite-volume
solutions are stored differently (`FV::FlowingVariables` instead of `FE::FlowingVariables`).

Build and run all three:

```bash
$ cd Tutorials/tut7
$ ./build.sh -j4
$ cd build
$ ./tut7 -ss /tutorial/discretization=cg -ss /output/name=cg
$ ./tut7 -ss /tutorial/discretization=dg -ss /output/name=dg
$ ./tut7 -ss /tutorial/discretization=kt -ss /output/name=kt
```

Each takes about a second. The readouts in `<name>_data.csv` hold $\sigma$, $m_\pi^2$ and $m_\sigma^2$ at the
minimum for every output step.

## Comparing the schemes

The condensate at the end of the flow, $t = 6$, for three uniform grids (`fe_order = 2` for CG and DG; KT ignores
it) and an adaptive run, with the number of residual evaluations:

| grid | cells | CG $\sigma$ | DG $\sigma$ | KT $\sigma$ |
| --- | --- | --- | --- | --- |
| `0:2e-3:0.15` | 75 | 0.31186 (851) | 0.31187 (2272) | 0.31303 (912) |
| `0:1e-3:0.15` (default) | 150 | 0.31309 (855) | 0.31305 (2149) | 0.31361 (1071) |
| `0:2.5e-4:0.15` | 600 | 0.31272 (1021) | 0.31273 (1894) | 0.31225 (865) |
| `0:2e-3:0.15`, adaptive, `level = 2` | 128 / 128 / 112 | 0.31275 (1302) | 0.31273 (2160) | 0.31319 (3704) |

A run of CG with `fe_order = 3` on a grid 32 times finer than the coarsest gives $\sigma = 0.31267$. What the table
shows:

- **CG** is the cheapest and converges smoothly. It is the default choice when the solution stays smooth.
- **DG** agrees with CG at the same resolution but needs about twice the residual evaluations. Its strength is in
  solutions with jumps, which CG cannot represent.
- **KT** has one value per cell, so at equal cell count it resolves less, and it converges more slowly. It is the
  robust choice when the solution develops genuine shocks, as in [Tutorial 5](tut5.md): the limiter keeps the
  solution free of oscillations where higher-order methods would produce them.

## Mesh adaptivity

`HAdaptivity` adapts the mesh at regular RG-time intervals. For each cell $K$ of width $h_K$ it estimates how well
the mesh resolves the solution,

```{math}
\eta_K = \frac{h_K^2\,|\partial_\rho^2 u|_K}{\max u - \min u}\,,
```

which is about 8 times the deviation of $u$ from its linear interpolant on the cell, relative to the range of $u$.
It is small where $u$ is smooth on the scale of the cell and grows where a kink, a shock or a narrow extremum forms;
each refinement lowers it by a factor 4. CG and DG estimate the second derivative from the FE solution, KT from the
cell averages (KT supports adaptivity in one field dimension only). A model needs no code for this; it can define its
own `cell_indicator` (and for DG a `face_indicator`) instead, which then has to have a known scale, e.g. a relative
error estimate.

Cells with $\eta_K$ above `tolerance` are refined, cells below `tolerance / 8` are coarsened. A mesh that resolves
the solution is therefore kept, and refinement starts only where a structure becomes unresolved. Here that is the edge of
the flat inner region of the potential, where $|\partial_\rho^2 u|$ grows as the flat region forms. Two details keep
the cost of the mesh changes down:

- every change makes the timestepper resolve a short transient, more so the less resolved the region was. So each
  cell's $\eta_K$ is extrapolated to the next adaptation by its growth since the previous one (`predict`), and a
  forming kink is refined one adaptation earlier;
- inside the flat region $\partial_\rho^2 u = 0$, while its edge moves through it during the flow. Coarsening there
  would be undone soon after, so cells created by the last `coarsen_age` = 10 mesh changes are not coarsened.

The adaptivity section of the parameter file controls it:

```toml
[discretization.adaptivity]
level = 2              # up to 2 refinements of the initial cells; 0 switches adaptivity off
start_adapt_at = 0.0   # RG time of the first adaptation
adapt_dt = 0.2         # RG time between adaptations
tolerance = 3e-4       # refine above, coarsen below tolerance / 8
```

The other keys (`predict`, `coarsen_tolerance`, `coarsen_age`, and the older `marking = "fraction"`, which refines a
fixed fraction of the total indicator at every adaptation) are listed in [Parameters](../getting_started/parameters.md).

Run it with

```bash
$ ./tut7 -ss /tutorial/discretization=cg -ss /discretization/grid/x_grid=0:2e-3:0.15 \
         -si /discretization/adaptivity/level=2 -ss /output/name=cg_adaptive
```

The last row of the table above shows the result. Starting from the coarse grid, CG and DG refine around the edge of the
flat region, to 128 cells, and reach the accuracy of the 600-cell grid; KT, with one value per cell, stays less accurate.
In one field dimension a uniform grid fine enough everywhere is cheap, though, so here the adaptive runs need more
residual evaluations than the 600-cell grid. Adaptivity pays off when the interesting structure is localized and
the uniform alternative is large: in higher field dimensions, or when the structure moves over a large domain.
