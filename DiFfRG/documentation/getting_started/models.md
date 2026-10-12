(Models)=
# Numerical models

In DiFfRG, the system of equations to solve is described by a *numerical model*: a class that defines the
unknowns, their initial conditions and their flow, plus what to read off and write out along the way. This page
explains the concepts; the complete list of hooks a model can define, with their defaults, is in the
{doc}`model.hh reference </doxygen/DiFfRG/html/model_8hh>`.

## Unknowns: FE functions, variables, extractors

A model has three kinds of components:

- **FE functions** $u_i(x)$, $i = 0,\dots,N_f-1$, functions on a discretized field space $x\in\mathbb{R}^d$. In a
  derivative expansion they are the field-dependent coefficients, e.g. $u = \partial_\rho V(\rho)$ with $x = \rho$.
- **Variables** $v_a$, $a = 0,\dots,N_v-1$, unknowns without field dependence: couplings, wave-function
  renormalizations, or whole dressing functions tabulated on a momentum grid.
- **Extractors** $e_b$, $b = 0,\dots,N_e-1$: not unknowns, but numbers computed from the FE functions at one point
  (by default the minimum of the potential). Through them, the variables and the FE equations themselves can
  depend on, e.g., the curvature mass at the minimum.

They are declared by name in a `ComponentDescriptor`:

```cpp
using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
using VariableDesc   = VariableDescriptor<Scalar<"Z">, FunctionND<"ZA", 96>>;   // a scalar and a grid of 96 values
using ExtractorDesc  = ExtractorDescriptor<Scalar<"eta">>;
using Components     = ComponentDescriptor<FEFunctionDesc, VariableDesc, ExtractorDesc>;

constexpr auto idxf = FEFunctionDesc{};   // idxf("v") == 1, resolved at compile time
constexpr auto idxv = VariableDesc{};     // idxv("ZA") + i indexes grid point i of ZA
constexpr auto idxe = ExtractorDesc{};
```

A system without FE functions uses an empty `FEFunctionDescriptor<>`, see [Tutorial 4](../tutorials/tut4.md).

## The equations

For the FE functions, every field-space discretization solves

```{math}
m_i(u, \partial_t u, x) + \nabla_x\cdot F_i(u, \nabla u, \nabla^2 u, e, v, x) + s_i(u, \nabla u, \nabla^2 u, e, v, x) = 0\,,
```

with the *mass* $m_i$ (default $m_i = \partial_t u_i$), the *flux* $F_i$ and the *source* $s_i$. The variables obey

```{math}
\partial_t v_a + D_a(v, e) = 0\,.
```

Here $t = \ln(\Lambda/k)$ is positive and grows towards the IR, the opposite sign of the common convention
$t_- = \ln(k/\Lambda)$, so $\partial_{t} u = -\partial_{t_-} u$. A flow written in $t_-$ can therefore be copied
without sign changes: if $\partial_{t_-} u = \partial_x f + g$, set $F = f$ and $s = g$. For $u = \partial_\rho V$,
the flux is simply the flow $\partial_{t_-} V$ of the potential.

A component whose mass term does not contain $\partial_t u_i$ is an algebraic equation. Only the SUNDIALS IDA
timesteppers can solve such systems; see [Timesteppers](timesteppers.md).

## Defining a model

A model derives from `def::AbstractModel` (the model passes itself as template argument) and from a few building
blocks in `def::` that supply defaults:

```cpp
class MyModel : public def::AbstractModel<MyModel, Components>,
                public def::fRG,                     // RG time: members t, k, k2, ..., updated from t
                public def::FlowBoundaries<MyModel>, // inflow/outflow boundary conditions (CG, DG, LDG)
                public def::AD<MyModel>              // Jacobians by automatic differentiation
{
public:
  static constexpr uint dim = 1;

  MyModel(const ConfigTree &json) : def::fRG(json), prm(json) {}

  template <typename Vector> void initial_condition(const Point<dim> &x, Vector &u_i) const
  {
    u_i[idxf("u")] = ...;
    u_i[idxf("v")] = ...;
  }

  template <typename NT, typename Solution>
  void flux(std::array<Tensor<1, dim, NT>, Components::count_fe_functions(0)> &F_i, const Point<dim> &x,
            const Solution &sol) const
  {
    const auto &u = get<"fe_functions">(sol);
    const auto &du = get<"fe_derivatives">(sol);
    F_i[idxf("u")][0] = ...;
    F_i[idxf("v")][0] = ...;
  }

  template <typename NT, typename Solution>
  void source(std::array<NT, Components::count_fe_functions(0)> &s_i, const Point<dim> &x, const Solution &sol) const
  {
    s_i[idxf("u")] = ...;
  }

private:
  const Parameters prm;
};
```

Every hook that is not defined keeps the default of `def::AbstractModel`, which is zero for flux, source and
variable flows. The number-type template parameter matters: with `def::AD`, the assembler calls `flux` and `source`
with automatic-differentiation types `NT` to obtain the Jacobian, so the code must not assume `double`.

The building blocks; a model picks one per row as needed:

| purpose | classes |
| --- | --- |
| RG time | `def::fRG` ($k = \Lambda e^{-t}$; members `t`, `k`, `k2`, …) or `def::Time` (`t` only) |
| Jacobians | `def::AD` (all by AD), `def::FE_AD` (FE part by AD, extractors frozen), `def::NoJacobians` (explicit steppers only) |
| interior numerical flux (DG, LDG) | `def::LLFFlux` (local Lax–Friedrichs), `def::NoNumFlux` |
| boundaries (CG, DG, LDG) | `def::FlowBoundaries` |
| boundaries (finite volume) | `def::FVDefaultBoundaries`, `def::RhoSymmetricLinearExtrapolationBoundaries`, … in {doc}`fv_boundaries.hh </doxygen/DiFfRG/html/fv__boundaries_8hh>` |
| pinned dofs | `def::ConstrainOriginBoundaryPointToZero<"u", Model>`, … (see [below](#affine-constraints)) |

Which hooks each discretization needs:

- **CG**: `flux`, `source`, boundary conditions.
- **DG, LDG**: additionally an interior numerical flux, e.g. `def::LLFFlux`. LDG builds auxiliary levels (e.g.
  $\partial_x u$) from `ldg_flux` / `ldg_source`, for equations with higher derivatives.
- **Finite volume (Kurganov–Tadmor)**: `flux` is the advective part only; the part that depends on $\nabla u$
  goes into `diffusion_flux`. Boundaries come from the finite-volume ghost-cell mixins.

[Tutorial 7](../tutorials/tut7.md) solves one model with all of them.

## What `sol` contains

`flux` and `source` receive the solution at the evaluation point as a tuple whose entries are addressed by name,
e.g. `get<"fe_derivatives">(sol)[idxf("u")][0]` for $\partial_{x_0} u$. Which entries exist depends on the
assembler:

| assembler | entries of `sol` in `flux` and `source` |
| --- | --- |
| CG, DG | `"fe_functions"`, `"fe_derivatives"`, `"fe_hessians"`, `"extractors"`, `"variables"`, `"cell_width"` |
| LDG main level | `"fe_functions"`, `"LDG1"`, `"LDG2"`, …, `"extractors"`, `"variables"`, `"cell_width"` |
| KT finite volume | `"fe_functions"`, `"fe_derivatives"`, `"extractors"`, `"variables"`, `"cell_width"` |

- Under KT, `"fe_derivatives"` is the scheme's reconstructed (limited) cell gradient; `diffusion_flux`
  additionally gets `"fe_third_derivatives"`, and `source` gets `"fe_hessians"` if the model declares
  `static constexpr bool source_uses_hessians = true`.
- Under CG and DG, a model declaring `static constexpr bool batch_reads_hessians = false` (or
  `batch_reads_derivatives = false`) gets no `"fe_hessians"` (or neither), and the assembler skips computing them.
- Reading an entry the assembler does not provide is a compile error. A model shared between assemblers can test
  `if constexpr (tuple_has<"fe_derivatives", Solution>)`.

The assemblers do not call `flux` and `source` one point at a time: they call the model's `evaluate_batch` for
all points at once, whose default calls the per-point hooks in a parallel loop. Overriding it lets a model
evaluate its momentum integrals for all points in one call, also on the GPU; see [Tutorial 6](../tutorials/tut6.md).

## Variables and extractors

```cpp
template <typename Vector> void initial_condition_variables(Vector &v_a) const;

// fills the residual r_a = -dt v_a; sol holds "variables" and "extractors"
template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &sol) const;

// called at the extractor point x; sol is as in flux()
template <typename NT, typename Solution>
void extract(std::array<NT, Components::count_extractors()> &e_b, const Point<dim> &x, const Solution &sol) const;
```

The extractor point is the root of the model's `EoM(x, u)` (default: the first FE function, i.e. for
$u = \partial_\rho V$ the minimum of the potential), optionally moved by `EoM_postprocess` or chosen from the whole
solution by `extractor_point`; see [Tutorial 5](../tutorials/tut5.md). `extract` runs before every residual and
Jacobian evaluation. With `def::AD` under the FEM assemblers, the Jacobian includes how the extractors depend on
the FE solution. With `def::FE_AD`, and always under KT, the extractors are frozen within a Newton step; this
costs Newton iterations, not accuracy.

A pure-variable system is assembled by `Variables::Assembler`; when FE functions are present, the FEM assembler
handles both sectors. [Tutorial 4](../tutorials/tut4.md) (variables only), [Tutorial 5](../tutorials/tut5.md)
(an FE potential driving a variable through extractors) and [Tutorial 8](../tutorials/tut8.md) (an FE potential
and momentum-dependent couplings feeding each other) are complete examples.

## Readouts

`readouts(output, x, sol)` is called once per output frame at the EoM point and writes observables, see
[Output](output.md#writing-data-from-a-model). `readouts_multiple` registers several readouts, each with its own
EoM.

## The reconstructed potential

FE functions usually describe derivatives of a potential, $u = \partial_\rho V$. For output, readouts and
extractors, DiFfRG reconstructs the potential itself, normalized to $V(0) = 0$, from the gradient given by
`raw_potential_gradient(x, u)` (default: the first `dim` FE functions). It is written to `<run>_potential.pvd`
and the HDF5 group `/potential`, and is available in readouts and extractors as the entries `"potential"`,
`"potential_gradient"` and `"potential_hessian"`; these entry names may still change. A model with a different
component layout overrides the gradient:

```cpp
template <int dim, typename Vector>
std::array<double, dim> raw_potential_gradient(const Point<dim> &, const Vector &u) const
{
  return {{u[idxf("u")]}};
}
```

The reconstruction is a solve over the whole mesh. A model whose extractors do not read the potential should
declare `static constexpr bool extract_uses_potential = false;` to skip it on every residual evaluation.

## Affine constraints

Sometimes a model should pin individual degrees of freedom instead of expressing a condition through fluxes or
boundaries, e.g. imposing $u(0) = 0$ or removing a zero mode. Before building their sparsity patterns, the
assemblers ask the model for such constraints through two hooks:

- `apply_boundary_affine_constraints(constraints, context)` for dofs on boundary faces,
- `apply_affine_constraints(constraints, context)` for any support dof, including interior ones.

`context` gives the dofs and support points of each FE function by name:

```cpp
const auto u_support = context.template support<"u">();    // all support dofs of u
const auto v_boundary = context.template boundary<"v">();  // only the boundary dofs of v
```

Each view has `.dofs` (an `IndexSet`) and `.points`, aligned so that `view.points[i]` belongs to
`view.dofs.nth_index_in_set(i)`.

### Ready-made: pin a component at the origin

```cpp
class MyModel : public def::AbstractModel<MyModel, Components>,
                public def::ConstrainOriginBoundaryPointToZero<"u", MyModel>,
                ...
```

`ConstrainOriginBoundaryPointToZero` constrains the boundary dof nearest to the origin to zero;
`ConstrainOriginSupportPointToZero` does the same for any support point, which is what finite-volume and DG0-like
layouts need. Symmetric ties are broken towards the non-negative side.

In more than one dimension, the model defines what "origin" means per component by an
`OriginConstraintCoordinate` policy; all dofs on the nearest discrete zero set of that coordinate are constrained.
For a two-field model where `"u"` is odd across $\phi_1 = 0$ and `"v"` across $\phi_2 = 0$:

```cpp
class O2Model : public def::AbstractModel<O2Model, Components>, ...
{
public:
  template <FixedString component_name> struct OriginConstraintCoordinate;

  template <typename Constraints, typename Context>
  void apply_affine_constraints(Constraints &constraints, const Context &context) const
  {
    def::ConstrainOriginSupportPointToZero<"u", O2Model>{}.apply_affine_constraints(constraints, context);
    def::ConstrainOriginSupportPointToZero<"v", O2Model>{}.apply_affine_constraints(constraints, context);
  }
};

template <> struct O2Model::OriginConstraintCoordinate<"u"> {
  static double signed_coordinate(const Point<2> &point) { return point[0]; }
};
template <> struct O2Model::OriginConstraintCoordinate<"v"> {
  static double signed_coordinate(const Point<2> &point) { return point[1]; }
};
```

### Custom constraints

```cpp
template <typename Constraints, typename Context>
void apply_boundary_affine_constraints(Constraints &constraints, const Context &context) const
{
  const auto v = context.template boundary<"v">();
  // v(0) = 1 on the boundary dof at the origin
  for (uint i = 0; i < v.dofs.n_elements(); ++i) {
    if (std::abs(v.points[i][0]) > 1e-12) continue;
    const auto dof = v.dofs.nth_index_in_set(i);
    constraints.add_line(dof);
    constraints.set_inhomogeneity(dof, 1.0);
    break;
  }
}
```

Affine constraints are for pinning a few named dofs. Boundary conditions that are naturally expressed through
numerical fluxes or finite-volume ghost cells belong there instead.

## Further reading

- The numerics behind the assemblers: the [deal.II tutorials](https://www.dealii.org/developer/doxygen/deal.II/Tutorial.html)
  for finite elements, and e.g. [this review](https://www3.nd.edu/~zxu2/acms60790S15/DG-general-approach.pdf) for
  discontinuous Galerkin methods.
- Derivative expansions solved with these methods: [arXiv:2305.00816](https://arxiv.org/abs/2305.00816).
- The API of the assemblers: {doc}`CG </doxygen/DiFfRG/html/classDiFfRG_1_1CG_1_1Assembler>`,
  {doc}`DG </doxygen/DiFfRG/html/classDiFfRG_1_1DG_1_1Assembler>`,
  {doc}`LDG </doxygen/DiFfRG/html/classDiFfRG_1_1LDG_1_1Assembler>`.
