(Models)=
# Numerical Models
In DiFfRG the computational setup is fully described by a ***numerical model***. 
A model essentially is a description of a large set of coupled differential equations and some additional methods to handle data output.

In general, we have three components to any flow, which are **FE functions** $ u_i(x),\,i\in\{0,\dots,N_f-1\} $, **variables** $ v_a,\,a\in\{0,\dots,N_v-1\} $ and **extractors** $ e_b,\,b\in\{0,\dots,N_e-1\} $. 

The latter two are just independent variables, whereas the FE functions depend additionaly on a field variable, $ u_i(x),\,x\in\mathbb{R}^d $. In other words, the FE functions explicitly live on a spatial discretization of the field space.

## Defining a model

Any model you define should at least be inherited from the abstract model class DiFfRG::def::AbstractModel to ensure that all necessary methods are at least defined to do nothing, i.e.
```cpp
using namespace DiFfRG;
class MyModel : public def::AbstractModel<MyModel>, ...
{
  ...
};
```
Inside the class we can now overwrite all methods from DiFfRG::def::AbstractModel in order to implement the right system of flow equations.
 
## Spatial discretization

The FE functions usually correspond to expansion coefficients in a derivative expansion. As an example consider a bosonic theory as in [this](https://arxiv.org/abs/2305.00816) paper: Treating a purely bosonic theory in first-order derivative expansion, the effective action is given by

```{math}
\large
  \Gamma_k[\phi] = \int_x \bigg(\frac{1}{2}Z(\rho)(\partial_\mu\phi)^2 + V(\rho) \bigg)\,,
```

where $ \rho = \phi^2 / 2 $.
A flowing reparametrization of the field $ \phi $ is being performed and is given by

```{math}
\large
  \dot\phi(x) = \frac{1}{2} \eta(\rho) \phi\,.
```

where we introduced the anomalous dimension $ \eta = \frac{\partial_{t_+} Z}{Z} $.

The flow is then fully parametrized in terms of FE functions

```{math}
\begin{aligned}\large
  u_1(x) &= m^2(\rho) = \partial_\rho V(\rho)\,, \\\large
  u_2(x) &= \eta(\rho)\,,
\end{aligned}
```

where we also chose the field $ x = \rho $. We see here that the FE functions live on a spatial discretization of the d-dimensional field space $ \mathbb{R}^d $.

With the above ansatz one can quickly compute flow equations from the Wetterich equation,

```{math}
\large
  k\partial_k \Gamma_k[\Phi] = \frac{1}{2}\text{Tr}\, G_{\alpha\beta}\,k\partial_k R^{\alpha\beta}\,.
```

We remark that the time 

```{math}
\large t = t_+ = \ln\left(\frac{\Lambda}{k}\right)\,,
```

 as used in DiFfRG, is opposite in sign to the RG-time as defined in most literature, $t_- = \ln\left(\frac{k}{\Lambda}\right)$. This is simply due to many time solvers not accepting negative time arguments.

In order to discretize the flow equations on a finite element space, the flow equations are expressed in the standard differential-algebraic form

```{math}
\large
  m_i(\partial_t u_j, u_j, x) + \partial_x F_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x) + s_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x) = 0\,,
```

where $ m_i $ are called the mass functions, $ F_i $ the fluxes and $ s_i $ the sources. The latter two are functions of the FE functions, their derivatives, the field variable, the variables and the extractors.

In principle, the above system of equations can contain both equations containing the time derivatives, i.e. differential components, and equations without time derivatives, i.e. algebraic components. In order to solve the resulting DAEs one is currently restricted to the **SUNDIALS IDA** solver, which is however highly efficient and actually recommended for most cases.

Alternatively, the restricted formulation, allowing only for differential components,

```{math}
\large
  m_{ij}(x) \partial_t u_j + \partial_x F_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x) + s_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x) = 0\,,
```

is used for all other provided ODE solvers, i.e. Runge-Kutta methods.

Note, that in the above definitions a change from $ t = t_+ \to t_- $ simply moves all terms onto the other side, i.e. when calculating the flow equations in the standard $t_-$, one can still copy and paste everything without changing signs if the mass functions are simply $ m_i = \partial_{t_+} u_i = - \partial_{t_-} u_i $ (as is default).

The above components of standard form have direct analogues in the abstract ***numerical model*** which must be reimplemented for any system of flow equations. For actual implementation examples, especially regarding the template structure, please take a look at the models contained in the `DiFfRG/models/` folder.

The relevant methods are also documented in DiFfRG::def::AbstractModel and read as follows:

- The mass function $m_i(\partial_t u_j, u_j, x)$ is implemented in the method 
```cpp
  template <int dim, typename NumberType, typename Vector, typename Vector_dot, size_t n_fe_functions>
  void mass(std::array<NumberType, n_fe_functions> &m_i, const Point<dim> &x, const Vector &u, const Vector_dot &u_dot) const;
```  
Note, that the precise template structure is not important, the only important thing is that the types are consistent with the rest of the model. It is however necessary to leave at least the NumberType, Vector, and Vector_dot template parameters, as these can differ between calls.  
The `m_i` argument is the resulting mass function $m_i$, with $N_f$ components. This method should fill the `m_i` argument with the desired structure of the flow equation. `x` is a d-dimensional array of field coordinates, and both `u` (~$u_i(x)$) and `u_dot` (~$\partial_t u_i(x)$) have $N_f$ components.  
The standard implementation of this method simply sets $m_i = \partial_t u_i$.  
 
.
- If not using a DAE, the mass matrix $m_{ij}(x)$ is implemented in the method 
```cpp
  template <int dim, typename NumberType, size_t n_fe_functions>
  void mass(std::array<std::array<NumberType, n_fe_functions>, M::Components::count_fe_functions()> &m_ij, const Point<dim> &x) const;
```
The `m_ij` argument is the resulting mass matrix $m_{ij}$, with $N_f$ components in each dimension. This method should fill the `mass` argument with the desired structure of the flow equation. `x` is a d-dimensional array of field coordinates.
The standard implementation of this method simply sets $m_{ij} = \delta_{ij}$.

.
- The flux function $F_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x)$ is given by
```cpp
  template <int dim, typename NumberType, typename Solution, size_t n_fe_functions>
  void flux(std::array<Tensor<1, dim, NumberType>, n_fe_functions> &F_i, const Point<dim> &x, const Solution &sol) const;
```
Onve again, it is necessary to leave the `NumberType` and `Solution` templates, whereas the rest can be dropped.
`F_i` has $N_f$ components, `x` gives the coordinate in field space and `sol` contains all other arguments of the flux function. In practice, `sol` is a `std::tuple<...>` which contains
  0. the array u_j
  1. the array of arrays $\partial_x u_j$
  2. the array of arrays of arrays $\partial_x^2 u_j$
  3. the array of extractors $e_b$
Lastly, the variables are communicated separately to the model, see the ***Other variables***-section below
The standard implementation of this method simply sets $F_i = 0$.

.
- The source function $s_i(u_j, \partial_x u_j, \partial_x^2 u_j, e_b, v_a, x)$ is given by
```cpp
  template <int dim, typename NumberType, typename Solution, size_t n_fe_functions>
  void source(std::array<NumberType, n_fe_functions> &s_i, const Point<dim> &x, const Solution &sol) const;
```
Again, it is necessary to leave the `NumberType` and `Solution` templates, whereas the rest can be dropped.
`s_i` has $N_f$ components, `x` gives the coordinate in field space and `sol` contains all other arguments of the flux function, with the layout as explained above in the flux case.
The standard implementation of this method simply sets $s_i = 0$.

Which entries `sol` actually carries depends on the assembler, and the entries are addressed by name:

| assembler | entries of `sol` in `flux` and `source` |
| --- | --- |
| CG, DG | `"fe_functions"`, `"fe_derivatives"`, `"fe_hessians"`, `"extractors"`, `"variables"`, `"cell_width"` |
| LDG main level | `"fe_functions"`, `"LDG1"`, `"LDG2"`, ..., `"extractors"`, `"variables"`, `"cell_width"` |
| KT-FV | `"fe_functions"`, `"fe_derivatives"`, `"extractors"`, `"variables"`, `"cell_width"` |

KT-FV's `diffusion_flux` gets `"fe_third_derivatives"` after `"fe_derivatives"`, and its `source` gets
`"fe_hessians"` after `"fe_derivatives"` if the model declares `static constexpr bool source_uses_hessians = true`.
Under CG and DG, a model that declares `static constexpr bool batch_reads_hessians = false` (or
`batch_reads_derivatives = false`) gets no `"fe_hessians"` (or neither of the two): the assembler then skips
computing them.

A model that reads an entry its assembler does not provide fails to compile. A model shared between assemblers
can ask `if constexpr (tuple_has<"fe_derivatives", Solution>)`. For KT-FV, `"fe_derivatives"` is the scheme's own
reconstructed cell gradient (the limited slope at the cell centre).

`"extractors"` holds the result of the model's own `extract()`, run once before the residual and once
before the jacobian. Under the FEM assemblers with `def::AD`, the jacobian includes the extractors' dependence on
the FE solution (through `jacobian_extractors`). Under KT-FV, and with `def::FE_AD`, the extractors are frozen
within a Newton step: their dependence does not appear in the jacobian. A model that solves something at the EoM
in `extract()` and reads it back in `flux()` then gets an approximate jacobian: that costs Newton iterations,
not accuracy, because the extraction is redone for every residual.

The assemblers do not call `flux` and `source` one point at a time themselves: they call the model's
`evaluate_batch` for all points at once, whose default runs the per-point callbacks in a parallel loop. A model
that overrides it can evaluate its momentum integrals for all points in a single `map_points()` call, also on the
GPU; see [Tutorial 6](../tutorials/tut6.md).

Picking up the example from above, we can now sketch the implementation of the numerical model as follows:
```cpp
using namespace DiFfRG;

using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
using Components = ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};

class MyModel : public def::AbstractModel<MyModel, Components>,
                public def::fRG,                    // this handles the fRG time
                ...
{
public:
  MyModel(const ConfigTree& json) : def::fRG(json), prm(json) {}

  template <int dim, typename NumberType, typename Vector, typename Vector_dot>
  void mass(std::array<NumberType, Components::count_fe_functions()> &m_i, const Point<dim> &x, const Vector &u, const Vector_dot &u_dot) const
  {
    m_i[idxf("u")] = u_dot[idxf("u")];
    m_i[idxf("v")] = -u[idxf("v")];
  }

  template <int dim, typename NumberType, typename Solution>
  void flux(std::array<Tensor<1, dim, NumberType>, Components::count_fe_functions(0)> &F_i, const Point<dim> &x, const Solution &sol) const;
  {
    F_i[idxf("u")][0] = ...; // Flux of m^2
    F_i[idxf("v")][0] = ...; // Flux of eta
  }

  template <int dim, typename NumberType, typename Solution, typename M = Model>
  void source(std::array<NumberType, M::Components::count_fe_functions(0)> &s_i, const Point<dim> &x, const Solution &sol) const
  {
    s_i[idxf("u")] = ...; // Source of m^2
    s_i[idxf("v")] = ...; // Source of eta
  }
};
```

## Affine constraints

Sometimes a model should pin a few individual degrees of freedom directly, instead of expressing the condition through a
flux or a boundary stencil. Typical examples are:
- imposing `u(0)=0`,
- fixing several FE functions at the origin,
- selecting one representative in the presence of a zero mode.

This is done through **affine constraints**.

### When is this used?

Before the assemblers rebuild sparsity patterns and operators, they ask the model whether some dofs should be removed
from the linear algebra and prescribed to a fixed value. The model can then add lines to the deal.II
`AffineConstraints` object.

There are two model-side entry points:
- `apply_boundary_affine_constraints(constraints, context)` for constraints on dofs sitting on boundary faces,
- `apply_affine_constraints(constraints, context)` for constraints that may need any support dof, including interior
  support points.

The assemblers call the lower-level `affine_constraints(...)` hook with the context object. Models should usually
implement one of the two `apply_*` hooks above or inherit one of the helper mixins.

### What is `context`?

The second argument of the `apply_*_affine_constraints(...)` hooks is a small helper object that exposes the dofs and
support points of each FE function by its compile-time name.

For example, if the model uses
```Cpp
using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">, Scalar<"v">>;
using Components = ComponentDescriptor<FEFunctionDesc>;
constexpr auto idxf = FEFunctionDesc{};
```
then inside one of the affine-constraint hooks you can write
```Cpp
const auto u_support = context.template support<"u">();
const auto v_boundary = context.template boundary<"v">();
```

Each of these views has two members:
- `.dofs`: the `IndexSet` of the selected FE function,
- `.points`: the corresponding support points.

These two are aligned: `view.points[i]` is the point belonging to `view.dofs.nth_index_in_set(i)`.

There are two kinds of views:
- `support<"...">()` returns **all** support dofs of that FE function,
- `boundary<"...">()` returns only the **boundary** dofs of that FE function.

Use the view that matches the condition: boundary-face constraints should inspect `boundary<"...">()`, while
cell-centered or interior support-point constraints should inspect `support<"...">()`.

### Smallest useful example: constrain one component

For the common one-dimensional case `u(0)=0`, there are two ready-made helpers:
```Cpp
template <typename Model> using ConstrainBoundaryUAtOrigin = def::ConstrainOriginBoundaryPointToZero<"u", Model>;
template <typename Model> using ConstrainSupportUAtOrigin = def::ConstrainOriginSupportPointToZero<"u", Model>;

class MyModel : public def::AbstractModel<MyModel, Components>,
                public ConstrainBoundaryUAtOrigin<MyModel>,
                public def::fRG,
                ...
{
  ...
};
```

Both helpers do the following:
1. inspect the selected view of the FE function `"u"` and choose the representative nearest to the origin coordinate,
2. break symmetric ties by preferring the non-negative side,
3. constrain the selected dof or dofs to `0`.

Use `ConstrainOriginBoundaryPointToZero` for dofs sitting on boundary faces. Use
`ConstrainOriginSupportPointToZero` for FV and DG0-like layouts where the representative nearest the origin can be an
interior or cell-centered support point.

In one-dimensional domains, the origin coordinate is simply `x[0]`. In multidimensional domains, the model must define
what “origin” means for each constrained component by providing an `OriginConstraintCoordinate` policy. The helper then
selects the nearest discrete zero level set of that signed coordinate and constrains all dofs on that selected level set.

For example, in a two-field `O(2)`-style model where `"u"` is odd across `phi_1 = 0` and `"v"` is odd across
`phi_2 = 0`, write:
```Cpp
class O2Model : public def::AbstractModel<O2Model, Components>,
                ...
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

Here `signed_coordinate(point) == 0` defines the constraint manifold. This is intentionally model-owned: it avoids
assuming that component order determines geometry. A descriptor order such as
`FEFunctionDescriptor<Scalar<"v">, Scalar<"u">>` still works as long as the policies above define the intended
coordinates.

### Manual example: constrain several components differently

If different FE functions should be treated differently, write a custom
`apply_affine_constraints(...)` method:
```Cpp
class MyModel : public def::AbstractModel<MyModel, Components>,
                public def::fRG,
                ...
{
public:
  template <typename Constraints, typename Context>
  void apply_affine_constraints(Constraints &constraints, const Context &context) const
  {
    const auto u = context.template support<"u">();

    // Enforce u(0) = 0 using a support point.
    for (uint i = 0; i < u.dofs.n_elements(); ++i) {
      if (std::abs(u.points[i][0]) > 1.0e-12) continue;

      const auto dof = u.dofs.nth_index_in_set(i);
      constraints.add_line(dof);
      constraints.set_inhomogeneity(dof, 0.0);
      break;
    }
  }

  template <typename Constraints, typename Context>
  void apply_boundary_affine_constraints(Constraints &constraints, const Context &context) const
  {
    const auto v = context.template boundary<"v">();

    // Enforce v(0) = 1 using a boundary dof.
    for (uint i = 0; i < v.dofs.n_elements(); ++i) {
      if (std::abs(v.points[i][0]) > 1.0e-12) continue;

      const auto dof = v.dofs.nth_index_in_set(i);
      constraints.add_line(dof);
      constraints.set_inhomogeneity(dof, 1.0);
      break;
    }
  }
};
```

The important point is that the selection happens by **name**, not by guessing that `"u"` is component `0` and `"v"`
is component `1`.

### Reusing helpers for several named FE functions

If several FE functions should receive the same type of origin constraint, it is usually cleaner to wrap the provided
single-component helper into a small mixin. In multidimensional domains, the model still has to provide the
`OriginConstraintCoordinate` policy for each constrained component.
```Cpp
template <typename Model>
class ConstrainUAndVAtOrigin
  : public def::ConstrainOriginSupportPointToZero<"u", Model>,
    public def::ConstrainOriginSupportPointToZero<"v", Model>
{
public:
  template <typename Constraints, typename Context>
  void apply_affine_constraints(Constraints &constraints, const Context &context) const
  {
    def::ConstrainOriginSupportPointToZero<"u", Model>::apply_affine_constraints(constraints, context);
    def::ConstrainOriginSupportPointToZero<"v", Model>::apply_affine_constraints(constraints, context);
  }
};
```

Then the model just inherits the wrapper:
```Cpp
class MyModel : public def::AbstractModel<MyModel, Components>,
                public ConstrainUAndVAtOrigin<MyModel>,
                ...
{
  ...
};
```

This gives the same “strategy mixin” style as the FV boundary helpers.

### When should this not be used?

Affine constraints are for pinning a few specific dofs. They are usually **not** the right tool for:
- full PDE boundary conditions that are already naturally expressed through numerical fluxes,
- KT ghost-cell behavior, which belongs into the FV boundary stencil helpers,
- conditions that should be applied to a whole face or through weak boundary terms.

If the condition is really “pick this named dof and set it to a fixed value”, affine constraints are a good fit.

### Assemblers and discretizations

The actual numerical calculation of the flow equations (rather, their weak form) is done by the so-called assemblers. These are responsible for the actual discretization of the flow equations on the finite element space. In DiFfRG, we provide a set of assemblers for different discretizations, which are all derived from the abstract assembler class DiFfRG::AbstractAssembler.

Although all require at least the above interface methods, certain additional methods are required for certain discretizations. For example, the discontinuous Galerkin (DG) assemblers require the implementation of the numerical fluxes, and both discontinuous and continuous Galerkin (CG, also called simply FEM here) assemblers require the implementation of the boundary condition fluxes.

To understand the underlying numerics, see e.g. this [review](https://www3.nd.edu/~zxu2/acms60790S15/DG-general-approach.pdf) and also the excellent [deal.ii tutorials](https://www.dealii.org/developer/doxygen/deal.II/Tutorial.html).

For further reference, please refer to the documentation of the respective assemblers.
- DiFfRG::DG::Assembler
- DiFfRG::LDG::Assembler
- DiFfRG::CG::Assembler

Underlying the assemblers are the actual discretizations, which are implemented in the DiFfRG::discretization namespace. These are responsible for the actual discretization of the field space, i.e. the construction of the finite element space. In DiFfRG, we provide a set of discretizations for different finite element spaces.
- DiFfRG::DG::Discretization
- DiFfRG::CG::Discretization

### Running

Putting everything together, we can write a straightforward main function to run the flow equations:

```cpp
#include <DiFfRG/DiFfRG.hh>

using namespace DiFfRG;

// Make choices for types: the model, its discretization, the assembler and the timestepper
using Model = MyModel;
constexpr uint dim = Model::dim;
using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;
using Assembler = CG::Assembler<Discretization>;
using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

int main(int argc, char *argv[])
{
  // Initialize DiFfRG (MPI + Kokkos) and read the parameter file / CLI overrides
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  const auto json = config_helper.get_json();

  // Define the objects needed to run the simulation
  Model model(json);
  RectangularMesh<dim> mesh(json);
  OutputSession<Assembler> output(json);
  const auto log = output.report_port();
  Discretization discretization(mesh, json, log);
  Assembler assembler(discretization, model, json);
  HAdaptivity mesh_adaptor(assembler, json);
  TimeStepper time_stepper(json, assembler, output, mesh_adaptor);

  // Set up the initial condition
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  // Start the timestepping
  Timer timer;
  time_stepper.run(initial_condition, 0., json.get_double("/timestepping/final_time"));

  // Print a bit of exit information through the run-owned reporter.
  log.info("Simulation finished after " + time_format(timer.wall_time()));
  return 0;
}
```

Every object is constructed from the parsed `json` configuration, so all parameters
(grid, FE order, tolerances, output, physics) are read from `parameter.json` (or
`parameter.toml`; see [Project Structure](structure.md)). The
timestepper is the SUNDIALS IDA solver, which is the recommended solver for most cases.
If you use a discontinuous Galerkin discretization (`DG`/`LDG`) it is also
necessary to supply a numerical flux, which can be done by modifying the numerical model
as follows:
```cpp
class MyModel : public def::AbstractModel<MyModel, Components>,
                public def::fRG,                    // this handles the fRG time
                public def::LLFFlux<MyModel>,        // use a LLF numflux
                public def::FlowBoundaries<MyModel>, // use Inflow/Outflow boundaries
                public def::AD<MyModel>              // define all jacobians per AD
{
  ...
};
```
Here, the local Lax-Friedrichs flux has been used for the numerical fluxes and the boundaries have been defined to be inflow-/outflow. We have also chosen to use the autodiff functionality for the calculation of the jacobians.

## Other variables

Besides FE functions, a model can carry **variables** $v_a$ and **extractors** $e_b$. These are degrees of freedom that do **not** live on the field-space discretization — they are plain values (or arrays of values) evolved alongside (or instead of) the FE functions. They are the natural representation for momentum-dependent truncations, where the flowing objects are dressing functions and couplings tabulated on a momentum grid rather than on the FEM field grid.

Variables and extractors are declared in the `ComponentDescriptor` next to the FE functions, using either a `Scalar<"name">` (a single value) or a grid-valued `FunctionND<"name", sizes...>`:

```cpp
using namespace DiFfRG;

static constexpr uint p_grid_size = 96;
using VariableDesc  = VariableDescriptor<Scalar<"m2">, FunctionND<"ZA", p_grid_size>>;
using ExtractorDesc = ExtractorDescriptor<Scalar<"observable">>;
// A pure variable system has an empty FEFunctionDescriptor<>:
using Components = ComponentDescriptor<FEFunctionDescriptor<>, VariableDesc, ExtractorDesc>;

constexpr auto idxv = VariableDesc{};   // idxv("ZA") + i indexes grid point i of ZA
```

The difference between the two: **variables** are evolved in RG time by their own flow equation, while **extractors** are auxiliary quantities computed from the current state and made available to the FE-function flux/source methods (useful for coupling a FEM sector to global quantities).

A model implements the following methods (from `DiFfRG::def::AbstractModel`, default to no-ops):

- The initial condition of the variables,
```cpp
template <typename Vector> void initial_condition_variables(Vector &v_a) const;
```
- The flow of the variables $\partial_t v_a$,
```cpp
template <typename Vector, typename Solution> void dt_variables(Vector &r_a, const Solution &sol) const;
```
where `r_a` is the residual to fill and `sol` is a named tuple from which the current variables are obtained via `get<"variables">(sol)`.
- The extractors, filled from the current solution,
```cpp
template <int dim, typename Vector, typename Solutions> void extract(Vector &e_b, const Point<dim> &x, const Solutions &sol) const;
```

Spatial readouts and extractors also receive an experimental scalar raw-potential view under the provisional named
entries `"potential"`, `"potential_gradient"`, and `"potential_hessian"`. These are the value, gradient, and
element-local Hessian of one common scalar CG2 potential reconstructed from the model's unmodified potential gradient.
The Hessian may jump across cell interfaces. These entry names may change while this interface is experimental. The
value uses the gauge `potential(origin) == 0`; its derivatives do not depend on that gauge.

By default, the raw gradient copies the first `dim` solution components, i.e. `{values[0], ..., values[dim - 1]}`;
missing components are zero-filled. A model with a different component layout must override it independently:

```cpp
template <int dim, typename Vector>
std::array<double, dim> raw_potential_gradient(const Point<dim> &, const Vector &values) const
{
  return {{values[idxf("u")]}}; // unmodified dU/drho, without the physical-EoM correction
}
```

Reconstructing that potential is a direct solve over the whole mesh, and the extractors are evaluated on every
residual and every jacobian -- so a model whose extractors never read the three potential entries should say so:

```cpp
static constexpr bool extract_uses_potential = false;
```

The entries are then filled with a type that has no operations, so reading one is a compile error rather than a
silent zero. Readouts are unaffected either way: the potential is written to the spatial output regardless.

The readout-specific EoM callback reconstructs a separate scalar CG2 potential and selects the evaluation point. The
raw-potential fields and extractors are then evaluated at that same point for CG, DG, LDG, and KT-FV. Existing
`"fe_functions"`, `"fe_derivatives"`, and `"fe_hessians"` entries keep their assembler-specific meanings.

Spatial potential output follows the run's common `/output/vtk` and `/output/hdf5` settings. With VTK enabled, the
common raw reconstruction is written to `<run>_potential.pvd` with field name `potential`; the separate potentials
reconstructed from the readout-specific physical EoM callbacks are written to `<run>_eom_potential.pvd` with fields
`eom_potential`, `eom_potential_1`, and so on. With HDF5 enabled, both series share the run's single `<run>.h5` file:
raw-potential frames are stored under `/potential/<six-digit frame>` and EoM-potential frames under
`/eom_potential/<six-digit frame>`. Disabling VTK therefore does not disable either potential series when HDF5 remains
enabled.

A system that consists of variables only (no FE functions) is assembled by `DiFfRG::Variables::Assembler` (spatial dimension `0`) and carried by `DiFfRG::FlowingVariables`; when FE functions are also present the two sectors are coupled and the FEM assembler handles both. Momentum grids are represented by coordinate systems (e.g. `DiFfRG::LogarithmicCoordinates1D`) and evaluated through interpolators (e.g. `DiFfRG::SplineInterpolator1D`); the flow kernels are generated as grid `map` integrators.

For a complete, worked momentum-dependent example, see [Tutorial 4](../tutorials/tut4.md).
