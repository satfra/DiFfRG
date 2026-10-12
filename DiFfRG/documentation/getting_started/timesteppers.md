# Timesteppers and linear solvers

A timestepper evolves the discretized system in RG time $t = \ln(\Lambda/k)$. All of them are class templates
taking the assembler as their first argument, and are constructed the same way:

```cpp
using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;
TimeStepper time_stepper(json, assembler, data_out);                // or (..., data_out, mesh_adaptor)
time_stepper.run(initial_condition, 0., json.get_double("/timestepping/final_time"));
```

In the notation used below, the FE functions $u$ obey $m(u,\partial_t u) + R(u, v) = 0$, where $m$ is the model's
mass term and $R$ collects flux and source, and the variables $v$ obey $\partial_t v + D(v, u) = 0$, with $D$ the
model's `dt_variables`.

## Choosing a timestepper

| timestepper | method | FE functions | variables | notes |
| --- | --- | --- | --- | --- |
| `TimeStepperSUNDIALS_IDA` | implicit, adaptive BDF (order 1–5) | yes | yes, implicit | **The default choice.** Handles algebraic equations (a mass term without $\partial_t u$), MPI, mesh adaptivity. |
| `TimeStepperSUNDIALS_IDA_BoostABM` | IDA for $u$, explicit Adams–Bashforth–Moulton for $v$ | yes | yes, explicit | For FE + variables when $D$ is expensive (momentum integrals): $D$ is never evaluated inside Newton iterations or for the Jacobian. |
| `TimeStepperSUNDIALS_IDA_BoostRK54`, `..._BoostRK78` | IDA for $u$, adaptive explicit Runge–Kutta for $v$ | yes | yes, explicit | As above, with an embedded Runge–Kutta pair for $v$. |
| `TimeStepperBoostABM` | explicit Adams–Bashforth–Moulton (order 8), optional step control | yes | yes | Pure-variable systems (momentum-dependent truncations without field dependence). Serial only. |
| `TimeStepperBoostRK54`, `TimeStepperBoostRK78` | adaptive explicit Runge–Kutta | yes | yes | Serial only, no mesh adaptivity. |
| `TimeStepperRK` | fixed-step explicit Runge–Kutta | no | yes | Pure-variable systems only. |
| `TimeStepperExplicitEuler` | forward Euler, fixed step | yes | no | Mainly for testing. |
| `TimeStepperImplicitEuler` | backward Euler, adaptive | yes | no | First order; mainly for testing and comparison. Serial only. |
| `TimeStepperTRBDF2` | TR-BDF2, adaptive, second order | yes | no | Serial only. |

Explicit steppers read `/timestepping/explicit`, implicit ones `/timestepping/implicit`, and the IDA + Boost
hybrids read both; see the [parameter reference](parameters.md#timestepping).

Some guidance:

- Flows of a field-dependent potential are stiff, and with spontaneous symmetry breaking the potential develops
  regions where explicit steppers need prohibitively small steps. Use IDA, with a relative tolerance of $10^{-7}$
  or tighter.
- For FE functions coupled to variables whose flow needs momentum integrals, use one of the IDA + Boost hybrids
  ([Tutorials 5](../tutorials/tut5.md) and [8](../tutorials/tut8.md) use one). With plain IDA, every Jacobian and Newton iteration would re-evaluate all momentum integrals.
- For a pure-variable system ([Tutorial 4](../tutorials/tut4.md)), `TimeStepperBoostABM` needs two evaluations of
  the right-hand side per step and is usually the cheapest. Use IDA if the system is stiff.
- The hybrids have three ways of coupling the explicit and implicit sector, chosen by
  `/timestepping/explicit/coupling_mode` (ABM only); see
  {doc}`TimeStepperSUNDIALS_IDA_BoostABM_impl </doxygen/DiFfRG/html/classDiFfRG_1_1TimeStepperSUNDIALS__IDA__BoostABM__impl>`.

## Linear solvers

Implicit timesteppers solve a linear system with the Jacobian in every Newton iteration. The solver is the second
template argument of the timestepper:

```cpp
using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler, GMRES>;
```

| solver | kind | when to use |
| --- | --- | --- |
| `UMFPack` | sparse direct | Default in serial builds. Robust; fine for 1D field spaces and moderately sized 2D ones. |
| `GMRES` | iterative (deal.II), Jacobi preconditioner by default | Large systems, e.g. field spaces with $d\geq 2$, where a factorization is too expensive. |
| `ScaledUMFPack`, `ScaledGMRES` | equilibrated versions of the above | Badly scaled Jacobians, see below. |
| `PETScDirect`, `PETScKrylov` | distributed (PETSc) | Selected automatically for distributed linear algebra in MPI builds. |

### Scaled solvers

For stiff flows or strongly inhomogeneous variables, entries of the Jacobian can differ by many orders of
magnitude between rows or columns. The scaled solvers equilibrate the linear system before solving it: for
$A x = b$ they build diagonal matrices $D_r$, $D_c$ from the largest absolute entry of each row and column, solve

```{math}
D_r A D_c\, z = D_r b
```

and return $x = D_c z$. Empty, tiny or non-finite scales fall back to 1. Only the linear solve is affected; the
equations and residuals are unchanged.

```cpp
using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler, ScaledGMRES>;    // or ScaledUMFPack
```

`ScaledGMRES` uses no preconditioner by default, since the equilibration already rescales the system. A
preconditioner can be given explicitly by naming the full wrapper:

```cpp
ScaledGMRES<SparseMatrix<double>, Vector<double>, dealii::PreconditionJacobi<SparseMatrix<double>>>
```

Prefer `ScaledGMRES` when the matrix is large or unscaled GMRES needs many iterations, and `ScaledUMFPack` when a
direct solve is affordable and robustness matters most.

### Using a solver directly

All solvers share one interface, which also allows using them outside a timestepper:

```cpp
ScaledUMFPack<SparseMatrix<double>, Vector<double>> solver;
solver.init(jacobian);
solver.invert();                              // factorizes; a no-op for iterative solvers
solver.solve(rhs, solution, tolerance);       // returns the iteration count for iterative solvers
```
