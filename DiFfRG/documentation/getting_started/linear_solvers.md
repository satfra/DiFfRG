# Scaled Linear Solvers {#ScaledLinearSolvers}

Implicit time steppers solve linear systems whenever they assemble a Jacobian.
For stiff flows or strongly inhomogeneous variables, these Jacobians can be badly
scaled: some rows or columns may contain entries that differ by many orders of
magnitude. DiFfRG provides scaled linear solver wrappers for this case.

The scaled solvers do not change the physical equation. They solve an
equilibrated linear system internally and map the result back to the original
variables.

## Available solvers

The generic wrapper is:
```Cpp
ScaledLinearSolver<SparseMatrixType, VectorType, InnerSolver>
```

The common aliases are:
```Cpp
ScaledGMRES<SparseMatrixType, VectorType>
ScaledUMFPack<SparseMatrixType, VectorType>
```

`ScaledGMRES` is the default iterative scaled solver. `ScaledUMFPack` is the
scaled direct solver.

The aliases expand to the generic wrapper with the corresponding inner solver:
```Cpp
template <typename SparseMatrixType, typename VectorType,
          typename PreconditionerType = dealii::PreconditionIdentity>
using ScaledGMRES =
    ScaledLinearSolver<SparseMatrixType, VectorType,
                       GMRES<SparseMatrixType, VectorType, PreconditionerType>>;

template <typename SparseMatrixType, typename VectorType>
using ScaledUMFPack =
    ScaledLinearSolver<SparseMatrixType, VectorType,
                       UMFPack<SparseMatrixType, VectorType>>;
```

`ScaledGMRES` uses identity preconditioning by default because the row and
column equilibration already rescales the linear system. Jacobi preconditioning
remains available explicitly:
```Cpp
ScaledGMRES<SparseMatrix<double>, Vector<double>,
            dealii::PreconditionJacobi<SparseMatrix<double>>>
```

## How the scaling works

For a linear system
\f[
  A x = b
\f]
the wrapper builds diagonal row and column scalings \f$D_r\f$ and \f$D_c\f$ and
solves
\f[
  D_r A D_c z = D_r b\,.
\f]
The final solution is recovered as
\f[
  x = D_c z\,.
\f]

The row and column scales are computed from the largest absolute entry in each
row and column. Empty, tiny, or non-finite scales fall back to `1`, so zero rows
or columns do not create infinities in the scaling vectors.

## Using a scaled solver in a timestepper

The implicit timestepper backend is selected through the last template argument.
For example, to use scaled GMRES with SUNDIALS IDA:
```Cpp
using TimeStepper =
    TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, ScaledGMRES>;
```

To use scaled UMFPack instead:
```Cpp
using TimeStepper =
    TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, ScaledUMFPack>;
```

Both aliases follow the same two-template-argument shape as the unscaled
`GMRES` and `UMFPack` wrappers, so they can be used in the same timestepper
template position.

## Using a scaled solver directly

Direct use mirrors the existing linear solver interface:
```Cpp
ScaledUMFPack<SparseMatrix<double>, Vector<double>> solver;
solver.init(jacobian);
solver.invert();
solver.solve(rhs, solution, tolerance);
```

For direct solvers such as `ScaledUMFPack`, call `invert()` after `init()`.
For iterative solvers such as `ScaledGMRES`, `invert()` is a no-op and the
iteration count is returned by `solve()`.

The generic wrapper can also be instantiated explicitly:
```Cpp
using InnerSolver = UMFPack<SparseMatrix<double>, Vector<double>>;
ScaledLinearSolver<SparseMatrix<double>, Vector<double>, InnerSolver> solver;
```

## Choosing between scaled GMRES and scaled UMFPack

Use `ScaledGMRES` when:
- the matrix is large enough that a direct factorization is expensive,
- an iteration count is useful diagnostic information,
- the unscaled GMRES solve stalls or needs many iterations.

Use `ScaledUMFPack` when:
- a direct solve is acceptable for the matrix size,
- robustness is more important than iteration diagnostics,
- you want the same scaling policy but with UMFPack factorization.

## Notes

- Scaling is applied to the linear system only; model equations and assembled
  residuals are unchanged.
- `ScaledUMFPack` works with sparse and block sparse deal.II matrices supported
  by the existing DiFfRG solver wrappers.
- The focused regression tests live in `tests/timestepping/scaled_gmres.cc`.
