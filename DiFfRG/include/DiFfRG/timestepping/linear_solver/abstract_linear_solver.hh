#pragma once

#include <DiFfRG/common/run_reporter.hh>

// external libraries
#include <deal.II/lac/sparse_matrix.h>

/**
 * @file abstract_linear_solver.hh
 * @brief The interface of the linear solvers used by the implicit timesteppers.
 *
 * The implicit timesteppers take the solver as a template template parameter `LinearSolver<SparseMatrixType,
 * VectorType>` and call it without virtual dispatch, so a solver only has to provide these members:
 * - `static constexpr bool performs_factorization`: whether factorization diagnostics (timing, condition estimate)
 *   apply to this solver. If true, the solver must also provide `estimate_scaled_rcond(matrix, max_iterations)`, used
 * by /timestepping/implicit/jacobian_diagnostics.
 * - `void init(const SparseMatrixType &A)`: use the matrix \f$A\f$, which stays alive and unchanged until the next
 *   init().
 * - `bool invert()`: prepare solves with \f$A\f$, e.g. factorize it. May throw on failure.
 * - `int solve(const VectorType &b, VectorType &x, double tol)`: solve \f$A x = b\f$; returns the number of
 *   iterations, or -1 for a direct solver.
 * - `void set_report_port(ReportPort)`, inherited from AbstractLinearSolver.
 * - optional `void set_iterative_refinement(bool)`, for /timestepping/implicit/iterative_refinement.
 *
 * Available: UMFPack (serial direct, the serial default), GMRES, ScaledGMRES, ScaledUMFPack, PETScDirect (MPI,
 * direct, the MPI default where PETSc has MUMPS), PETScKrylov (MPI, iterative); see DefaultLinearSolver.
 */

namespace DiFfRG
{
  /**
   * @brief Common base of the linear solvers; see abstract_linear_solver.hh for the interface a solver provides.
   *
   * The member functions here only throw: the timesteppers call the derived class directly.
   */
  template <typename SparseMatrixType, typename VectorType> class AbstractLinearSolver
  {
  public:
    AbstractLinearSolver() {}

    /** Attach the run reporter. Derived solvers may submit solver-specific ProgressEvents through this port. */
    void set_report_port(ReportPort port) { report_port = std::move(port); }

    void init(const SparseMatrixType &matrix)
    {
      (void)matrix;
      throw std::runtime_error("AbstractLinearSolver::init: not implemented");
    };

    bool invert()
    {
      throw std::runtime_error("AbstractLinearSolver::invert: not implemented");
      return false;
    };

    int solve(const VectorType &src, VectorType &dst, const double tol)
    {
      (void)src;
      (void)dst;
      (void)tol;
      throw std::runtime_error("AbstractLinearSolver::solve: not implemented");
      return -1;
    };

  protected:
    ReportPort report_port;
  };
} // namespace DiFfRG
