#pragma once

// external libraries
#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_PETSC

#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_control.h>

// standard library
#include <stdexcept>

// DiFfRG
#include <DiFfRG/timestepping/linear_solver/abstract_linear_solver.hh>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief The preconditioner PETScKrylov defaults to.
     *
     * BoomerAMG (hypre) is the right default for the elliptic-ish implicit systems the FE/FV
     * discretizations produce, and it is what scales where a direct solve does not. It is only
     * available if PETSc was built with hypre, which is what -DPETSC_GMRES=ON arranges; deal.II
     * reports the outcome through DEAL_II_PETSC_WITH_HYPRE.
     *
     * Without it we fall back to block Jacobi, which needs no extra PETSc packages at all. That
     * is a weaker preconditioner, not a broken one -- the solver still converges, just in more
     * iterations -- so -DPETSC_GMRES=OFF degrades rather than fails.
     */
#ifdef DEAL_II_PETSC_WITH_HYPRE
    using PETScDefaultPreconditioner = dealii::PETScWrappers::PreconditionBoomerAMG;
#else
    using PETScDefaultPreconditioner = dealii::PETScWrappers::PreconditionBlockJacobi;
#endif

    /**
     * @brief The settings a preconditioner is built with. Generic case: whatever deal.II defaults to.
     */
    template <typename PreconditionerType> typename PreconditionerType::AdditionalData precondition_settings()
    {
      return typename PreconditionerType::AdditionalData();
    }

#ifdef DEAL_II_PETSC_WITH_HYPRE
    /**
     * @brief BoomerAMG, with a coarse-grid solver that does not crash on these systems.
     *
     * deal.II defaults relaxation_type_coarse to GaussianElimination, and hypre's implementation of
     * it segfaults on the matrices this code produces -- hypre_GaussElimSolve -> hypre_MPI_Allgatherv
     * for the finite-volume (Kurganov-Tadmor) Jacobian, and hypre_Memcpy onto a null destination for
     * the 1D O(N) CG one. Both happen at a SINGLE rank as well, and both happen on matrices whose
     * diagnostics are healthy (1024 rows, nnz 5952, min |diag| 976.5, no zero diagonals, finite
     * norms), so this is hypre's coarse solve failing, not bad assembly. Only the 2D CG system
     * survived the deal.II default.
     *
     * SORJacobi on the coarsest level costs a little convergence on paper and keeps every other
     * property of the multigrid hierarchy -- which matters, because the alternative on offer is
     * dropping AMG entirely and losing the one preconditioner here that scales.
     *
     * This is a workaround for a third-party defect, so it is written to be easy to retire: delete
     * this specialization and BoomerAMG goes back to deal.II's defaults.
     */
    template <>
    inline dealii::PETScWrappers::PreconditionBoomerAMG::AdditionalData
    precondition_settings<dealii::PETScWrappers::PreconditionBoomerAMG>()
    {
      using AMG = dealii::PETScWrappers::PreconditionBoomerAMG;
      AMG::AdditionalData data;
      data.relaxation_type_coarse = AMG::AdditionalData::RelaxationType::SORJacobi;
      return data;
    }
#endif
  } // namespace internal

  /**
   * @brief Distributed Krylov solve of a PETSc matrix, for the MPI-parallel FE/FV path.
   *
   * Satisfies the same duck-typed contract as UMFPack and GMRES -- init / invert / solve /
   * performs_factorization, with the template signature the timesteppers expect as a
   * template-template argument -- so it drops into TimeStepperSUNDIALS_IDA and friends with no
   * change at any call site.
   *
   * performs_factorization is false: this is an iterative solve, so there is nothing to
   * factorize up front and `invert()` is a no-op. Callers guard on that flag
   * (jacobian_diagnostics.hh's factorize_with_diagnostics, and the `if constexpr` at each
   * timestepper's setup_jacobian) and therefore skip factorization diagnostics for it, exactly
   * as they already do for the serial GMRES.
   *
   * @tparam SparseMatrixType a PETSc MPI matrix type
   * @tparam VectorType the matching PETSc MPI vector type
   * @tparam PreconditionerType see internal::PETScDefaultPreconditioner
   */
  template <typename SparseMatrixType, typename VectorType,
            typename PreconditionerType = internal::PETScDefaultPreconditioner>
  class PETScKrylov : public AbstractLinearSolver<SparseMatrixType, VectorType>
  {
  public:
    static constexpr bool performs_factorization = false;

    PETScKrylov() : matrix(nullptr) {}

    void init(const SparseMatrixType &matrix)
    {
      this->matrix = &matrix;
      // Rebuilding the preconditioner on every Jacobian is deliberate: IDA hands us a new
      // Jacobian only when it has decided the old one is stale, so reusing a preconditioner
      // across init() calls would be reusing it across exactly the changes it exists to track.
      preconditioner.initialize(matrix, internal::precondition_settings<PreconditionerType>());
    }

    bool invert() { return false; }

    int solve(const VectorType &src, VectorType &dst, const double tol)
    {
      if (!matrix) throw std::runtime_error("PETScKrylov::solve: matrix not initialized");

      // Same iteration cap as the serial GMRES wrapper. src.size() is the *global* size for a
      // PETSc vector, which is what we want here -- the cap should not depend on rank count.
      dealii::SolverControl solver_control(std::max<std::size_t>(1000, src.size() / 10), tol);
      dealii::PETScWrappers::SolverGMRES solver(solver_control);

      try {
        solver.solve(*matrix, dst, src, preconditioner);
      } catch (std::exception &e) {
        std::cerr << "PETSc GMRES linear solver failed: " << e.what() << std::endl;
        throw;
      }

      return solver_control.last_step();
    }

  private:
    const SparseMatrixType *matrix;
    PreconditionerType preconditioner;
  };
} // namespace DiFfRG

#endif // DEAL_II_WITH_PETSC
