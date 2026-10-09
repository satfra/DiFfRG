#pragma once

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/timestepping/linear_solver/PETScDirect.hh>
#include <DiFfRG/timestepping/linear_solver/PETScKrylov.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>

/**
 * @file default_linear_solver.hh
 *
 * @brief The linear solver a timestepper uses when the application does not name one.
 *
 * Lives here rather than in common/linear_algebra.hh on purpose: the solvers sit above the
 * linear-algebra layer, so naming them from common/ would invert the include order and
 * become a cycle the first time a solver header wants is_distributed_la.
 */

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Maps a (matrix, vector) pair to the solver that suits it.
     *
     * Serial linear algebra gets UMFPack, which is what every application used before this
     * default existed. Distributed linear algebra gets a *direct* solve via MUMPS where PETSc
     * was built with it, so that switching to an MPI build changes where the work happens but
     * not the character of the solve -- IDA's Newton sees the same convergence behaviour it
     * does serially. PETScKrylov is the fallback when MUMPS is absent; it is iterative, and a
     * flow that is fine-tuned against a direct solve may need its tolerances revisited.
     */
    template <typename SparseMatrixType, typename VectorType> struct _default_solver {
      using type = UMFPack<SparseMatrixType, VectorType>;
    };

#ifdef DEAL_II_WITH_PETSC
    // PETScDirect is declared only when deal.II reports PETSc was built with MUMPS -- the whole
    // class sits inside that guard -- so the choice has to be made by the preprocessor here. An
    // `if constexpr` would not help: the name must be visible for this to parse at all.
    template <> struct _default_solver<dealii::PETScWrappers::MPI::SparseMatrix, dealii::PETScWrappers::MPI::Vector> {
#ifdef DEAL_II_PETSC_WITH_MUMPS
      using type = PETScDirect<dealii::PETScWrappers::MPI::SparseMatrix, dealii::PETScWrappers::MPI::Vector>;
#else
      using type = PETScKrylov<dealii::PETScWrappers::MPI::SparseMatrix, dealii::PETScWrappers::MPI::Vector>;
#endif
    };
#endif
  } // namespace internal

  /**
   * @brief The build-configuration-dependent default linear solver.
   *
   * The indirection through _default_solver is load-bearing and must not be "simplified" to
   * `using DefaultLinearSolver = UMFPack<...>`. As a template template argument, a *simple*
   * alias is collapsed to the template it names by GCC but not by Clang; an indirect one is
   * kept distinct by both. Since the timesteppers are compiled out of line, a spelling that
   * collapses on one compiler and not the other would match the explicit instantiations in
   * src/ only where it collapses, and produce undefined references everywhere else.
   *
   * The consequence of staying distinct is worth knowing: TimeStepperSUNDIALS_IDA_impl<Assembler>
   * and TimeStepperSUNDIALS_IDA_impl<Assembler, UMFPack> are different types even in a serial
   * build, and both need their own explicit instantiation.
   */
  template <typename SparseMatrixType, typename VectorType>
  using DefaultLinearSolver = typename internal::_default_solver<SparseMatrixType, VectorType>::type;
} // namespace DiFfRG
