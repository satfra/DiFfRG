#pragma once

// external libraries
#include <deal.II/base/config.h>

// MUMPS is what makes a *distributed* direct solve possible; it is an opt-in PETSc package
// (-DPETSC_MUMPS=ON), and deal.II reports whether PETSc actually has it. Note that deal.II
// declares PETScWrappers::SparseDirectMUMPS unconditionally and only fails at runtime, so
// gating here is what turns "wrong answer at 3am" into "does not compile".
#if defined(DEAL_II_WITH_PETSC) && defined(DEAL_II_PETSC_WITH_MUMPS)

#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver_control.h>

// standard library
#include <memory>
#include <stdexcept>

// DiFfRG
#include <DiFfRG/timestepping/linear_solver/abstract_linear_solver.hh>

namespace DiFfRG
{
  /**
   * @brief Distributed direct solve of a PETSc matrix via MUMPS.
   *
   * The closest distributed analogue of the serial UMFPack path, and the right choice when the
   * Jacobian is too ill-conditioned for a Krylov method -- which, for the stiff DAEs IDA hands
   * us, it regularly is. Satisfies the same duck-typed contract as UMFPack, GMRES and
   * PETScKrylov, so it drops into the timesteppers with no call-site change.
   *
   * The solver object is kept alive across solve() calls and reset only in init(). That is
   * deliberate: PETSc caches the factorization inside the KSP/PC as long as the operator is
   * unchanged, so the several back-solves IDA performs per Jacobian reuse one factorization.
   * Rebuilding the solver per solve would silently refactorize every time.
   *
   * performs_factorization is false even though this *is* a direct solver. The flag does not
   * mean "direct"; callers use it to decide whether to record factorization diagnostics
   * (success flag, condition estimate -- see jacobian_diagnostics.hh). deal.II's MUMPS wrapper
   * exposes no separate factorization step to trigger or time, and no condition estimate, so
   * claiming otherwise would report a factorization that never independently happened.
   */
  template <typename SparseMatrixType, typename VectorType>
  class PETScDirect : public AbstractLinearSolver<SparseMatrixType, VectorType>
  {
  public:
    static constexpr bool performs_factorization = false;

    PETScDirect() : matrix(nullptr) {}

    void init(const SparseMatrixType &matrix)
    {
      this->matrix = &matrix;
      // New operator => the cached factorization is stale. Drop it; the next solve rebuilds.
      solver.reset();
      control.reset();
    }

    bool invert() { return false; }

    int solve(const VectorType &src, VectorType &dst, const double tol)
    {
      if (!matrix) throw std::runtime_error("PETScDirect::solve: matrix not initialized");

      if (!solver) {
        // A direct solve needs no iteration budget; MUMPS returns in one KSP step. The
        // tolerance is carried anyway so a caller tightening it is not silently ignored.
        control = std::make_unique<dealii::SolverControl>(1, tol);
        solver = std::make_unique<dealii::PETScWrappers::SparseDirectMUMPS>(*control);
      } else {
        control->set_tolerance(tol);
      }

      try {
        solver->solve(*matrix, dst, src);
      } catch (std::exception &e) {
        std::cerr << "PETSc MUMPS direct solver failed: " << e.what() << std::endl;
        throw;
      }

      return control->last_step();
    }

  private:
    const SparseMatrixType *matrix;
    // SparseDirectMUMPS holds its SolverControl by reference, so the control must outlive it;
    // declaration order here is load-bearing (members are destroyed in reverse).
    std::unique_ptr<dealii::SolverControl> control;
    std::unique_ptr<dealii::PETScWrappers::SparseDirectMUMPS> solver;
  };
} // namespace DiFfRG

#endif // DEAL_II_WITH_PETSC && DEAL_II_PETSC_WITH_MUMPS
