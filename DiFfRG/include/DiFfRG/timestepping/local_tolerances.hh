#pragma once

// DiFfRG
#include <DiFfRG/discretization/common/abstract_assembler.hh>

// external libraries
#include <deal.II/base/mpi.h>
#include <deal.II/lac/vector_operation.h>

// standard library
#include <algorithm>
#include <limits>

namespace DiFfRG
{
  /**
   * @brief Per-dof absolute tolerances for SUNDIALS IDA, taken from AbstractAssembler::local_abs_tolerances.
   *
   * IDA reads absolute tolerances only when it is (re)initialised. Model tolerances may depend on the state, so
   * the stepper asks refresh_needed() at every output time, from SUNDIALS::IDA::solver_should_restart: when some
   * dof needs a tolerance tighter than the one IDA uses by more than the factor
   * /timestepping/implicit/local_tolerance_refresh, the new set is adopted and IDA restarts warm (it keeps the last
   * step size). Models should give tolerances that vary smoothly with the state, or every sign change of a scale
   * forces a restart.
   */
  template <typename VectorType, typename SparseMatrixType, uint dim> class LocalAbsTolerances
  {
  public:
    LocalAbsTolerances(const AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler, double abs_tol,
                       double rel_tol, double refresh)
        : assembler(assembler), abs_tol(abs_tol), rel_tol(rel_tol), refresh(refresh)
    {
    }

    /// Compute the tolerances for `solution`. Returns false if the assembler/model provides none.
    bool init(const VectorType &solution)
    {
      active = compute(atol, solution);
      return active;
    }

    /**
     * True if the tolerances for `solution` ask for more accuracy than the ones IDA uses, by more than the refresh
     * factor on some dof; the new set is then adopted. A tolerance that has merely become looser than needed is
     * kept: it costs steps, not accuracy, and every restart drops IDA back to order 1 with a fresh Jacobian.
     */
    bool refresh_needed(const VectorType &solution)
    {
      if (!active || !(refresh > 1.)) return false;
      compute(candidate, solution);
      bool drifted = candidate.size() != atol.size();
      if (!drifted)
        for (const auto i : atol.locally_owned_elements())
          if (candidate(i) * refresh < atol(i)) {
            drifted = true;
            break;
          }
      // Every rank has to take the same decision: the stepper restarts IDA collectively.
      if constexpr (requires { atol.get_mpi_communicator(); })
        drifted = dealii::Utilities::MPI::max(int(drifted), atol.get_mpi_communicator()) > 0;
      if (drifted) atol = candidate;
      return drifted;
    }

    VectorType &get() { return atol; }
    double min() const
    {
      return extreme([](double a, double b) { return std::min(a, b); }, true);
    }
    double max() const
    {
      return extreme([](double a, double b) { return std::max(a, b); }, false);
    }

  private:
    /// The smallest (largest) tolerance over all ranks.
    template <typename Pick> double extreme(const Pick &pick, const bool smallest) const
    {
      if (atol.size() == 0) return abs_tol;
      double result = smallest ? std::numeric_limits<double>::max() : std::numeric_limits<double>::lowest();
      for (const auto i : atol.locally_owned_elements())
        result = pick(result, double(atol(i)));
      if constexpr (requires { atol.get_mpi_communicator(); })
        result = smallest ? dealii::Utilities::MPI::min(result, atol.get_mpi_communicator())
                          : dealii::Utilities::MPI::max(result, atol.get_mpi_communicator());
      return result;
    }

    bool compute(VectorType &out, const VectorType &solution) const
    {
      if (!assembler.local_abs_tolerances(out, solution, abs_tol, rel_tol)) return false;
      // IDA needs strictly positive tolerances
      for (const auto i : out.locally_owned_elements())
        if (!(out(i) > 0.)) out(i) = abs_tol;
      out.compress(dealii::VectorOperation::insert);
      return true;
    }

    const AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler;
    const double abs_tol, rel_tol, refresh;
    VectorType atol, candidate;
    bool active = false;
  };
} // namespace DiFfRG
