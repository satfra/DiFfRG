#pragma once

// DiFfRG
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/timestepping/jacobian_diagnostics.hh>

// standard library
#include <exception>

namespace DiFfRG
{
  /**
   * @brief Changes an assembled IDA Jacobian \f$J = \partial F/\partial u + \alpha\,\partial F/\partial\dot u\f$ to a
   * new \f$\alpha\f$ without reassembling it; see IDADriver::update_jacobian_alpha.
   *
   * remember() records the state and \f$\alpha\f$ of each full assembly. The first update() after it assembles
   * \f$\partial F/\partial\dot u\f$ at that state (AbstractAssembler::jacobian_mass), and every update() adds
   * \f$(\alpha_{\rm new} - \alpha)\,\partial F/\partial\dot u\f$ to the matrix.
   */
  template <typename VectorType, typename SparseMatrixType, uint dim> class JacobianAlphaUpdate
  {
  public:
    explicit JacobianAlphaUpdate(AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler) : assembler(assembler)
    {
    }

    /// The Jacobian was assembled at time @p t, state (@p y, @p y_dot) and factor @p alpha.
    void remember(const double t, const VectorType &y, const VectorType &y_dot, const double alpha)
    {
      this->t = t;
      this->y = y;
      this->y_dot = y_dot;
      this->alpha = alpha;
      valid = true;
      mass_valid = false;
    }

    /// The remembered Jacobian is gone, e.g. after a mesh adaptation or a failed update.
    void forget() { valid = mass_valid = false; }

    /// Change @p jacobian, assembled as remembered, to @p new_alpha. Returns false if nothing is remembered.
    bool update(SparseMatrixType &jacobian, const double new_alpha)
    {
      if (!valid) return false;
      if (!mass_valid) {
        assembler.reinit_matrix(mass_jacobian);
        mass_jacobian = 0;
        assembler.set_time(t);
        assembler.jacobian_mass(mass_jacobian, y, y_dot, 1., 0.);
        mass_valid = true;
      }
      jacobian.add(new_alpha - alpha, mass_jacobian);
      alpha = new_alpha;
      return true;
    }

    /**
     * @brief update() @p jacobian and prepare @p solver with it, agreed between all ranks. On failure the remembered
     * Jacobian is forgotten and false returned, so that IDADriver reassembles it.
     */
    template <typename LinearSolver>
    bool update_and_factorize(SparseMatrixType &jacobian, const double new_alpha, LinearSolver &solver)
    {
      bool failed = false;
      try {
        failed = !update(jacobian, new_alpha);
        if (!failed) {
          solver.init(jacobian);
          JacobianFactorizationDiagnostics factorization;
          factorize_with_diagnostics(solver, jacobian, factorization, false);
          if constexpr (LinearSolver::performs_factorization) failed = factorization.factorization_success != 1.;
        }
      } catch (std::exception &) {
        failed = true;
      }
      if (MPI::any_of(assembler.get_communicator(), failed)) {
        forget();
        return false;
      }
      return true;
    }

  private:
    AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler;
    double t = 0., alpha = 0.;
    VectorType y, y_dot;
    SparseMatrixType mass_jacobian;
    bool valid = false, mass_valid = false;
  };
} // namespace DiFfRG
