#pragma once

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/discretization/common/snapshot_state.hh>
#include <DiFfRG/discretization/common/solution_view.hh>
#include <DiFfRG/discretization/data/output_session.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <unsigned int dim, typename VectorType> class DataOutput;

  namespace internal
  {
    /// Compile-time check that a readout helper takes (id, EoM, outputter).
    template <typename... Args> constexpr void validate_readout_helper_arity()
    {
      static_assert(sizeof...(Args) == 3,
                    "The readout helper now requires a stable readout ID: helper(id, EoM, outputter)");
    }

  } // namespace internal

  /**
   * @brief Interface shared by all assemblers (CG, DG, LDG, FV, variables only).
   *
   * An assembler turns a discretization and a model into the vectors and matrices a timestepper needs: the mass
   * term, the residual and their Jacobians with respect to the degrees of freedom (dofs). Timesteppers only talk to
   * this interface.
   *
   * ### Equation
   * The model describes \f$N_f\f$ fields \f$u_i(x)\f$, \f$x \in \Omega \subset \mathbb{R}^{dim}\f$, through
   * \f[
   *   m_i(u, \partial_t u) + \nabla\cdot F_i(u, \nabla u, \dots) + s_i(u, \nabla u, \dots) = 0\,,
   *   \qquad i = 0, \dots, N_f - 1\,,
   * \f]
   * with mass \f$m_i\f$ (Model::mass), flux \f$F_i\f$ (a vector in \f$\mathbb{R}^{dim}\f$, Model::flux) and source
   * \f$s_i\f$ (Model::source). For an ordinary flow, \f$m_i = \sum_j M_{ij}\,\partial_t u_j\f$ is linear in
   * \f$\partial_t u\f$; components with \f$\partial m_i / \partial(\partial_t u) = 0\f$ are algebraic constraints.
   *
   * Multiplying with a test function \f$\phi\f$ (one per dof) and integrating the flux by parts gives the
   * residual this interface assembles,
   * \f[
   *   r_\phi = w_m \int_\Omega m\,\phi
   *         + w_r \Big[ \int_\Omega \big( s\,\phi - F\cdot\nabla\phi \big)
   *         + \int_{\partial\Omega} (\widehat F\cdot n)\,\phi
   *         + \sum_{\text{interior faces } f} \int_f (\widehat F\cdot n_f)\,[\![\phi]\!] \Big]\,,
   * \f]
   * where \f$w_m\f$, \f$w_r\f$ are weights chosen by the timestepper, \f$n\f$ is the outward unit normal,
   * \f$\widehat F\f$ is the numerical flux on a face (Model::boundary_numflux on \f$\partial\Omega\f$) and
   * \f$[\![\phi]\!]\f$ is the jump of \f$\phi\f$ across the face \f$f\f$ with normal \f$n_f\f$. The
   * interior-face sum is present only for discontinuous spaces (DG, LDG); the FV assembler builds the analogous
   * cell-average balance. The flow is then \f$r(u, \partial_t u) = 0\f$.
   *
   * Methods that take \f$\partial_t u\f$ explicitly evaluate \f$m(u, \partial_t u)\f$. The convenience overloads
   * without it pass \f$u\f$ in place of \f$\partial_t u\f$: for a mass linear in \f$\partial_t u\f$ they
   * compute \f$M u\f$ (and its Jacobian \f$M\f$), which is what explicit steppers need.
   *
   * Additional scalar *variables* \f$v\f$ (Model::dt_variables) are handled by residual_variables and
   * jacobian_variables.
   *
   * @tparam VectorType Vector type of the dofs (e.g. dealii::Vector<double>).
   * @tparam SparseMatrixType Matrix type of the Jacobians.
   * @tparam dim Spatial dimension.
   */
  template <typename VectorType, typename SparseMatrixType, uint dim> class AbstractAssembler
  {
  public:
    using NumberType = typename get_type::NumberType<VectorType>;

    /// Aggregate assembly statistics for the run summary.
    virtual SummaryEvent summary() const { return {}; }

    /**
     * @brief Write the FE functions and the model's readouts into one output frame.
     *
     * The FE functions appear under their names from the model's ComponentDescriptor; if given, their time
     * derivatives and residuals are added as `<name>_dot` and `<name>_residual`.
     *
     * @param data_out The current output frame.
     * @param solution The spatial solution.
     * @param variables The additional variables (empty if the model has none).
     * @param dt_solution \f$\partial_t u\f$; empty to skip the `<name>_dot` fields.
     * @param residual The residual vector; empty to skip the `<name>_residual` fields.
     */
    virtual void attach_data_output(OutputFrame<dim, VectorType> &data_out, const VectorType &solution,
                                    const VectorType &variables = VectorType(),
                                    const VectorType &dt_solution = VectorType(),
                                    const VectorType &residual = VectorType()) = 0;

    [[deprecated("Override attach_data_output(OutputFrame<dim, VectorType>&, ...) instead")]] virtual void
    attach_data_output(DataOutput<dim, VectorType> &, const VectorType &, const VectorType & = VectorType(),
                       const VectorType & = VectorType(), const VectorType & = VectorType()) = delete;

    /**
     * @brief Rebuild all mesh-dependent data. Required after the mesh changed (e.g. adaptive refinement).
     */
    virtual void reinit() = 0;

    /**
     * @brief Set the current RG time \f$t\f$; forwarded to the model.
     */
    virtual void set_time(double t) = 0;

    /**
     * @brief Whether the model asks for a restart of the time integration at the accepted state @p solution at
     * time @p t (see DiFfRG::HasNeedsRestart). Called once per accepted implicit step. Default: never.
     *
     * @param solution A vector in which every dof is readable (a refreshed SolutionView).
     */
    virtual bool needs_restart([[maybe_unused]] const double t, [[maybe_unused]] const VectorType &solution) const
    {
      return false;
    }

    /// Whether needs_restart can ever return true, i.e. whether the timestepper has to ask it.
    virtual bool requests_restarts() const { return false; }

    /**
     * @brief The sparsity pattern of the Jacobian.
     *
     * @return The pattern, of type get_type::SparsityPattern<SparseMatrixType>.
     */
    virtual const get_type::SparsityPattern<SparseMatrixType> &get_sparsity_pattern_jacobian() const = 0;

    /**
     * @brief The dofs whose equation contains a time derivative; all others, and all constrained dofs, are algebraic.
     */
    virtual IndexSet get_differential_indices() const = 0;

    /**
     * @brief Per-dof absolute tolerances for the implicit (IDA) timesteppers.
     *
     * Instead of the uniform `/timestepping/implicit/abs_tol` and `rel_tol`, a model can set one tolerance per dof
     * through the hook `abs_tolerances` (def::HasAbsTolerances), useful when components live on very different scales;
     * IDA then runs with relative tolerance 0, so these are the whole tolerance. They may depend on the state; the IDA steppers re-read them when some dof needs a tolerance tighter by more
     * than the factor `/timestepping/implicit/local_tolerance_refresh`.
     *
     * @param atol Output; sized and filled when the function returns true.
     * @param solution The current spatial solution.
     * @param abs_tol The stepper's uniform absolute tolerance, passed on to the model hook.
     * @param rel_tol The stepper's relative tolerance, passed on to the model hook.
     * @return false if the assembler or model provides no per-dof tolerances; the stepper then uses abs_tol.
     */
    virtual bool local_abs_tolerances([[maybe_unused]] VectorType &atol, [[maybe_unused]] const VectorType &solution,
                                      [[maybe_unused]] double abs_tol, [[maybe_unused]] double rel_tol) const
    {
      return false;
    }

    /**
     * @brief Resize @p vector to the dofs of this discretization (with the parallel layout, if distributed).
     */
    virtual void reinit_vector(VectorType &vector) const = 0;

    /**
     * @brief Reinitialize @p matrix with the Jacobian's sparsity pattern (and parallel layout, if distributed).
     */
    virtual void reinit_matrix(SparseMatrixType &matrix) const = 0;

    /**
     * @brief The MPI communicator of the linear algebra; MPI_COMM_SELF for a serial discretization.
     */
    virtual MPI_Comm get_communicator() const = 0;

    /**
     * @brief Set up @p view as a read-only copy of the solution in which every global dof is readable on every
     * rank. Used by output and the EoM search; see SolutionView.
     */
    virtual void reinit_solution_view(SolutionView<VectorType> &view) const = 0;

    /**
     * @brief The mass matrix \f$M_{\phi\psi} = \int_\Omega \phi\,\psi\f$ of the test functions.
     */
    virtual const SparseMatrixType &get_mass_matrix() const = 0;

    /**
     * @name Flow snapshots
     *
     * Used by AbstractTimestepper::run to write and restore snapshots. The defaults throw, so an assembler without
     * snapshot support fails only when snapshots are requested.
     */
    //@{

    /**
     * @brief Record the spatial state (mesh and values) independently of the dof numbering.
     *
     * @param spatial_replica A vector in which every global dof is readable (a refreshed SolutionView). Read-only,
     * so it may be called on the writing rank alone.
     */
    virtual SnapshotSpatialState capture_snapshot_state([[maybe_unused]] const VectorType &spatial_replica) const
    {
      throw std::runtime_error("This assembler does not support flow snapshots.");
    }

    /**
     * @brief Rebuild the snapshot's mesh if it differs from the current one, reinitialize, and fill @p spatial with
     * the snapshot's values. Collective: call on every rank.
     */
    virtual void restore_snapshot_state([[maybe_unused]] const SnapshotSpatialState &state,
                                        [[maybe_unused]] VectorType &spatial)
    {
      throw std::runtime_error("This assembler does not support restarting from flow snapshots.");
    }

    /// Let the model store its history-dependent state (Model::save_state); see ModelState.
    virtual void save_model_state([[maybe_unused]] ModelState &state) const {}

    /// Hand restored state to the model (Model::load_state). @return whether the model implements load_state.
    virtual bool load_model_state([[maybe_unused]] const ModelState &state) { return false; }
    //@}

    /**
     * @name Residual and jacobian functions
     */
    //@{

    /**
     * @brief Residual of the variables, \f$D(v, u)\f$ = Model::dt_variables, evaluated with the extractors at
     * the EoM point. Note the convention: the variables obey \f$\partial_t v + D(v, u) = 0\f$.
     *
     * @param residual Output, one entry per variable.
     * @param variables The current variables \f$v\f$.
     * @param spatial_solution The spatial solution \f$u\f$.
     */
    virtual void residual_variables([[maybe_unused]] VectorType &residual, [[maybe_unused]] const VectorType &variables,
                                    [[maybe_unused]] const VectorType &spatial_solution)
    {
      throw std::runtime_error("residual_variables() is not implemented by this assembler");
    };

    /**
     * @brief Jacobian \f$\partial D / \partial v\f$ of the variables' residual.
     *
     * @param jacobian Output, a square matrix of size n_variables.
     * @param variables The current variables \f$v\f$.
     * @param spatial_solution The spatial solution \f$u\f$.
     */
    virtual void jacobian_variables([[maybe_unused]] FullMatrix<NumberType> &jacobian,
                                    [[maybe_unused]] const VectorType &variables,
                                    [[maybe_unused]] const VectorType &spatial_solution)
    {
      throw std::runtime_error("jacobian_variables() is not implemented by this assembler");
    };

    /**
     * @brief Mass term with @p solution_global in place of \f$\partial_t u\f$, i.e.
     * \f$w_m \int_\Omega m(u, u)\,\phi\f$. For a mass linear in \f$\partial_t u\f$ this is \f$w_m\,M u\f$.
     *
     * Calls mass(mass, solution_global, solution_global, weight).
     */
    void mass(VectorType &mass, const VectorType &solution_global, NumberType weight)
    {
      this->mass(mass, solution_global, solution_global, weight);
    }

    /**
     * @brief Mass term \f$w_m \int_\Omega m(u, \partial_t u)\,\phi\f$, one entry per test function \f$\phi\f$.
     *
     * @param mass Output vector.
     * @param solution_global The spatial solution \f$u\f$.
     * @param solution_global_dot Its time derivative \f$\partial_t u\f$.
     * @param weight The weight \f$w_m\f$.
     */
    virtual void mass(VectorType &mass, const VectorType &solution_global, const VectorType &solution_global_dot,
                      NumberType weight) = 0;

    /**
     * @brief Residual \f$r_\phi\f$ (see the class description) with @p solution_global in place of
     * \f$\partial_t u\f$ in the mass term.
     *
     * Calls residual(residual, solution_global, weight, solution_global, weight_mass, variables).
     */
    void residual(VectorType &residual, const VectorType &solution_global, NumberType weight, NumberType weight_mass,
                  const VectorType &variables = VectorType())
    {
      this->residual(residual, solution_global, weight, solution_global, weight_mass, variables);
    }

    /**
     * @brief Residual \f$r_\phi\f$ of the class description: mass, flux, source and face terms.
     *
     * @param residual Output vector.
     * @param solution_global The spatial solution \f$u\f$.
     * @param weight The weight \f$w_r\f$ of flux and source.
     * @param solution_global_dot Its time derivative \f$\partial_t u\f$.
     * @param weight_mass The weight \f$w_m\f$ of the mass.
     * @param variables The variables \f$v\f$, on which flux and source may depend; empty if none.
     */
    virtual void residual(VectorType &residual, const VectorType &solution_global, NumberType weight,
                          const VectorType &solution_global_dot, NumberType weight_mass,
                          const VectorType &variables = VectorType()) = 0;

    /**
     * @brief Adds \f$w_m\,\partial m / \partial(\partial_t u)\f$ to @p jacobian, evaluated with @p solution_global
     * in place of \f$\partial_t u\f$. For a mass linear in \f$\partial_t u\f$ this is \f$w_m M\f$.
     *
     * Calls jacobian_mass(jacobian, solution_global, solution_global, mass_weight, 0).
     */
    void jacobian_mass(SparseMatrixType &jacobian, const VectorType &solution_global, NumberType mass_weight = 1.)
    {
      this->jacobian_mass(jacobian, solution_global, solution_global, mass_weight, 0.);
    }

    /**
     * @brief Adds the Jacobian of the mass term to @p jacobian:
     * \f[
     *   \alpha \int_\Omega \frac{\partial m}{\partial(\partial_t u_\psi)}\,\phi
     *   + \beta \int_\Omega \frac{\partial m}{\partial u_\psi}\,\phi\,,
     * \f]
     * row \f$\phi\f$, column \f$\psi\f$ (dof with coefficient \f$u_\psi\f$).
     *
     * @param jacobian Matrix to add to.
     * @param solution_global The spatial solution \f$u\f$.
     * @param solution_global_dot Its time derivative \f$\partial_t u\f$.
     * @param alpha Weight of the derivative with respect to \f$\partial_t u\f$.
     * @param beta Weight of the derivative with respect to \f$u\f$.
     */
    virtual void jacobian_mass(SparseMatrixType &jacobian, const VectorType &solution_global,
                               const VectorType &solution_global_dot, NumberType alpha = 1., NumberType beta = 1.) = 0;

    /**
     * @brief Jacobian of residual(residual, solution_global, weight, mass_weight, variables): adds to @p jacobian.
     *
     * Calls jacobian(jacobian, solution_global, weight, solution_global, mass_weight, mass_weight, variables).
     */
    void jacobian(SparseMatrixType &jacobian, const VectorType &solution_global, NumberType weight,
                  NumberType mass_weight, const VectorType &variables = VectorType())
    {
      this->jacobian(jacobian, solution_global, weight, solution_global, mass_weight, mass_weight, variables);
    }

    /**
     * @brief Adds the Jacobian of the residual with respect to \f$u\f$ to @p jacobian:
     * the mass part as in jacobian_mass (weights \f$\alpha\f$, \f$\beta\f$) plus @p weight times the derivative of
     * the flux, source and face terms. For an implicit step, \f$\alpha\f$ is the factor in
     * \f$\partial(\partial_t u)/\partial u\f$ set by the time discretization.
     *
     * @param jacobian Matrix to add to.
     * @param solution_global The spatial solution \f$u\f$.
     * @param weight The weight \f$w_r\f$ of flux and source.
     * @param solution_global_dot Its time derivative \f$\partial_t u\f$.
     * @param alpha Weight of \f$\partial m / \partial(\partial_t u)\f$.
     * @param beta Weight of \f$\partial m / \partial u\f$.
     * @param variables The variables \f$v\f$; empty if none.
     */
    virtual void jacobian(SparseMatrixType &jacobian, const VectorType &solution_global, NumberType weight,
                          const VectorType &solution_global_dot, NumberType alpha, NumberType beta,
                          const VectorType &variables = VectorType()) = 0;
    //@}
  };
} // namespace DiFfRG
