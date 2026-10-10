#pragma once

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/abstract_timestepper.hh>
#include <DiFfRG/timestepping/default_linear_solver.hh>
#include <DiFfRG/timestepping/solver/kinsol.hh>
#include <DiFfRG/timestepping/solver/newton.hh>
#include <DiFfRG/timestepping/timestep_control/pi.hh>

namespace DiFfRG
{
  /**
   * @brief TR-BDF2 with adaptive step size: a trapezoidal stage followed by a BDF2 stage, second order.
   *
   * With \f$\gamma = 2 - \sqrt 2\f$ and the notation of abstract_timestepper.hh, a step from \f$u_n\f$ solves first
   * for \f$u_\gamma\f$ at \f$t_n + \gamma\Delta t\f$, then for \f$u_{n+1}\f$:
   * \f[
   *   m(u_\gamma) - m(u_n) + \frac{\gamma\Delta t}{2}\big(R(u_n) + R(u_\gamma)\big) = 0\,,
   * \f]
   * \f[
   *   m(u_{n+1}) - \frac{m(u_\gamma)}{\gamma(2 - \gamma)} + \frac{(1 - \gamma)^2}{\gamma(2 - \gamma)}\, m(u_n)
   *   + \frac{\gamma}{2}\Delta t\, R(u_{n+1}) = 0\,,
   * \f]
   * using \f$(1-\gamma)/(2-\gamma) = \gamma/2\f$. Both stages thus have the Newton matrix
   * \f$M + \frac{\gamma}{2}\Delta t\, J\f$, \f$J = \partial R/\partial u\f$, and share one Jacobian and linear solver.
   *
   * The step size is chosen by a PI controller (TC_PI) from the local error estimate of Bank et al. (1985),
   * \f[
   *   \Big(M + \frac{\gamma}{2}\Delta t\, J\Big)\, e = 2 k \Delta t \Big(-\frac{R(u_n)}{\gamma}
   *   + \frac{R(u_\gamma)}{\gamma(1-\gamma)} - \frac{R(u_{n+1})}{1-\gamma}\Big)\,, \qquad
   *   k = \frac{-3\gamma^2 + 4\gamma - 2}{12(2-\gamma)}\,.
   * \f]
   * A step is rejected and retried with a smaller \f$\Delta t\f$ if
   * \f$\sqrt{\frac1N\sum_i (e_i / (\text{atol} + \text{rtol}\,|u_i|))^2} > 1\f$, with atol, rtol =
   * /timestepping/implicit/abs_tol, rel_tol (which also set Newton's tolerance). Steps stay in
   * [implicit/minimal_dt, implicit/maximal_dt]; at minimal_dt a step is accepted regardless of the error estimate
   * and of Newton convergence.
   * Frames are written at \f$t_\text{start} + n\,\f$output_dt, interpolated within the step that reaches them
   * (quadratically through \f$u_n\f$, \f$u_\gamma\f$ and \f$u_{n+1}\f$).
   *
   * Only the FE part is integrated; a flow with variables throws. Serial only; the mesh adaptor is called
   * before every step.
   *
   * @tparam LinearSolver solver for the Newton systems; see DefaultLinearSolver.
   */
  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  class TimeStepperTRBDF2_impl : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    static_assert(!is_distributed_la<VectorType>,
                  "TimeStepperTRBDF2 has no distributed (PETSc) implementation yet. "
                  "In an MPI build a plain RectangularMesh is partitioned and selects PETSc linear algebra: "
                  "use RectangularMeshSerial<dim>, or TimeStepperSUNDIALS_IDA / _IDA_BoostABM / _IDA_BoostRK "
                  "for a distributed run.");

    using Base = AbstractTimestepper<VectorType, SparseMatrixType, dim>;

  public:
    using NumberType = typename Base::NumberType;
    using BlockVectorType = typename Base::BlockVectorType;

    using Base::assembler, Base::data_out, Base::config, Base::adaptor;

    /// The public entry point, which handles snapshots and restarts; see AbstractTimestepper::run.
    using Base::run;

    // Not `using Base::Base;`: the base needs to know which /timestepping/ section to read, and it
    // cannot ask a virtual function for that from inside its own constructor.

    /// Construct without mesh adaptation; see AbstractTimestepper.
    TimeStepperTRBDF2_impl(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                           OutputSession_impl<dim, VectorType> &data_out)
        : Base(config, assembler, data_out, /*implicit=*/true, /*explicit=*/false)
    {
    }
    /// Construct with a mesh adaptor; see AbstractTimestepper.
    TimeStepperTRBDF2_impl(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                           OutputSession_impl<dim, VectorType> &data_out, AbstractAdaptor<VectorType> &adaptor)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/true, /*explicit=*/false)
    {
    }
    using Base::output_dt, Base::impl, Base::expl;

    virtual void run_segment(AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start,
                             const double t_stop) override;

    // The interface TC_PI uses to control the step; valid only during run().

    /// Jacobians built in the last step, both stages.
    uint get_jacobians();
    /// Weighted RMS norm of the local error estimate of the last attempted step (infinite if Newton failed).
    double get_error() const { return error_estimate; }
    /// Accept steps whose Newton iteration did not converge; see Newton::set_ignore_nonconv.
    void set_ignore_nonconv(bool x);

  private:
    std::shared_ptr<Newton<VectorType>> ptr_newton_TR;
    std::shared_ptr<Newton<VectorType>> ptr_newton_BDF2;
    double error_estimate = 0.;
  };

  // ##############################################################################
  // Application-facing spelling
  // ##############################################################################
  //
  // The class above takes the linear algebra spelled out because it is compiled out of line: the
  // set of valid arguments is closed by the explicit instantiations in src/, and keying it on the
  // assembler would explode that set into one instantiation per model. Applications name the
  // assembler instead and let these aliases project it onto that fixed parameter list.
  //
  // Note that <Assembler> and <Assembler, UMFPack> are *different types* even in a serial build --
  // see the comment on DefaultLinearSolver -- so both need their own explicit instantiation.

  /// TR-BDF2 timestepper for an assembler; see TimeStepperTRBDF2_impl.
  template <typename Assembler, template <typename...> typename LinearSolver = DefaultLinearSolver>
  using TimeStepperTRBDF2 = TimeStepperTRBDF2_impl<typename Assembler::VectorType, typename Assembler::SparseMatrixType,
                                                   Assembler::dim, LinearSolver>;
} // namespace DiFfRG
