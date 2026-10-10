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

namespace DiFfRG
{
  /**
   * @brief Backward Euler with adaptive step size.
   *
   * Each step solves, by Newton's method with the matrix \f$M + \Delta t\, \partial R/\partial u\f$,
   * \f[
   *   m(u_{n+1}) - m(u_n) + \Delta t\, R(u_{n+1}) = 0\,,
   * \f]
   * which is \f$M(u_{n+1} - u_n) + \Delta t\, R(u_{n+1}) = 0\f$ for the usual linear mass term (notation of
   * abstract_timestepper.hh). First order in \f$\Delta t\f$.
   *
   * The step size is chosen by a PI controller (TC_PI) from the Newton convergence measure of the last step (see
   * Newton), not from an estimate of the time discretization error: /timestepping/implicit/abs_tol and rel_tol set how
   * accurately each step's equation is solved, and the step grows as long as Newton converges easily. Steps stay in
   * [implicit/minimal_dt, implicit/maximal_dt]; at minimal_dt a step is accepted even if Newton did not converge.
   * A frame is written at the first step past each multiple of output_dt, and at the end.
   *
   * Only the FE part is integrated; a flow with variables throws. Serial only; the mesh adaptor is called
   * before every step.
   *
   * @tparam LinearSolver solver for the Newton systems; see DefaultLinearSolver.
   */
  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  class TimeStepperImplicitEuler_impl : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    static_assert(!is_distributed_la<VectorType>,
                  "TimeStepperImplicitEuler has no distributed (PETSc) implementation yet. "
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
    TimeStepperImplicitEuler_impl(const ConfigTree &config,
                                  AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                  OutputSession_impl<dim, VectorType> &data_out)
        : Base(config, assembler, data_out, /*implicit=*/true, /*explicit=*/false)
    {
    }
    /// Construct with a mesh adaptor; see AbstractTimestepper.
    TimeStepperImplicitEuler_impl(const ConfigTree &config,
                                  AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                  OutputSession_impl<dim, VectorType> &data_out, AbstractAdaptor<VectorType> &adaptor)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/true, /*explicit=*/false)
    {
    }
    using Base::output_dt, Base::impl, Base::expl;

    virtual void run_segment(AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start,
                             const double t_stop) override;
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

  /// Backward-Euler timestepper for an assembler; see TimeStepperImplicitEuler_impl.
  template <typename Assembler, template <typename...> typename LinearSolver = DefaultLinearSolver>
  using TimeStepperImplicitEuler =
      TimeStepperImplicitEuler_impl<typename Assembler::VectorType, typename Assembler::SparseMatrixType,
                                    Assembler::dim, LinearSolver>;
} // namespace DiFfRG
