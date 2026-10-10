#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/abstract_timestepper.hh>
#include <DiFfRG/timestepping/default_linear_solver.hh>

#include <functional>
#include <utility>

namespace DiFfRG
{
  /**
   * @brief Implicit, adaptive time stepping with SUNDIALS IDA (variable-order BDF, order 1 to 5).
   *
   * IDA solves the system in residual form, \f$F(t, y, \dot y) = 0\f$, with \f$y = (u, v)\f$ and
   * \f[
   *   F_u = m(u, \dot u) + R(u, v)\,, \qquad F_v = \dot v + D(v, u)\,,
   * \f]
   * see abstract_timestepper.hh for the notation. Each step solves \f$F = 0\f$ by Newton's method with the matrix
   * \f$\partial F/\partial y + \alpha\, \partial F/\partial \dot y\f$, where \f$\alpha\f$ is set by IDA from the
   * step size and the BDF order. The \f$u\f$- and \f$v\f$-blocks of this matrix are built separately, i.e. the
   * couplings \f$\partial F_u/\partial v\f$ and \f$\partial F_v/\partial u\f$ are left out; they slow Newton's
   * convergence but do not change the solution.
   *
   * Components of \f$u\f$ whose mass term does not depend on \f$\dot u\f$ are algebraic constraints; they are
   * excluded from the error test. At the start and after every restart (e.g. after mesh adaptation), IDA computes the
   * algebraic components and the \f$\dot u\f$ of the others from \f$F = 0\f$. Output is written at exactly
   * \f$t_\text{start} + n\, \mathrm{output\_dt}\f$, interpolated by IDA.
   *
   * Handles FE problems with or without variables, and pure-variables systems (`dim == 0`). Distributed linear
   * algebra is supported. A callback that fails (non-finite state or residual, an exception from the model, a
   * failed factorization or linear solve) makes IDA retry with a smaller step; the run aborts after 200 consecutive
   * failures or if the time stops advancing.
   *
   * Reads /timestepping/implicit, see abstract_timestepper.hh.
   *
   * @tparam VectorType vector type of the assembler.
   * @tparam SparseMatrixType sparse matrix type of the assembler.
   * @tparam dim spatial dimension, 0 for a pure-variables system.
   * @tparam LinearSolver solver for the Newton systems; see DefaultLinearSolver.
   */
  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver>
  class TimeStepperSUNDIALS_IDA_impl : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    using Base = AbstractTimestepper<VectorType, SparseMatrixType, dim>;

  public:
    using NumberType = typename Base::NumberType;
    using BlockVectorType = typename Base::BlockVectorType;
    using IDAErrorDofCallback = std::function<void(const IDAErrorDofDiagnostics &)>;

    using Base::assembler, Base::data_out, Base::config, Base::adaptor;

    /// The public entry point, which handles snapshots and restarts; see AbstractTimestepper::run.
    using Base::run;

    // Not `using Base::Base;`: the base needs to know which /timestepping/ section to read, and it
    // cannot ask a virtual function for that from inside its own constructor.

    /// Construct without mesh adaptation; see AbstractTimestepper.
    TimeStepperSUNDIALS_IDA_impl(const ConfigTree &config,
                                 AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                 OutputSession_impl<dim, VectorType> &data_out)
        : Base(config, assembler, data_out, /*implicit=*/true, /*explicit=*/false)
    {
    }
    /// Construct with a mesh adaptor; see AbstractTimestepper.
    TimeStepperSUNDIALS_IDA_impl(const ConfigTree &config,
                                 AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                 OutputSession_impl<dim, VectorType> &data_out, AbstractAdaptor<VectorType> &adaptor)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/true, /*explicit=*/false)
    {
    }
    using Base::output_dt, Base::impl, Base::expl;

    virtual void run_segment(AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start,
                             const double t_stop) override;

    /// Called with every report of /timestepping/implicit/ida_error_dof_diagnostics, in addition to the log.
    void set_ida_error_dof_callback(IDAErrorDofCallback callback) { ida_error_dof_callback = std::move(callback); }

  private:
    void run(VectorType &initial_data, const double t_start, const double t_stop);
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void run_vars(VectorType &initial_data, const double t_start, const double t_stop);

    IDAErrorDofCallback ida_error_dof_callback;
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

  /**
   * @brief SUNDIALS IDA timestepper for an assembler; see TimeStepperSUNDIALS_IDA_impl.
   *
   * @tparam Assembler anything exposing VectorType, SparseMatrixType and dim, usually the Assembler.
   * @tparam LinearSolver defaults to UMFPack for serial and a distributed solver for PETSc linear algebra; see
   * DefaultLinearSolver.
   */
  template <typename Assembler, template <typename...> typename LinearSolver = DefaultLinearSolver>
  using TimeStepperSUNDIALS_IDA =
      TimeStepperSUNDIALS_IDA_impl<typename Assembler::VectorType, typename Assembler::SparseMatrixType, Assembler::dim,
                                   LinearSolver>;
} // namespace DiFfRG
