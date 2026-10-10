#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/abstract_timestepper.hh>
#include <DiFfRG/timestepping/boost_rk.hh>
#include <DiFfRG/timestepping/default_linear_solver.hh>

namespace DiFfRG
{
  /**
   * @brief SUNDIALS IDA for the FE part, an adaptive explicit Runge-Kutta method (Boost.Odeint) for the variables.
   *
   * IDA (see TimeStepperSUNDIALS_IDA_impl) integrates the FE part \f$m(u, \dot u) + R(u, v) = 0\f$; the variables
   * \f$\dot v = -D(v, u)\f$ are integrated by an explicit method. Use this when \f$D\f$ is expensive (e.g. momentum
   * integrals): IDA treats \f$v\f$ as a given function of time, so \f$D\f$ is evaluated only by the explicit steps,
   * not in every Newton iteration and not for the Jacobian. In turn, the explicit method reads \f$u(t)\f$ by
   * interpolation between the steps IDA has accepted.
   *
   * Requires an FE system with variables. The variables are integrated redundantly on every rank. The variables
   * written at an output time are interpolated from the explicit steps.
   *
   * The Runge-Kutta steps adapt to /timestepping/explicit/abs_tol and rel_tol, starting from explicit/dt and capped at
   * explicit/maximal_dt.
   *
   * Reads /timestepping/implicit and /timestepping/explicit, see abstract_timestepper.hh.
   *
   * @tparam VectorType vector type of the assembler.
   * @tparam SparseMatrixType sparse matrix type of the assembler.
   * @tparam dim spatial dimension (at least 1).
   * @tparam LinearSolver solver for IDA's Newton systems; see DefaultLinearSolver.
   * @tparam prec 0: Cash-Karp 5(4), 1: Fehlberg 7(8).
   */
  template <typename VectorType, typename SparseMatrixType, uint dim, template <typename...> typename LinearSolver,
            int prec>
  class TimeStepperSUNDIALS_IDA_BoostRK_impl : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
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
    TimeStepperSUNDIALS_IDA_BoostRK_impl(const ConfigTree &config,
                                         AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                         OutputSession_impl<dim, VectorType> &data_out)
        : Base(config, assembler, data_out, /*implicit=*/true, /*explicit=*/true)
    {
    }
    /// Construct with a mesh adaptor; see AbstractTimestepper.
    TimeStepperSUNDIALS_IDA_BoostRK_impl(const ConfigTree &config,
                                         AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                         OutputSession_impl<dim, VectorType> &data_out,
                                         AbstractAdaptor<VectorType> &adaptor)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/true, /*explicit=*/true)
    {
    }
    using Base::output_dt, Base::impl, Base::expl;

    using error_stepper_type = typename stepperChoice<prec>::value;

    virtual void run_segment(AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start,
                             const double t_stop) override;

  private:
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
  };

  // ##############################################################################
  // Application-facing spelling
  // ##############################################################################
  //
  // The class above takes the linear algebra spelled out because it is compiled out of line: the
  // set of valid arguments is closed by the explicit instantiations in src/. Applications name the
  // assembler instead and let these aliases project it onto that fixed parameter list.

  /// IDA + Runge-Kutta timestepper for an assembler; see TimeStepperSUNDIALS_IDA_BoostRK_impl.
  template <typename Assembler, template <typename...> typename LinearSolver, int prec>
  using TimeStepperSUNDIALS_IDA_BoostRK =
      TimeStepperSUNDIALS_IDA_BoostRK_impl<typename Assembler::VectorType, typename Assembler::SparseMatrixType,
                                           Assembler::dim, LinearSolver, prec>;

  /// IDA for the FE part, Cash-Karp 5(4) Runge-Kutta for the variables.
  template <typename Assembler, template <typename...> typename LinearSolver = DefaultLinearSolver>
  using TimeStepperSUNDIALS_IDA_BoostRK54 = TimeStepperSUNDIALS_IDA_BoostRK<Assembler, LinearSolver, 0>;

  /// IDA for the FE part, Fehlberg 7(8) Runge-Kutta for the variables.
  template <typename Assembler, template <typename...> typename LinearSolver = DefaultLinearSolver>
  using TimeStepperSUNDIALS_IDA_BoostRK78 = TimeStepperSUNDIALS_IDA_BoostRK<Assembler, LinearSolver, 1>;
} // namespace DiFfRG
