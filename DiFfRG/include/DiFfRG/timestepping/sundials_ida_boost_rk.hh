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
   * @brief A class to perform time stepping using the adaptive Boost Runge-Kutta method for the explicit part and
   * SUNDIALS IDA for the implicit part. In this scheme, the IDA stepper is the controller and the Boost RK stepper
   * solves the explicit part of the problem on-demand.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   * @tparam prec Algorithm choice: 0 for Cash-Karp54 (5th order), 1 for Fehlberg78 (8th order)
   */
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename...> typename LinearSolver, int prec>
  class TimeStepperSUNDIALS_IDA_BoostRK_impl : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    using Base = AbstractTimestepper<VectorType, SparseMatrixType, dim>;

  public:
    using NumberType = typename Base::NumberType;
    using BlockVectorType = typename Base::BlockVectorType;

    using Base::assembler, Base::data_out, Base::config, Base::adaptor;

    /// The public entry point, which handles snapshots and restarts; see AbstractTimestepper::run.

    using Base::run;
    /// Forwards to the base with this stepper's kind. Not `using Base::Base;`: the base needs to
    /// know which /timestepping/ section to read, and it cannot ask a virtual function for that
    /// from inside its own constructor.
    TimeStepperSUNDIALS_IDA_BoostRK_impl(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                         OutputSession_impl<dim, VectorType> &data_out)
        : Base(config, assembler, data_out, /*implicit=*/true, /*explicit=*/true)
    {
    }
    TimeStepperSUNDIALS_IDA_BoostRK_impl(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                         OutputSession_impl<dim, VectorType> &data_out, AbstractAdaptor<VectorType> &adaptor)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/true, /*explicit=*/true)
    {
    }
    using Base::output_dt, Base::impl, Base::expl;

    using error_stepper_type = typename stepperChoice<prec>::value;

    virtual void run_segment(AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start,
                             const double t_stop) override;

  private:
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void update_variables(VectorType &variable_y, const VectorType &spatial_y, const double t);
  };

  // ##############################################################################
  // Application-facing spelling
  // ##############################################################################
  //
  // The class above takes the linear algebra spelled out because it is compiled out of line: the
  // set of valid arguments is closed by the explicit instantiations in src/. Applications name the
  // assembler instead and let these aliases project it onto that fixed parameter list.
  //
  // The argument is anything exposing VectorType, SparseMatrixType and dim -- an Assembler, or a
  // Discretization where the assembler type is not a single type (e.g. a test that runs one
  // discretization against several models).

  template <typename Assembler, template <typename...> typename LinearSolver, int prec>
  using TimeStepperSUNDIALS_IDA_BoostRK =
      TimeStepperSUNDIALS_IDA_BoostRK_impl<typename Assembler::VectorType, typename Assembler::SparseMatrixType,
                                           Assembler::dim, LinearSolver, prec>;

  /**
   * @brief Boost Cash-Karp54 for the explicit part, SUNDIALS IDA for the implicit part.
   *
   * IDA is the controller; the Boost RK stepper solves the explicit part on demand.
   */
  template <typename Assembler, template <typename...> typename LinearSolver = DefaultLinearSolver>
  using TimeStepperSUNDIALS_IDA_BoostRK54 = TimeStepperSUNDIALS_IDA_BoostRK<Assembler, LinearSolver, 0>;

  /**
   * @brief Boost Fehlberg78 for the explicit part, SUNDIALS IDA for the implicit part.
   */
  template <typename Assembler, template <typename...> typename LinearSolver = DefaultLinearSolver>
  using TimeStepperSUNDIALS_IDA_BoostRK78 = TimeStepperSUNDIALS_IDA_BoostRK<Assembler, LinearSolver, 1>;
} // namespace DiFfRG
