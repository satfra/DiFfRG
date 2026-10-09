#pragma once

// external libraries
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/abstract_timestepper.hh>

namespace DiFfRG
{
  template <int prec> struct stepperChoice {
    static_assert(prec == 0 || prec == 1, "Only precisions 0 and 1 are supported!");

    using error_stepper_type_0 = boost::numeric::odeint::runge_kutta_cash_karp54<Eigen::VectorXd>;
    using error_stepper_type_1 = boost::numeric::odeint::runge_kutta_fehlberg78<Eigen::VectorXd>;

    using value = std::conditional_t<prec == 0, error_stepper_type_0, error_stepper_type_1>;
  };

  /**
   * @brief A class to perform time stepping using adaptive Boost Runge-Kutta methods.
   * This stepper uses adaptive time steps and is fully explicit.
   *
   * @tparam VectorType Type of the vector
   * @tparam dim Dimension of the problem
   * @tparam prec Algorithm choice: 0 for Cash-Karp54 (5th order), 1 for Fehlberg78 (8th order)
   *
   */
  template <typename VectorType, typename SparseMatrixType, uint dim, int prec>
  class TimeStepperBoostRK_impl : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    static_assert(!is_distributed_la<VectorType>,
                  "TimeStepperBoostRK is serial-only: it integrates through Eigen, which needs contiguous storage. "
                  "In an MPI build a plain RectangularMesh is partitioned and selects PETSc linear algebra: "
                  "use RectangularMeshSerial<dim>, or TimeStepperSUNDIALS_IDA / _IDA_BoostABM / _IDA_BoostRK "
                  "for a distributed run.");

    using Base = AbstractTimestepper<VectorType, SparseMatrixType, dim>;

  public:
    using NumberType = typename Base::NumberType;
    using InverseSparseMatrixType = typename get_type::InverseSparseMatrixType<SparseMatrixType>;
    using BlockVectorType = typename Base::BlockVectorType;

    using Base::assembler, Base::data_out, Base::config, Base::adaptor;

    /// The public entry point, which handles snapshots and restarts; see AbstractTimestepper::run.

    using Base::run;
    /// Forwards to the base with this stepper's kind. Not `using Base::Base;`: the base needs to
    /// know which /timestepping/ section to read, and it cannot ask a virtual function for that
    /// from inside its own constructor.
    TimeStepperBoostRK_impl(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                            OutputSession_impl<dim, VectorType> &data_out)
        : Base(config, assembler, data_out, /*implicit=*/false, /*explicit=*/true)
    {
    }
    TimeStepperBoostRK_impl(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                            OutputSession_impl<dim, VectorType> &data_out, AbstractAdaptor<VectorType> &adaptor)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/false, /*explicit=*/true)
    {
    }
    using Base::output_dt, Base::impl, Base::expl;

    using error_stepper_type = typename stepperChoice<prec>::value;

    /**
     * @brief Run the time stepping algorithm.
     */
    virtual void run_segment(AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start,
                             const double t_stop) override;

  private:
    void run(VectorType &initial_data, const double t_start, const double t_stop);
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void run_vars(VectorType &initial_data, const double t_start, const double t_stop);
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

  template <typename Assembler, int prec>
  using TimeStepperBoostRK = TimeStepperBoostRK_impl<typename Assembler::VectorType,
                                                     typename Assembler::SparseMatrixType, Assembler::dim, prec>;

  /// Time stepping with the adaptive Boost Cash-Karp54 method.
  template <typename Assembler> using TimeStepperBoostRK54 = TimeStepperBoostRK<Assembler, 0>;

  /// Time stepping with the adaptive Boost Fehlberg78 method.
  template <typename Assembler> using TimeStepperBoostRK78 = TimeStepperBoostRK<Assembler, 1>;
} // namespace DiFfRG
