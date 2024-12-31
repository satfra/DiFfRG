#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/common/abstract_data.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/abstract_timestepper.hh>

namespace DiFfRG
{
  /**
   * @brief This timestepping class utilizes the ARKode solver from SUNDIALS.
   * It is specifically meant to be used for a spatial discretization which uses implicit timestepping and coupled to
   * additional variables which are evolved explicitly. Therefore, this solver is an ImEx solver and requires the
   * dynamics of the additional variables to be far slower than the dynamics of the spatial discretization. In the used
   * Assembler, the two additional methods residual_variables and set_additional_data must be implemented, which are
   * used to compute the explicit residual and to make the additional variables available to the implicit residual.
   *
   * @tparam VectorType This must be Vector<NumberType>. Other types are currently not supported, as we use a
   * BlockVector internally to store the solution.
   * @tparam dim The dimensionality of the spatial discretization
   */
  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  class TimeStepperSUNDIALS_ARKode : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    using Base = AbstractTimestepper<VectorType, SparseMatrixType, dim>;

  public:
    using NumberType = typename Base::NumberType;
    using InverseSparseMatrixType = typename Base::InverseSparseMatrixType;
    using BlockVectorType = typename Base::BlockVectorType;

    using Base::assembler, Base::data_out, Base::json, Base::adaptor;
    using Base::Base;
    using Base::console_out;
    using Base::verbosity, Base::output_dt, Base::impl, Base::expl;

    virtual void run(AbstractFlowingVariables<NumberType> *initial_condition, const double t_start,
                     const double t_stop) override;

    /**
     * @brief Run the timestepping algorithm with the given initial condition.
     * This method treats the entire system explicitly, e.g. if the spatial discretization is not yet stiff.
     */
    void run_explicit(AbstractFlowingVariables<NumberType> *initial_condition, const double t_start,
                      const double t_stop);

  private:
    void run(VectorType &initial_data, const double t_start, const double t_stop);
    void run(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void run_explicit(VectorType &initial_data, const double t_start, const double t_stop);
    void run_explicit(BlockVectorType &initial_data, const double t_start, const double t_stop);
    void run_explicit_vars(VectorType &initial_data, const double t_start, const double t_stop);
  };
} // namespace DiFfRG