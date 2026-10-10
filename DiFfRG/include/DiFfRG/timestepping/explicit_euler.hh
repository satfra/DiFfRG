#pragma once

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
  /**
   * @brief Forward Euler with a fixed step.
   *
   * \f[
   *   u_{n+1} = u_n - \Delta t\, M^{-1} R(u_n)\,, \qquad \Delta t = \text{/timestepping/explicit/dt}\,,
   * \f]
   * with the last step shortened to end on the final time. Only the FE part is integrated; a flow with variables
   * throws. Frames are written at \f$t_\text{start} + n\,\f$output_dt, and at the final time, linearly interpolated
   * within the step that reaches them. Serial only; the mesh adaptor is called before every step.
   */
  template <typename VectorType, typename SparseMatrixType = dealii::SparseMatrix<get_type::NumberType<VectorType>>,
            uint dim = 0>
  class TimeStepperExplicitEuler_impl : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    static_assert(!is_distributed_la<VectorType>,
                  "TimeStepperExplicitEuler is serial-only: "
                  "it integrates through Eigen, which needs contiguous storage. "
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

    // Not `using Base::Base;`: the base needs to know which /timestepping/ section to read, and it
    // cannot ask a virtual function for that from inside its own constructor.

    /// Construct without mesh adaptation; see AbstractTimestepper.
    TimeStepperExplicitEuler_impl(const ConfigTree &config,
                                  AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                  OutputSession_impl<dim, VectorType> &data_out)
        : Base(config, assembler, data_out, /*implicit=*/false, /*explicit=*/true)
    {
    }
    /// Construct with a mesh adaptor; see AbstractTimestepper.
    TimeStepperExplicitEuler_impl(const ConfigTree &config,
                                  AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                                  OutputSession_impl<dim, VectorType> &data_out, AbstractAdaptor<VectorType> &adaptor)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/false, /*explicit=*/true)
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

  /// Forward-Euler timestepper for an assembler; see TimeStepperExplicitEuler_impl.
  template <typename Assembler>
  using TimeStepperExplicitEuler = TimeStepperExplicitEuler_impl<typename Assembler::VectorType,
                                                                 typename Assembler::SparseMatrixType, Assembler::dim>;
} // namespace DiFfRG
