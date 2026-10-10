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
   * @brief Explicit 8-step Adams-Bashforth-Moulton method, with optional step-size control.
   *
   * Integrates \f$\dot u = -M^{-1} R(u, v)\f$ and \f$\dot v = -D(v, u)\f$ (see abstract_timestepper.hh) with an
   * Adams-Bashforth predictor and one Adams-Moulton corrector, both of order 8, which costs two evaluations of the
   * right-hand side per step. The first 7 steps, before enough history exists, are Cash-Karp 5(4) Runge-Kutta steps.
   *
   * The base step is the largest \f$\Delta t \le\f$ /timestepping/explicit/dt that divides the interval into equal
   * steps. With /timestepping/explicit/error_control (default true) the local error is estimated from the difference
   * between predictor and corrector; a step that fails the test
   * \f$|e_i| \le \mathrm{abs\_tol} + \mathrm{rel\_tol}\, |y_i|\f$ is retaken at half the step size; at
   * explicit/minimal_dt it is accepted anyway. After 16 steps in a row with an error below a quarter of the tolerance
   * the step size doubles, up to the base step. Without error control every step is the base step.
   *
   * Handles FE problems with or without variables and pure-variables systems (`dim == 0`). Output frames are linearly
   * interpolated between steps. Serial only; the mesh adaptor is not called.
   */
  template <typename VectorType, typename SparseMatrixType = dealii::SparseMatrix<get_type::NumberType<VectorType>>,
            uint dim = 0>
  class TimeStepperBoostABM_impl : public AbstractTimestepper<VectorType, SparseMatrixType, dim>
  {
    static_assert(!is_distributed_la<VectorType>,
                  "TimeStepperBoostABM is serial-only: it integrates through Eigen, which needs contiguous storage. "
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
    TimeStepperBoostABM_impl(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                             OutputSession_impl<dim, VectorType> &data_out)
        : Base(config, assembler, data_out, /*implicit=*/false, /*explicit=*/true)
    {
    }
    /// Construct with a mesh adaptor; see AbstractTimestepper.
    TimeStepperBoostABM_impl(const ConfigTree &config, AbstractAssembler<VectorType, SparseMatrixType, dim> &assembler,
                             OutputSession_impl<dim, VectorType> &data_out, AbstractAdaptor<VectorType> &adaptor)
        : Base(config, assembler, data_out, adaptor, /*implicit=*/false, /*explicit=*/true)
    {
    }
    using Base::output_dt, Base::impl, Base::expl;

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
  // set of valid arguments is closed by the explicit instantiations in src/, and keying it on the
  // assembler would explode that set into one instantiation per model. Applications name the
  // assembler instead and let these aliases project it onto that fixed parameter list.

  /// Explicit Adams-Bashforth-Moulton timestepper for an assembler; see TimeStepperBoostABM_impl.
  template <typename Assembler>
  using TimeStepperBoostABM =
      TimeStepperBoostABM_impl<typename Assembler::VectorType, typename Assembler::SparseMatrixType, Assembler::dim>;
} // namespace DiFfRG
