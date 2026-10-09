// external libraries
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/time_stepping.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/rk.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperRK_impl<VectorType, SparseMatrixType, dim>::set_method(const uint method)
  {
    this->method = method;
  }

  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperRK_impl<VectorType, SparseMatrixType, dim>::run_segment(
      AbstractFlowingVariables<NumberType, VectorType> &initial_condition, const double t_start, const double t_stop)
  {

    auto &full_data = initial_condition.data();
    if constexpr (dim == 0)
      run_vars(full_data.block(1), t_start, t_stop);
    else {
      if (full_data.n_blocks() != 1)
        run(full_data, t_start, t_stop);
      else
        run(full_data.block(0), t_start, t_stop);
    }
  }

  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperRK_impl<VectorType, SparseMatrixType, dim>::run(VectorType & /*initial_data*/,
                                                                  const double /*t_start*/, const double /*t_stop*/)
  {
    throw std::runtime_error("TimeStepperRK_impl::run: not implemented");
  }

  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperRK_impl<VectorType, SparseMatrixType, dim>::run(BlockVectorType & /*initial_data*/,
                                                                  const double /*t_start*/, const double /*t_stop*/)
  {
    throw std::runtime_error("TimeStepperRK_impl::run: not implemented");
  }

  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperRK_impl<VectorType, SparseMatrixType, dim>::run_vars(VectorType &initial_data, const double t_start,
                                                                       const double t_stop)
  {
    if (initial_data.size() == 0)
      throw std::runtime_error("TimeStepperRK_impl::run: y contains no variables, use a different timestepper!");

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    auto output_step = [&](const double t, const VectorType &sol) {
      if (!is_close(last_save, t, 1e-10)) {
        assembler.set_time(t);

        data_out.write_frame(t, [&](auto &frame) { assembler.attach_data_output(frame, Vector<double>(), sol); });

        last_save = t;
      }
    };
    const auto output_each = uint(output_dt / expl.dt);

    TimeStepping::runge_kutta_method used_method;
    switch (method) {
    case 1:
      used_method = TimeStepping::runge_kutta_method::FORWARD_EULER;
      break;
    case 2:
      used_method = TimeStepping::runge_kutta_method::RK_THIRD_ORDER;
      break;
    case 3:
      used_method = TimeStepping::runge_kutta_method::SSP_THIRD_ORDER;
      break;
    case 4:
      used_method = TimeStepping::runge_kutta_method::RK_CLASSIC_FOURTH_ORDER;
      break;
    default:
      throw std::runtime_error("TimeStepperRK_impl::run: method not implemented");
    }

    TimeStepping::ExplicitRungeKutta<VectorType> explicit_runge_kutta(used_method);

    // Initialize initial condition
    VectorType y = initial_data;

    output_step(t_start, y);
    DiscreteTime time(t_start, t_stop, expl.dt);
    GrowingVectorMemory<VectorType> mem;
    while (time.is_at_end() == false) {
      explicit_runge_kutta.evolve_one_time_step(
          [&](const double t, const VectorType &y) {
            CalcDtTimer calc_timer;

            typename VectorMemory<VectorType>::Pointer f(mem);
            f->reinit(y);
            assembler.set_time(t);
            assembler.residual_variables(*f, y, Vector<double>());
            *f *= -1.;

            this->log.progress({.topic = progress_topics::explicit_residual,
                                .time = t,
                                .duration_ms = calc_timer.lap(),
                                .minimum_verbosity = 1});

            return *f;
          },
          time.get_current_time(), time.get_next_step_size(), y);
      time.advance_time();

      // output the solution every output_each steps
      if (time.get_step_number() % output_each == 0) {
        // also, check if the solution contains NaNs, if so, throw an error
        if (!std::isfinite(y.l2_norm())) throw std::runtime_error("TimeStepperRK_impl::run: NaNs in solution");

        output_step(time.get_current_time(), y);
      }
    }

    initial_data = y;
    this->drain_output();
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 0>;
template class DiFfRG::TimeStepperRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1>;
template class DiFfRG::TimeStepperRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2>;
template class DiFfRG::TimeStepperRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3>;

template class DiFfRG::TimeStepperRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0>;
template class DiFfRG::TimeStepperRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1>;
template class DiFfRG::TimeStepperRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2>;
template class DiFfRG::TimeStepperRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3>;
