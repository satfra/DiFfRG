// standard library
#include <algorithm>

// external libraries
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/explicit_euler.hh>
#include <DiFfRG/timestepping/timestep_control/output_grid.hh>

namespace DiFfRG
{
  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperExplicitEuler_impl<VectorType, SparseMatrixType, dim>::run_segment(
      AbstractFlowingVariables<NumberType, VectorType> &initial_condition, double start, double stop)
  {
    this->require_no_variables(initial_condition, "TimeStepperExplicitEuler");

    const SparseMatrixType &mass_matrix = assembler.get_mass_matrix();
    InverseSparseMatrixType inverse_mass_matrix;
    inverse_mass_matrix.initialize(mass_matrix);

    // Start out by saving step 0
    assembler.set_time(start);

    VectorType old_solution = initial_condition.spatial_data();
    VectorType solution = initial_condition.spatial_data();

    data_out.write_frame(start, [&](auto &frame) { assembler.attach_data_output(frame, solution); });

    // Frames on the grid start + n output_dt, linearly interpolated within the step that reaches them.
    internal::OutputGrid output_grid(start, output_dt);
    VectorType frame_state;
    double t = start;
    // Not `t < stop`: the rounding in t += dt must not leave a last step of size ~1e-17.
    while (stop - t > 1e-10 * expl.dt) {
      CalcDtTimer calc_timer;
      if (adaptor(t, old_solution)) {
        solution = old_solution;
        inverse_mass_matrix.initialize(mass_matrix);
      }

      // The last step is shortened to land on stop exactly: run() hands the state at stop to the
      // next segment or writes it to a snapshot.
      const double dt = std::min(expl.dt, stop - t);

      solution = 0.;
      assembler.set_time(t);
      assembler.residual(solution, old_solution, -dt, 0.);
      inverse_mass_matrix.solve(solution);
      solution += old_solution;

      output_grid.advance_to(t + dt, [&](const double t_out) {
        const double theta = (t_out - t) / dt;
        frame_state = old_solution;
        frame_state.sadd(1. - theta, theta, solution);
        assembler.set_time(t_out);
        data_out.write_frame(t_out, [&](auto &frame) { assembler.attach_data_output(frame, frame_state); });
      });
      t += dt;
      old_solution.swap(solution);

      this->log.progress({.topic = progress_topics::explicit_residual,
                          .time = t,
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1});
    }

    // After the swap at the end of the loop body, old_solution holds the state at t.
    initial_condition.spatial_data() = old_solution;
    this->drain_output();
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperExplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1>;
template class DiFfRG::TimeStepperExplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2>;
template class DiFfRG::TimeStepperExplicitEuler_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3>;

template class DiFfRG::TimeStepperExplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1>;
template class DiFfRG::TimeStepperExplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2>;
template class DiFfRG::TimeStepperExplicitEuler_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3>;
