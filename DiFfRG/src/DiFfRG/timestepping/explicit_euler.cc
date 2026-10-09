// standard library
#include <algorithm>

// external libraries
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/explicit_euler.hh>

namespace DiFfRG
{
  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperExplicitEuler_impl<VectorType, SparseMatrixType, dim>::run_segment(
      AbstractFlowingVariables<NumberType, VectorType> &initial_condition, double start, double stop)
  {

    const SparseMatrixType &mass_matrix = assembler.get_mass_matrix();
    InverseSparseMatrixType inverse_mass_matrix;
    inverse_mass_matrix.initialize(mass_matrix);

    // Start out by saving step 0
    assembler.set_time(start);

    VectorType old_solution = initial_condition.spatial_data();
    VectorType solution = initial_condition.spatial_data();

    data_out.write_frame(start, [&](auto &frame) { assembler.attach_data_output(frame, solution); });

    double last_save = start;
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

      t += dt;
      if ((t - last_save + 1e-4 * expl.dt) >= output_dt) {
        data_out.write_frame(t, [&](auto &frame) { assembler.attach_data_output(frame, solution); });
        last_save = t;
      }
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
