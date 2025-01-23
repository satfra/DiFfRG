// external libraries
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/sundials/arkode.h>
#include <deal.II/sundials/ida.h>

// DiFfRG
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/sundials_ida_arkode.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA_ARKode<VectorType, SparseMatrixType, dim, LinearSolver>::run(
      AbstractFlowingVariables<NumberType> *initial_condition, const double t_start, const double t_stop)
  {
    this->data_out = this->get_data_out();
    this->adaptor = this->get_adaptor();

    auto &full_data = initial_condition->data();
    if (full_data.n_blocks() != 1)
      run(full_data, t_start, t_stop);
    else
      throw std::runtime_error("TimeStepperSUNDIALS_IDA_ARKode::run: initial condition must have at least two blocks!");
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA_ARKode<VectorType, SparseMatrixType, dim, LinearSolver>::run(
      BlockVectorType &initial_data, const double t_start, const double t_stop)
  {
    if (initial_data.n_blocks() != 2)
      throw std::runtime_error("TimeStepperSUNDIALS_ARKode_vars::run: y must have two blocks!");
    if (initial_data.block(1).size() == 0)
      throw std::runtime_error(
          "TimeStepperSUNDIALS_ARKode_vars::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    SparseMatrixType spatial_jacobian(assembler->get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver;
    const uint n_FE_dofs = initial_data.block(0).size();

    // Create a SUNDIALS IDA object with the right settings for spatial data
    typename SUNDIALS::IDA<VectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5,
                                                                20, 0, impl.abs_tol, impl.rel_tol);
    typename SUNDIALS::IDA<VectorType> time_stepper(ida_data);

    // Create a SUNDIALS ARKode object with the right settings for the variables
    const auto add_data = typename SUNDIALS::ARKode<VectorType>::AdditionalData(
        t_start, t_stop, expl.dt, t_stop, expl.minimal_dt, 8, 0, false, false, true, 3, expl.abs_tol, expl.rel_tol);
    SUNDIALS::ARKode<VectorType> variable_solver(add_data);
    const VectorType *spatial_y_buffer = nullptr;
    double last_t = 0.;
    variable_solver.explicit_function = [&](const double t, const VectorType &y, VectorType &ydot) {
      const auto now = std::chrono::high_resolution_clock::now();

      ydot = 0;
      assembler->set_time(t);
      assembler->residual_variables(ydot, y, *spatial_y_buffer);
      ydot *= -1.;

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "explicit residual", 1, ms_passed);
    };
    auto update_variables = [&](VectorType &variable_y, const VectorType &spatial_y, const double t) {
      if (last_t >= t || is_close(t, 0.)) return;
      const auto now = std::chrono::high_resolution_clock::now();

      spatial_y_buffer = &spatial_y;
      variable_solver.solve_ode_incrementally(variable_y, t);
      assembler->set_time(t);
      last_t = t;

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "update variables", 1, ms_passed);
    };

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = 0.;
    uint failure_counter = 0;

    // Initialize initial condition
    VectorType spatial_y = initial_data.block(0);
    VectorType spatial_y_dot = initial_data.block(0);
    VectorType variable_y = initial_data.block(1);

    // Pointer to current residual for monitoring
    VectorType *residual;

    // Tells SUNDIALS to do an internal reset, e.g. if we do local refinement
    time_stepper.solver_should_restart = [&](const double t, VectorType &sol, VectorType &sol_dot) -> bool {
      if ((*adaptor)(t, sol)) {
        assembler->reinit_vector(sol_dot);
        spatial_jacobian.reinit(assembler->get_sparsity_pattern_jacobian());
        return true;
      }
      return false;
    };

    time_stepper.differential_components = [&]() {
      IndexSet dof_indices = assembler->get_differential_indices();
      IndexSet differential_indices(n_FE_dofs);
      differential_indices.add_indices(dof_indices.begin(), dof_indices.end());
      return differential_indices;
    };

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](VectorType &v) { assembler->reinit_vector(v); };

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    time_stepper.output_step = [&](const double t, const VectorType &sol, const VectorType &sol_dot,
                                   uint /*step_number*/) {
      if (!is_close(last_save, t, 1e-10)) {
        assembler->set_time(t);

        assembler->attach_data_output(*data_out, sol, variable_y, sol_dot, (*residual));
        data_out->flush(t);

        last_save = t;
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &res) -> int {
      const auto now = std::chrono::high_resolution_clock::now();

      if (is_close(t, stuck_t, impl.minimal_dt * 1e-1))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (!is_close(t, 0.) && stuck > 50)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (is_close(t, 0.) && stuck > 200)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (failure_counter > 200) {
        std::cerr << "timestep failure, at t = " << t << std::endl;
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!std::isfinite(y.l1_norm())) {
        std::cerr << "residual: y0 is not finite" << std::endl;
        return ++failure_counter;
      }

      try {
        res = 0;
        assembler->set_time(t);
        update_variables(variable_y, y, t);
        assembler->residual(res, y, 1., y_dot, 1., variable_y);
        residual = &res;
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return ++failure_counter;
      }

      if (!std::isfinite(res.l1_norm())) {
        std::cerr << "residual: res0 is not finite" << std::endl;
        return ++failure_counter;
      }

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "implicit residual", 1, ms_passed);

      failure_counter = 0;
      return 0;
    };
    // Calculate the jacobian d(y_dot + F(y))/dy + d(y_dot*alpha)/dy_dot
    time_stepper.setup_jacobian = [&](const double t, const VectorType &y, const VectorType &y_dot,
                                      const double alpha) -> int {
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      try {
        auto now = std::chrono::high_resolution_clock::now();

        spatial_jacobian = 0;
        assembler->set_time(t);
        update_variables(variable_y, y, t);
        assembler->jacobian(spatial_jacobian, y, 1., y_dot, alpha, 1., variable_y);
        linSolver.init(spatial_jacobian);

        const auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(t, "jacobian construction", 1, ms_passed);
        now = std::chrono::high_resolution_clock::now();

        if (linSolver.invert()) {
          const auto ms_passed =
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                  .count();
          console_out(t, "jacobian inversion", 2, ms_passed);
        }
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return ++failure_counter;
      }

      failure_counter = 0;
      return 0;
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const VectorType &src, VectorType &dst, const double tol) -> int {
      try {
        const auto now = std::chrono::high_resolution_clock::now();
        const auto sol_iterations = linSolver.solve(src, dst, tol);
        if (sol_iterations >= 0) {
          const auto ms_passed =
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                  .count();
          console_out(stuck_t, "linear solver (" + std::to_string(sol_iterations) + " it)", 2, ms_passed);
        }
      } catch (std::exception &) {
        return ++failure_counter;
      }
      return 0;
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(spatial_y, spatial_y_dot);
    } catch (const std::exception &e) {
      spdlog::error("Timestepping failed: {}", e.what());
      time_stepper.output_step(stuck_t, spatial_y, spatial_y_dot, 0);
      throw;
    }

    initial_data.block(0) = spatial_y;
    initial_data.block(1) = variable_y;
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                      DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                      DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                      DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                      DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                      DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                      DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                                      DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                                      DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                                      DiFfRG::GMRES>;

template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                                      DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                                      DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA_ARKode<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                                      DiFfRG::GMRES>;