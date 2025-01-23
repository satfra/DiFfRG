// external libraries
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/sundials/ida.h>

// DiFfRG
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/data_output.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/sundials_ida.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, LinearSolver>::run(
      AbstractFlowingVariables<NumberType> *initial_condition, const double t_start, const double t_stop)
  {
    this->data_out = this->get_data_out();
    this->adaptor = this->get_adaptor();

    auto &full_data = initial_condition->data();
    if constexpr (dim == 0)
      run_vars(full_data.block(1), t_start, t_stop);
    else {
      if (full_data.n_blocks() != 1)
        run(full_data, t_start, t_stop);
      else
        run(full_data.block(0), t_start, t_stop);
    }
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, LinearSolver>::run(VectorType &initial_data,
                                                                                     const double t_start,
                                                                                     const double t_stop)
  {
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    SparseMatrixType jacobian(assembler->get_sparsity_pattern_jacobian());
    LinearSolver<SparseMatrixType, VectorType> linSolver;

    // Create a SUNDIALS IDA object with the right settings
    typename SUNDIALS::IDA<VectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5,
                                                                10, 0, impl.abs_tol, impl.rel_tol);
    typename SUNDIALS::IDA<VectorType> time_stepper(ida_data);

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = 0.;
    uint failure_counter = 0;

    // Initialize initial condition
    VectorType y = initial_data;
    VectorType y_dot = initial_data;
    y_dot *= 0.;

    // Pointer to current residual for monitoring
    VectorType *residual;

    // Tells SUNDIALS to do an internal reset, e.g. if we do local refinement
    time_stepper.solver_should_restart = [&](const double t, VectorType &sol, VectorType &sol_dot) -> bool {
      if ((*adaptor)(t, sol)) {
        assembler->reinit_vector(sol_dot);
        jacobian.reinit(assembler->get_sparsity_pattern_jacobian());
        return true;
      }
      return false;
    };

    time_stepper.differential_components = [&]() { return assembler->get_differential_indices(); };

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](VectorType &v) { assembler->reinit_vector(v); };

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    time_stepper.output_step = [&](const double t, const VectorType &sol, const VectorType &sol_dot,
                                   unsigned int /*step_number*/) {
      if (!is_close(last_save, t, 1e-10)) {
        assembler->set_time(t);
        assembler->attach_data_output(*data_out, sol, Vector<double>(), sol_dot, (*residual));
        data_out->flush(t);

        last_save = t;
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &res) -> int {
      const auto now = std::chrono::high_resolution_clock::now();

      if (is_close(t, stuck_t))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (!is_close(t, 0.) && stuck > 100)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (is_close(t, 0.) && stuck > 200)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (failure_counter > 200) throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      if (!std::isfinite(y.l1_norm())) return ++failure_counter;

      assembler->set_time(t);

      res = 0;
      assembler->residual(res, y, 1., y_dot, 1.);
      residual = &res;

      if (!std::isfinite(res.l1_norm())) return ++failure_counter;

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

      assembler->set_time(t);

      try {
        auto now = std::chrono::high_resolution_clock::now();

        jacobian = 0;
        assembler->jacobian(jacobian, y, 1., y_dot, alpha, 1.);
        linSolver.init(jacobian);

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
      time_stepper.solve_dae(y, y_dot);
    } catch (const std::exception &e) {
      spdlog::get("log")->error("Timestepping failed: {}", e.what());
      time_stepper.output_step(stuck_t, y, y_dot, 0);
      throw;
    }

    initial_data = y;
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, LinearSolver>::run(BlockVectorType &initial_data,
                                                                                     const double t_start,
                                                                                     const double t_stop)
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
    const uint n_vars = initial_data.block(1).size();
    FullMatrix<NumberType> variable_jacobian(n_vars);
    FullMatrix<NumberType> variable_jacobian_inverse(n_vars);

    // Create a SUNDIALS IDA object with the right settings
    typename SUNDIALS::IDA<BlockVectorType>::AdditionalData ida_data(
        t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5, 20, 0, impl.abs_tol, impl.rel_tol);
    typename SUNDIALS::IDA<BlockVectorType> time_stepper(ida_data);

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = 0.;
    uint failure_counter = 0;

    // Initialize initial condition
    BlockVectorType y = initial_data;
    BlockVectorType y_dot = initial_data;
    y_dot *= 0.;

    // Pointer to current residual for monitoring
    BlockVectorType *residual;

    // Tells SUNDIALS to do an internal reset, e.g. if we do local refinement
    time_stepper.solver_should_restart = [&](const double t, BlockVectorType &sol, BlockVectorType &sol_dot) -> bool {
      if ((*adaptor)(t, sol.block(0))) {
        assembler->reinit_vector(sol_dot.block(0));
        spatial_jacobian.reinit(assembler->get_sparsity_pattern_jacobian());
        return true;
      }
      return false;
    };

    time_stepper.differential_components = [&]() {
      IndexSet dof_indices = assembler->get_differential_indices();
      IndexSet differential_indices(n_FE_dofs + n_vars);
      differential_indices.add_indices(dof_indices.begin(), dof_indices.end());
      differential_indices.add_range(n_FE_dofs, n_FE_dofs + n_vars);
      return differential_indices;
    };

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](BlockVectorType &v) {
      v.reinit(2);
      assembler->reinit_vector(v.block(0));
      v.block(1).reinit(n_vars);
      v.collect_sizes();
    };

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    time_stepper.output_step = [&](const double t, const BlockVectorType &sol, const BlockVectorType &sol_dot,
                                   unsigned int /*step_number*/) {
      if (!is_close(last_save, t, 1e-10)) {
        assembler->set_time(t);
        assembler->attach_data_output(*data_out, sol.block(0), sol.block(1), sol_dot.block(0), (*residual).block(0));
        data_out->flush(t);

        last_save = t;
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const BlockVectorType &y, const BlockVectorType &y_dot,
                                BlockVectorType &res) -> int {
      const auto now = std::chrono::high_resolution_clock::now();

      if (is_close(t, stuck_t, impl.minimal_dt))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (!is_close(t, 0.) && stuck > 100)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (is_close(t, 0.) && stuck > 200)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (failure_counter > 200) {
        std::cerr << "timestep failure, at t = " << t << std::endl;
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!std::isfinite(y.l1_norm())) {
        if (!std::isfinite(y.block(0).l1_norm())) std::cerr << "residual: y0 is not finite" << std::endl;
        if (!std::isfinite(y.block(1).l1_norm())) std::cerr << "residual: y1 is not finite" << std::endl;
        return ++failure_counter;
      }

      try {
        res = 0;
        assembler->set_time(t);
        assembler->residual_variables(res.block(1), y.block(1), y.block(0));
        assembler->residual(res.block(0), y.block(0), 1., y_dot.block(0), 1., y.block(1));
        res.block(1) += y_dot.block(1);
        residual = &res;
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return ++failure_counter;
      }

      if (!std::isfinite(res.l1_norm())) {
        if (!std::isfinite(res.block(0).l1_norm())) std::cerr << "residual: res0 is not finite" << std::endl;
        if (!std::isfinite(res.block(1).l1_norm())) std::cerr << "residual: res1 is not finite" << std::endl;
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
    time_stepper.setup_jacobian = [&](const double t, const BlockVectorType &y, const BlockVectorType &y_dot,
                                      const double alpha) -> int {
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      try {
        auto now = std::chrono::high_resolution_clock::now();

        spatial_jacobian = 0;
        variable_jacobian = 0;
        assembler->set_time(t);
        assembler->jacobian(spatial_jacobian, y.block(0), 1., y_dot.block(0), alpha, 1., y.block(1));
        assembler->jacobian_variables(variable_jacobian, y.block(1), y.block(0));
        variable_jacobian *= -1.;
        variable_jacobian.diagadd(alpha);

        linSolver.init(spatial_jacobian);

        auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(t, "jacobian construction", 1, ms_passed);
        now = std::chrono::high_resolution_clock::now();

        linSolver.invert();
        variable_jacobian_inverse.invert(variable_jacobian);
        ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        console_out(t, "jacobian inversion", 2, ms_passed);

        if (!std::isfinite(spatial_jacobian.frobenius_norm())) {
          std::cerr << "spatial_jacobian is not finite" << std::endl;
          return ++failure_counter;
        }
        if (!std::isfinite(variable_jacobian.frobenius_norm())) {
          std::cerr << "variable_jacobian is not finite" << std::endl;
          return ++failure_counter;
        }
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return ++failure_counter;
      }

      failure_counter = 0;
      return 0;
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const BlockVectorType &src, BlockVectorType &dst, const double tol) -> int {
      try {
        const auto now = std::chrono::high_resolution_clock::now();

        const auto sol_iterations = linSolver.solve(src.block(0), dst.block(0), tol);
        variable_jacobian_inverse.vmult(dst.block(1), src.block(1));

        const auto ms_passed =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
                .count();
        if (sol_iterations >= 0)
          console_out(stuck_t, "linear solver (" + std::to_string(sol_iterations) + " it)", 2, ms_passed);
        else
          console_out(stuck_t, "linear solver", 2, ms_passed);
      } catch (std::exception &) {
        return ++failure_counter;
      }
      return 0;
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(y, y_dot);
    } catch (const std::exception &e) {
      spdlog::get("log")->error("Timestepping failed: {}", e.what());
      time_stepper.output_step(stuck_t, y, y_dot, 0);
      throw;
    }

    initial_data = y;
  }

  template <typename VectorType, typename SparseMatrixType, uint dim,
            template <typename, typename> typename LinearSolver>
  void TimeStepperSUNDIALS_IDA<VectorType, SparseMatrixType, dim, LinearSolver>::run_vars(VectorType &initial_data,
                                                                                          const double t_start,
                                                                                          const double t_stop)
  {
    if (initial_data.size() == 0)
      throw std::runtime_error("TimeStepperSUNDIALS_IDA::run: y contains no variables, use a different timestepper!");
    // Start by setting up all needed matrices, i.e. jacobian, inverse of jacobian and the mass matrix (with two
    // sparsity patterns)
    const uint n_vars = initial_data.size();
    FullMatrix<NumberType> variable_jacobian(n_vars);
    FullMatrix<NumberType> variable_jacobian_inverse(n_vars);

    // Create a SUNDIALS IDA object with the right settings
    typename SUNDIALS::IDA<VectorType>::AdditionalData ida_data(t_start, t_stop, impl.dt, output_dt, impl.minimal_dt, 5,
                                                                20, 0, impl.abs_tol, impl.rel_tol);
    typename SUNDIALS::IDA<VectorType> time_stepper(ida_data);

    // Define some variables for monitoring
    uint stuck = 0;
    double stuck_t = 0.;
    uint failure_counter = 0;

    // Initialize initial condition
    VectorType y = initial_data;
    VectorType y_dot = initial_data;
    y_dot *= 0.;

    // Called whenever a vector needs to initalized
    time_stepper.reinit_vector = [&](VectorType &v) { v.reinit(n_vars); };

    // At output_dt intervals this function saves intermediate solutions
    double last_save = -1.;
    time_stepper.output_step = [&](const double t, const VectorType &sol, const VectorType & /*sol_dot*/,
                                   uint /*step_number*/) {
      if (!is_close(last_save, t, 1e-10)) {
        assembler->set_time(t);
        assembler->attach_data_output(*data_out, Vector<double>(), sol);
        data_out->flush(t);

        last_save = t;
      }
    };

    //  Calculate the residual of y_dot + F(y)
    time_stepper.residual = [&](const double t, const VectorType &y, const VectorType &y_dot, VectorType &res) -> int {
      const auto now = std::chrono::high_resolution_clock::now();

      if (is_close(t, stuck_t, impl.minimal_dt))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }

      if (!is_close(t, 0.) && stuck > 100)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (is_close(t, 0.) && stuck > 200)
        throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      if (failure_counter > 200) {
        std::cerr << "timestep failure, at t = " << t << std::endl;
        throw std::runtime_error("timestep failure, at t = " + std::to_string(t));
      }
      if (!std::isfinite(y.l1_norm())) {
        std::cerr << "vector is not finite" << std::endl;
        return ++failure_counter;
      }

      try {
        res = 0;
        assembler->set_time(t);
        assembler->residual_variables(res, y, Vector<double>());
        res += y_dot;
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return ++failure_counter;
      }

      if (!std::isfinite(res.l1_norm())) {
        std::cerr << "residual is not finite" << std::endl;
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
    time_stepper.setup_jacobian = [&](const double t, const VectorType &y, const VectorType & /*y_dot*/,
                                      const double alpha) -> int {
      if (failure_counter > 200) throw std::runtime_error("timestep failure at jacobian");

      const auto now = std::chrono::high_resolution_clock::now();

      try {
        variable_jacobian = 0;
        assembler->set_time(t);
        assembler->jacobian_variables(variable_jacobian, y, Vector<double>());
        variable_jacobian *= -1.;
        variable_jacobian.diagadd(alpha);

        variable_jacobian_inverse.invert(variable_jacobian);

        if (!std::isfinite(variable_jacobian.frobenius_norm())) {
          std::cerr << "variable_jacobian is not finite" << std::endl;
          return ++failure_counter;
        }
      } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return ++failure_counter;
      }

      const auto ms_passed =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - now)
              .count();
      console_out(t, "jacobian", 1, ms_passed);

      failure_counter = 0;
      return 0;
    };

    // Solve the linear system J dst = src
    time_stepper.solve_with_jacobian = [&](const VectorType &src, VectorType &dst, const double) -> int {
      try {
        variable_jacobian_inverse.vmult(dst, src);
      } catch (std::exception &) {
        return ++failure_counter;
      }
      return 0;
    };

    // Start the time loop
    try {
      time_stepper.solve_dae(y, y_dot);
    } catch (const std::exception &e) {
      spdlog::get("log")->error("Timestepping failed: {}", e.what());
      time_stepper.output_step(stuck_t, y, y_dot, 0);
      throw;
    }

    initial_data = y;
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 0,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 1,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 2,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 3,
                                               DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                               DiFfRG::UMFPack>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                               DiFfRG::UMFPack>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 0, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 1, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 2, DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::SparseMatrix<double>, 3, DiFfRG::GMRES>;

template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0,
                                               DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1,
                                               DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2,
                                               DiFfRG::GMRES>;
template class DiFfRG::TimeStepperSUNDIALS_IDA<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3,
                                               DiFfRG::GMRES>;