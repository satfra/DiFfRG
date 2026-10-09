// standard library
#include <algorithm>
#include <cmath>
#include <random>

// external libraries
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/common/eigen.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/common/abstract_adaptor.hh>
#include <DiFfRG/discretization/common/abstract_assembler.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/timestepping/boost_abm.hh>

namespace DiFfRG
{
  using namespace dealii;

  namespace
  {
    /**
     * @brief The largest step <= dt that divides [t_start, t_stop] into equal steps.
     *
     * ABM is a fixed-step multistep method, so the last step cannot simply be shortened. Stepping
     * with dt itself would overshoot t_stop by up to one step whenever the interval is not a
     * multiple of it, and run() relies on the state at t_stop being exactly that.
     */
    double uniform_step(const double t_start, const double t_stop, const double dt)
    {
      const double length = t_stop - t_start;
      if (!(length > 0.)) return dt;
      const double n_steps = std::max(1., std::ceil(length / dt - 1e-9));
      return length / n_steps;
    }
  } // namespace

  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperBoostABM_impl<VectorType, SparseMatrixType, dim>::run_segment(
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
  void TimeStepperBoostABM_impl<VectorType, SparseMatrixType, dim>::run(VectorType &initial_data, const double t_start,
                                                                        const double t_stop)
  {
    const SparseMatrixType &mass_matrix = assembler.get_mass_matrix();
    InverseSparseMatrixType inverse_mass_matrix;
    inverse_mass_matrix.initialize(mass_matrix);

    // These are just buffers
    dealii::Vector<double> y_dealii(initial_data);
    dealii::Vector<double> dy_dealii(initial_data);

    // At output_dt intervals this function saves intermediate solutions
    dealii::Vector<double> output_dealii(initial_data);
    std::vector<Eigen::VectorXd> output_vec;
    std::vector<double> output_times;
    uint n_saves = 0;
    Eigen::VectorXd sol_save;
    auto output_step = [&](const Eigen::VectorXd &sol, const double t) {
      output_vec.push_back(sol);
      output_times.push_back(t);

      sol_save = sol;

      while (n_saves <= uint((t - t_start + 1e-3 * output_dt) / output_dt)) {
        // Interpolate the solution to the output time
        double t_save = t_start + output_dt * n_saves;
        if (t_save > t) break;
        n_saves++;
        if (output_times.size() >= 2) {
          const double t_prev = output_times[output_times.size() - 2];
          const double t_next = output_times[output_times.size() - 1];
          const double alpha = (t_save - t_prev) / (t_next - t_prev);
          sol_save = alpha * output_vec[output_times.size() - 1] + (1. - alpha) * output_vec[output_times.size() - 2];
        }

        this->log.progress({.topic = progress_topics::output, .time = t_save, .minimum_verbosity = 3});
        eigen_to_dealii(sol_save, output_dealii);

        assembler.set_time(t_save);

        data_out.write_frame(t_save, [&](auto &frame) {
          assembler.attach_data_output(frame, output_dealii, Vector<double>(), dy_dealii);
        });
      }
    };

    double stuck_t = 0.;
    uint stuck = 0;
    auto residual = [&](const Eigen::VectorXd &x, Eigen::VectorXd &dxdt, const double t) {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t, expl.minimal_dt / 100.))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }
      if (stuck > 50) throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));

      eigen_to_dealii(x, y_dealii);

      dy_dealii = 0;

      assembler.set_time(t);
      assembler.residual(dy_dealii, y_dealii, -1., 0.);
      inverse_mass_matrix.solve(dy_dealii);

      if (!std::isfinite(dy_dealii.l2_norm()))
        throw std::runtime_error("TimeStepperBoostRK_impl::run_vars: dy is not finite!");

      dealii_to_eigen(dy_dealii, dxdt);

      this->log.progress({.topic = progress_topics::explicit_residual,
                          .time = t,
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1});
    };

    // Initialize initial condition
    Eigen::VectorXd y_eigen(initial_data.size());
    dealii_to_eigen(initial_data, y_eigen);

    using namespace boost::numeric::odeint;
    adams_bashforth_moulton<8, Eigen::VectorXd> abm;
    const double cur_dt = uniform_step(t_start, t_stop, expl.dt);
    double step_time = t_start;
    output_step(y_eigen, step_time);

    uint step = 0;
    while (step_time <= t_stop && !is_close(step_time, t_stop, 1e-3 * cur_dt)) {
      abm.do_step(residual, y_eigen, step_time, cur_dt);
      step++;
      step_time += cur_dt;
      output_step(y_eigen, step_time);
    }

    this->log.info("TimeStepperBoostABM_impl::run: finished after {} steps", step);

    eigen_to_dealii(y_eigen, initial_data);
    this->drain_output();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperBoostABM_impl<VectorType, SparseMatrixType, dim>::run(BlockVectorType &initial_data,
                                                                        const double t_start, const double t_stop)
  {
    if (initial_data.n_blocks() != 2) throw std::runtime_error("TimeStepperBoostRK_impl::run: y must have two blocks!");
    if (initial_data.block(1).size() == 0)
      throw std::runtime_error("TimeStepperBoostRK_impl::run: y contains no variables, use a different timestepper!");

    const SparseMatrixType &mass_matrix = assembler.get_mass_matrix();
    InverseSparseMatrixType inverse_mass_matrix;
    inverse_mass_matrix.initialize(mass_matrix);

    // These are just buffers
    dealii::BlockVector<double> y_dealii(initial_data);
    dealii::BlockVector<double> dy_dealii(initial_data);

    // At output_dt intervals this function saves intermediate solutions
    dealii::BlockVector<double> output_dealii(initial_data);
    std::vector<Eigen::VectorXd> output_vec;
    std::vector<double> output_times;
    uint n_saves = 0;
    Eigen::VectorXd sol_save;
    auto output_step = [&](const Eigen::VectorXd &sol, const double t) {
      output_vec.push_back(sol);
      output_times.push_back(t);

      sol_save = sol;

      while (n_saves <= uint((t - t_start + 1e-3 * output_dt) / output_dt)) {
        // Interpolate the solution to the output time
        double t_save = t_start + output_dt * n_saves;
        if (t_save > t) break;
        n_saves++;
        if (output_times.size() >= 2) {
          const double t_prev = output_times[output_times.size() - 2];
          const double t_next = output_times[output_times.size() - 1];
          const double alpha = (t_save - t_prev) / (t_next - t_prev);
          sol_save = alpha * output_vec[output_times.size() - 1] + (1. - alpha) * output_vec[output_times.size() - 2];
        }

        this->log.progress({.topic = progress_topics::output, .time = t_save, .minimum_verbosity = 3});
        eigen_to_dealii(sol_save, output_dealii);

        assembler.set_time(t_save);

        data_out.write_frame(t_save, [&](auto &frame) {
          assembler.attach_data_output(frame, output_dealii.block(0), output_dealii.block(1), dy_dealii.block(0));
        });
      }
    };

    double stuck_t = 0.;
    uint stuck = 0;
    auto residual = [&](const Eigen::VectorXd &x, Eigen::VectorXd &dxdt, const double t) {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t, expl.minimal_dt / 100.))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }
      if (stuck > 50) throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));

      eigen_to_dealii(x, y_dealii);

      dy_dealii = 0;
      assembler.set_time(t);
      assembler.residual_variables(dy_dealii.block(1), y_dealii.block(1), y_dealii.block(0));
      dy_dealii.block(1) *= -1;
      assembler.residual(dy_dealii.block(0), y_dealii.block(0), -1., 0., y_dealii.block(1));
      inverse_mass_matrix.solve(dy_dealii.block(0));

      if (!std::isfinite(dy_dealii.l2_norm()))
        throw std::runtime_error("TimeStepperBoostRK_impl::run_vars: dy is not finite!");

      dealii_to_eigen(dy_dealii, dxdt);

      this->log.progress({.topic = progress_topics::explicit_residual,
                          .time = t,
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1});
    };

    // Initialize initial condition
    Eigen::VectorXd y_eigen(initial_data.size());
    dealii_to_eigen(initial_data, y_eigen);

    using namespace boost::numeric::odeint;
    adams_bashforth_moulton<8, Eigen::VectorXd> abm;
    const double cur_dt = uniform_step(t_start, t_stop, expl.dt);
    double step_time = t_start;
    output_step(y_eigen, step_time);

    uint step = 0;
    while (step_time <= t_stop && !is_close(step_time, t_stop, 1e-3 * cur_dt)) {
      abm.do_step(residual, y_eigen, step_time, cur_dt);
      step++;
      step_time += cur_dt;
      output_step(y_eigen, step_time);
    }

    this->log.info("TimeStepperBoostABM_impl::run: finished after {} steps", step);

    eigen_to_dealii(y_eigen, initial_data);
    this->drain_output();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim>
  void TimeStepperBoostABM_impl<VectorType, SparseMatrixType, dim>::run_vars(VectorType &initial_data,
                                                                             const double t_start, const double t_stop)
  {
    if (initial_data.size() == 0)
      throw std::runtime_error("TimeStepperRK_impl::run: y contains no variables, use a different timestepper!");

    // At output_dt intervals this function saves intermediate solutions
    dealii::Vector<double> output_dealii(initial_data.size());
    std::vector<Eigen::VectorXd> output_vec;
    std::vector<double> output_times;
    uint n_saves = 0;
    Eigen::VectorXd sol_save;
    auto output_step = [&](const Eigen::VectorXd &sol, const double t) {
      output_vec.push_back(sol);
      output_times.push_back(t);

      sol_save = sol;

      while (n_saves <= uint((t - t_start) / output_dt + 1e-3)) {
        // Interpolate the solution to the output time
        double t_save = t_start + output_dt * n_saves;
        if (!is_close(t_save, t, 1e-6 * output_dt) && t_save > t) break;
        n_saves++;
        if (output_times.size() >= 2) {
          const double t_prev = output_times[output_times.size() - 2];
          const double t_next = output_times[output_times.size() - 1];
          const double alpha = (t_save - t_prev) / (t_next - t_prev);
          sol_save = alpha * output_vec[output_times.size() - 1] + (1. - alpha) * output_vec[output_times.size() - 2];
        }

        this->log.progress({.topic = progress_topics::output, .time = t_save, .minimum_verbosity = 3});
        eigen_to_dealii(sol_save, output_dealii);

        assembler.set_time(t_save);

        data_out.write_frame(
            t_save, [&](auto &frame) { assembler.attach_data_output(frame, Vector<double>(), output_dealii); });
      }
    };

    // Initialize initial condition
    Eigen::VectorXd y_eigen(initial_data.size());
    dealii_to_eigen(initial_data, y_eigen);

    // These are just buffers
    dealii::Vector<double> y_dealii(initial_data.size());
    dealii::Vector<double> dy_dealii(initial_data.size());

    double stuck_t = 0.;
    uint stuck = 0;
    auto residual = [&](const Eigen::VectorXd &x, Eigen::VectorXd &dxdt, const double t) {
      CalcDtTimer calc_timer;
      if (is_close(t, stuck_t, expl.minimal_dt / 100.))
        stuck++;
      else {
        stuck = 0;
        stuck_t = t;
      }
      if (stuck > 50) throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));

      eigen_to_dealii(x, y_dealii);

      dy_dealii = 0;

      assembler.set_time(t);
      assembler.residual_variables(dy_dealii, y_dealii, Vector<double>());

      if (!std::isfinite(dy_dealii.l2_norm()))
        throw std::runtime_error("TimeStepperBoostABM_impl::run_vars: dy is not finite!");

      dealii_to_eigen(dy_dealii, dxdt);
      dxdt *= -1;

      this->log.progress({.topic = progress_topics::explicit_residual,
                          .time = t,
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1});
    };

    using namespace boost::numeric::odeint;
    adams_bashforth_moulton<8, Eigen::VectorXd> abm;
    const double cur_dt = uniform_step(t_start, t_stop, expl.dt);
    double step_time = t_start;
    output_step(y_eigen, step_time);

    uint step = 0;
    while (step_time <= t_stop && !is_close(step_time, t_stop, 1e-3 * cur_dt)) {
      abm.do_step(residual, y_eigen, step_time, cur_dt);
      step++;
      step_time += cur_dt;
      output_step(y_eigen, step_time);
    }

    this->log.info("TimeStepperBoostABM_impl::run_vars: finished after {} steps", step);

    eigen_to_dealii(y_eigen, initial_data);
    this->drain_output();
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperBoostABM_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 0>;
template class DiFfRG::TimeStepperBoostABM_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1>;
template class DiFfRG::TimeStepperBoostABM_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2>;
template class DiFfRG::TimeStepperBoostABM_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3>;

template class DiFfRG::TimeStepperBoostABM_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0>;
template class DiFfRG::TimeStepperBoostABM_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1>;
template class DiFfRG::TimeStepperBoostABM_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2>;
template class DiFfRG::TimeStepperBoostABM_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3>;
