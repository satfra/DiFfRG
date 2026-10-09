// standard library
#include <algorithm>
#include <array>
#include <cmath>

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
#include <DiFfRG/timestepping/boost_rk.hh>
#include <deal.II/lac/vector.h>

namespace DiFfRG
{
  using namespace dealii;

  template <typename VectorType, typename SparseMatrixType, uint dim, int prec>
  void TimeStepperBoostRK_impl<VectorType, SparseMatrixType, dim, prec>::run_segment(
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

  template <typename VectorType, typename SparseMatrixType, uint dim, int prec>
  void TimeStepperBoostRK_impl<VectorType, SparseMatrixType, dim, prec>::run(VectorType &initial_data,
                                                                             const double t_start, const double t_stop)
  {
    const SparseMatrixType &mass_matrix = assembler.get_mass_matrix();
    InverseSparseMatrixType inverse_mass_matrix;
    inverse_mass_matrix.initialize(mass_matrix);

    // These are just buffers
    dealii::Vector<double> y_dealii(initial_data);
    dealii::Vector<double> dy_dealii(initial_data);

    // At output_dt intervals this function saves intermediate solutions
    dealii::Vector<double> output_dealii(initial_data);
    std::array<Eigen::VectorXd, 2> output_ring;
    std::array<double, 2> output_ring_times{};
    uint ring_count = 0;
    uint n_saves = 0;
    Eigen::VectorXd sol_save;
    auto output_step = [&](const Eigen::VectorXd &sol, const double t) {
      output_ring[ring_count % 2] = sol;
      output_ring_times[ring_count % 2] = t;
      ring_count++;

      sol_save = sol;

      while (n_saves <= uint((t - t_start) / output_dt)) {
        // Interpolate the solution to the output time
        double t_save = t_start + output_dt * n_saves;
        if (t_save > t) break;
        n_saves++;
        if (ring_count >= 2) {
          const uint cur = (ring_count - 1) % 2;
          const uint prev = (ring_count - 2) % 2;
          const double t_prev = output_ring_times[prev];
          const double t_next = output_ring_times[cur];
          const double alpha = (t_save - t_prev) / (t_next - t_prev);
          sol_save = alpha * output_ring[cur] + (1. - alpha) * output_ring[prev];
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
      if (expl.detect_stuck) {
        if (is_close(t, stuck_t, expl.minimal_dt / 100.))
          stuck++;
        else {
          stuck = 0;
          stuck_t = t;
        }
        if (stuck > 50) throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }

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
    auto stepper = make_controlled<error_stepper_type>(expl.abs_tol, expl.rel_tol);
    integrate_adaptive(stepper, residual, y_eigen, t_start, t_stop, expl.dt, output_step);

    eigen_to_dealii(y_eigen, initial_data);
    this->drain_output();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim, int prec>
  void TimeStepperBoostRK_impl<VectorType, SparseMatrixType, dim, prec>::run(BlockVectorType &initial_data,
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
    std::array<Eigen::VectorXd, 2> output_ring;
    std::array<double, 2> output_ring_times{};
    uint ring_count = 0;
    uint n_saves = 0;
    Eigen::VectorXd sol_save;
    auto output_step = [&](const Eigen::VectorXd &sol, const double t) {
      output_ring[ring_count % 2] = sol;
      output_ring_times[ring_count % 2] = t;
      ring_count++;

      sol_save = sol;

      while (n_saves <= uint((t - t_start) / output_dt)) {
        // Interpolate the solution to the output time
        double t_save = t_start + output_dt * n_saves;
        if (t_save > t) break;
        n_saves++;
        if (ring_count >= 2) {
          const uint cur = (ring_count - 1) % 2;
          const uint prev = (ring_count - 2) % 2;
          const double t_prev = output_ring_times[prev];
          const double t_next = output_ring_times[cur];
          const double alpha = (t_save - t_prev) / (t_next - t_prev);
          sol_save = alpha * output_ring[cur] + (1. - alpha) * output_ring[prev];
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
      if (expl.detect_stuck) {
        if (is_close(t, stuck_t, expl.minimal_dt / 100.))
          stuck++;
        else {
          stuck = 0;
          stuck_t = t;
        }
        if (stuck > 50) throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }

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
    auto stepper = make_controlled<error_stepper_type>(expl.abs_tol, expl.rel_tol);
    integrate_adaptive(stepper, residual, y_eigen, t_start, t_stop, expl.dt, output_step);

    eigen_to_dealii(y_eigen, initial_data);
    this->drain_output();
  }

  template <typename VectorType, typename SparseMatrixType, uint dim, int prec>
  void TimeStepperBoostRK_impl<VectorType, SparseMatrixType, dim, prec>::run_vars(VectorType &initial_data,
                                                                                  const double t_start,
                                                                                  const double t_stop)
  {
    if (initial_data.size() == 0)
      throw std::runtime_error("TimeStepperRK_impl::run: y contains no variables, use a different timestepper!");

    // At output_dt intervals this function saves intermediate solutions
    dealii::Vector<double> output_dealii(initial_data.size());
    std::array<Eigen::VectorXd, 2> output_ring;
    std::array<double, 2> output_ring_times{};
    uint ring_count = 0;
    uint n_saves = 0;
    Eigen::VectorXd sol_save;
    auto output_step = [&](const Eigen::VectorXd &sol, const double t) {
      output_ring[ring_count % 2] = sol;
      output_ring_times[ring_count % 2] = t;
      ring_count++;

      sol_save = sol;

      while (n_saves <= uint((t - t_start) / output_dt)) {
        // Interpolate the solution to the output time
        double t_save = t_start + output_dt * n_saves;
        if (t_save > t) break;
        n_saves++;
        if (ring_count >= 2) {
          const uint cur = (ring_count - 1) % 2;
          const uint prev = (ring_count - 2) % 2;
          const double t_prev = output_ring_times[prev];
          const double t_next = output_ring_times[cur];
          const double alpha = (t_save - t_prev) / (t_next - t_prev);
          sol_save = alpha * output_ring[cur] + (1. - alpha) * output_ring[prev];
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
      if (expl.detect_stuck) {
        if (is_close(t, stuck_t, expl.minimal_dt / 100.))
          stuck++;
        else {
          stuck = 0;
          stuck_t = t;
        }
        if (stuck > 50) throw std::runtime_error("timestepping got stuck at t = " + std::to_string(t));
      }

      eigen_to_dealii(x, y_dealii);

      dy_dealii = 0;

      assembler.set_time(t);
      assembler.residual_variables(dy_dealii, y_dealii, Vector<double>());

      if (!std::isfinite(dy_dealii.l2_norm()))
        throw std::runtime_error("TimeStepperBoostRK_impl::run_vars: dy is not finite!");

      dealii_to_eigen(dy_dealii, dxdt);
      dxdt *= -1;

      this->log.progress({.topic = progress_topics::explicit_residual,
                          .time = t,
                          .duration_ms = calc_timer.lap(),
                          .minimum_verbosity = 1});
    };

    using namespace boost::numeric::odeint;
    auto stepper = make_controlled<error_stepper_type>(expl.abs_tol, expl.rel_tol);
    // integrate_adaptive(stepper, residual, y_eigen, t_start, t_stop, expl.dt, output_step);

    uint step = 0;
    double cur_dt = expl.dt;
    double step_time = t_start;
    failed_step_checker fail_checker; // to throw a runtime_error if step size adjustment fails
    output_step(y_eigen, step_time);
    while (t_stop - step_time > 1e-12 * std::max(1., std::abs(t_stop))) {
      controlled_step_result res;
      // Never step past t_stop: run() hands the state at t_stop to the next segment or a snapshot.
      cur_dt = std::min({cur_dt, expl.maximal_dt, t_stop - step_time});
      do {
        res = stepper.try_step(residual, y_eigen, step_time, cur_dt);
        fail_checker(); // check number of failed steps
      } while (res == fail);

      step++;
      output_step(y_eigen, step_time);
      fail_checker.reset(); // if we reach here, the step was successful -> reset fail checker
    }

    this->log.info("TimeStepperBoostRK_impl::run_vars: finished after {} steps", step);

    eigen_to_dealii(y_eigen, initial_data);
    this->drain_output();
  }
} // namespace DiFfRG

template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 0, 0>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1, 0>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2, 0>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3, 0>;

template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 0, 1>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 1, 1>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 2, 1>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::SparseMatrix<double>, 3, 1>;

template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0, 0>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1, 0>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2, 0>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3, 0>;

template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 0, 1>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 1, 1>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 2, 1>;
template class DiFfRG::TimeStepperBoostRK_impl<dealii::Vector<double>, dealii::BlockSparseMatrix<double>, 3, 1>;
