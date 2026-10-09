// external libraries
#include <deal.II/base/config.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/block_vector.h>

// DiFfRG
#include <DiFfRG/timestepping/solver/kinsol.hh>

namespace DiFfRG
{
  using namespace dealii;
  template <typename VectorType_>
  KINSOL<VectorType_>::KINSOL(double abstol_, double reltol_, double assemble_threshold_, uint max_steps_,
                              uint n_stepsize_iterations_)
      : abstol(abstol_), reltol(reltol_), assemble_threshold(assemble_threshold_), max_steps(max_steps_),
        n_stepsize_iterations(n_stepsize_iterations_), ignore_nonconv(false)
  {
    typename SUNDIALS::KINSOL<VectorType>::AdditionalData add_data;
    add_data.strategy = SUNDIALS::KINSOL<VectorType>::AdditionalData::SolutionStrategy::linesearch;
    add_data.function_tolerance = abstol;
    add_data.maximum_non_linear_iterations = max_steps;
    add_data.no_init_setup = true;
    // SUNDIALS 7.x defaults the maximum scaled Newton step to 1000*||u_scale||_2 when the initial
    // guess is zero, which can be smaller than the actual Newton step and makes the line search
    // clamp the step and fail with KIN_MXNEWT_5X_EXCEEDED. Disable the bound (a large value) so the
    // full Newton step is taken; globalization is still provided by the line search itself.
    add_data.maximum_newton_step = 1e10;
    kinsol = std::make_shared<SUNDIALS::KINSOL<VectorType>>(add_data);
  }

  template <typename VectorType_> void KINSOL<VectorType_>::reinit(const VectorType &u) { update_jacobian(u); }

  template <typename VectorType_> void KINSOL<VectorType_>::operator()(VectorType &iterate)
  {
    Timer timer;

    initial_vector = iterate;

    kinsol->reinit_vector = [&](VectorType &x) { x.reinit(iterate); };

    kinsol->residual = [&](const VectorType &evaluation_point, VectorType &residual) {
      this->residual(residual, evaluation_point);
      residual_vector = residual;
      return 0;
    };

    kinsol->setup_jacobian = [&](const VectorType &current_u, const VectorType & /*current_f*/) {
      this->update_jacobian(current_u);
      return 0;
    };

    kinsol->solve_with_jacobian = [&](const VectorType &rhs, VectorType &dst, const double /*tolerance*/) {
      this->lin_solve(dst, rhs);
      return 0;
    };

    // Root-mean-square residual at the given iterate, and a matching convergence test against the
    // requested absolute/relative tolerances.
    auto residual_converged = [&](VectorType &it) {
      VectorType res;
      res.reinit(it);
      this->residual(res, it);
      residual_vector = res;
      using std::max, std::sqrt;
      const double rms = res.l2_norm() / sqrt(static_cast<double>(res.size()));
      const double it_rms = it.l2_norm() / sqrt(static_cast<double>(it.size()));
      return rms <= max(abstol, reltol * it_rms);
    };

    // SUNDIALS' linesearch strategy can report KIN_LINESEARCH_NONCONV (-5) or KIN_MAXITER_REACHED
    // (-7) when Newton has effectively already reached the solution (e.g. a linear system solved in
    // a single step, or a solve started from an already-converged iterate): the step becomes too
    // small for the line search to make further progress. deal.II turns the negative return code
    // into an exception. We accept the iterate when its residual confirms convergence (or when the
    // user opted into ignore_nonconv) and re-throw genuine failures.
    auto robust_solve = [&](VectorType &it) {
      try {
        kinsol->solve(it);
      } catch (const dealii::ExceptionBase &) {
        if (!ignore_nonconv && !residual_converged(it)) throw;
      }
    };

    robust_solve(iterate);
    // Only attempt a second Newton pass when the first did not actually converge -- re-solving from
    // an already-converged iterate makes the line search stall (KIN_MAXITER_REACHED).
    if (!ignore_nonconv && !residual_converged(iterate)) robust_solve(iterate);
    iterate_vector = iterate;

    timings_newton.push_back(timer.wall_time());
  }

  template <typename VectorType_> double KINSOL<VectorType_>::threshold(const double thr)
  {
    const double t = assemble_threshold;
    assemble_threshold = thr;
    return t;
  }

  template <typename VectorType_> const VectorType_ &KINSOL<VectorType_>::get_residual() const
  {
    return residual_vector;
  }
  template <typename VectorType_> const VectorType_ &KINSOL<VectorType_>::get_initial() const { return initial_vector; }
  template <typename VectorType_> const VectorType_ &KINSOL<VectorType_>::get_iterate() const { return iterate_vector; }

  template <typename VectorType_> double KINSOL<VectorType_>::average_time_newton() const
  {
    double t = 0.;
    double n = timings_newton.size();
    for (const auto &t_ : timings_newton)
      t += t_ / n;
    return t;
  }
  template <typename VectorType_> unsigned int KINSOL<VectorType_>::num_newton_calls() const
  {
    return timings_newton.size();
  }

  template <typename VectorType_> void KINSOL<VectorType_>::set_ignore_nonconv(bool x) { ignore_nonconv = x; }

  template <typename VectorType_> double KINSOL<VectorType_>::get_error() { return resnorm = get_EEst(); }
  template <typename VectorType_> unsigned int KINSOL<VectorType_>::get_step() { return step; }
  template <typename VectorType_> unsigned int KINSOL<VectorType_>::get_jacobians() { return jacobians; }

  template <typename VectorType_> double KINSOL<VectorType_>::get_EEst()
  {
    const VectorType &err = residual_vector;
    const VectorType &u_prev = initial_vector;
    const VectorType &u = iterate_vector;

    eest_tmp.reinit(u);

    using std::abs, std::max, std::sqrt;
    tbb::parallel_for(tbb::blocked_range<unsigned int>(0, eest_tmp.size()), [&](tbb::blocked_range<unsigned int> r) {
      for (unsigned int n = r.begin(); n < r.end(); ++n) {
        eest_tmp[n] = abs(err[n]) / (abstol + max(abs(u_prev[n]), abs(u[n])) * reltol);
      }
    });
    return eest_tmp.l2_norm() / sqrt(eest_tmp.size());
  }

  template <typename VectorType_> bool KINSOL<VectorType_>::check()
  {
    if (step > max_steps) return false;
    if (step == 0) return true;
    double err = get_EEst();
    converged = err <= 1.;
    return !converged;
  }
} // namespace DiFfRG

template class DiFfRG::KINSOL<dealii::BlockVector<double>>;
template class DiFfRG::KINSOL<dealii::Vector<double>>;