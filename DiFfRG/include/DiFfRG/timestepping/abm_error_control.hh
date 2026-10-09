#pragma once

// external libraries
#include <Eigen/Dense>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>

// standard library
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Adams-Bashforth-Moulton predictor-corrector (PEC: Adams-Bashforth predictor, one Adams-Moulton
     * correction, both of order Steps) whose step size may change between steps, with a local error estimate on every
     * step.
     *
     * The stepper keeps the right-hand side at past points of the trajectory. When the step size changes, this history
     * is resampled onto the new step size: halving interpolates the polynomial through the stored points (the same
     * polynomial the Adams formulas integrate, so the order is kept), doubling picks every other point of a history
     * that is kept 2 * Steps long. Only while there are fewer than Steps points -- the first Steps - 1 steps of a run --
     * are steps taken by an embedded Runge-Kutta pair (Cash-Karp 5(4)) instead, whose error estimate stands in.
     *
     * Error estimate of a multistep step: with the Adams-Bashforth coefficients g_m = 1 - sum_{j<m} g_j / (m + 1 - j),
     * the predictor has error constant g_Steps and the corrector g*_Steps = g_Steps - g_{Steps-1}. The local error of the
     * corrected step is therefore milne_factor = |g*_Steps| / g_{Steps-1} times the predictor-corrector difference
     * (~1/33 at Steps = 8).
     *
     * do_step(system, x, t, dt) continues the trajectory from (x, t), which is either where the last step ended or,
     * after a step was discarded, where it started.
     */
    template <size_t Steps> class ABMWithErrorEstimate
    {
      static_assert(Steps >= 2);

    public:
      using State = Eigen::VectorXd;
      static constexpr size_t steps = Steps;

      /// Advance x from t by dt. The estimated local error of the step is error_estimate() afterwards.
      template <typename System> void do_step(System &&system, State &x, const double t, const double dt)
      {
        if (history.empty() || t != t_front) {
          history.emplace_front(x.size());
          system(x, history.front(), t);
          if (history.size() > 2 * Steps) history.pop_back();
          t_front = t;
        }
        if (dt != history_dt) resample(dt);

        if (history.size() < Steps) {
          boost::numeric::odeint::runge_kutta_cash_karp54<State> rk;
          rk.do_step(system, x, history.front(), t, dt, error);
          return;
        }

        // Predictor: x_p = x + dt sum_j b_j f_{n-j}
        predicted = x;
        for (size_t j = 0; j < Steps; ++j)
          predicted += (dt * coefficients.ab[j]) * history[j];
        f_predicted.resize(x.size());
        system(predicted, f_predicted, t + dt);
        // Corrector: x + dt (b*_0 f(x_p, t + dt) + sum_{j>0} b*_j f_{n+1-j})
        x += (dt * coefficients.am[0]) * f_predicted;
        for (size_t j = 1; j < Steps; ++j)
          x += (dt * coefficients.am[j]) * history[j - 1];
        error = milne_factor() * (x - predicted);
      }

      const State &error_estimate() const { return error; }

      static constexpr double milne_factor()
      {
        const auto g = ab_constants();
        const double corrector = g[Steps] - g[Steps - 1];
        return (corrector < 0. ? -corrector : corrector) / g[Steps - 1];
      }

    private:
      /// The Adams-Bashforth constants g_0 ... g_Steps of the backward-difference form.
      static constexpr std::array<double, Steps + 1> ab_constants()
      {
        std::array<double, Steps + 1> g{1.};
        for (size_t m = 1; m <= Steps; ++m) {
          g[m] = 1.;
          for (size_t j = 0; j < m; ++j)
            g[m] -= g[j] / double(m + 1 - j);
        }
        return g;
      }

      /// The weights of f_{n-j} (Adams-Bashforth) and f_{n+1-j} (Adams-Moulton), from the backward-difference forms
      /// sum_m g_m nabla^m f_n and sum_m g*_m nabla^m f_{n+1}, with nabla^m f_k = sum_j (-1)^j binom(m, j) f_{k-j}.
      struct Coefficients {
        std::array<double, Steps> ab{};
        std::array<double, Steps> am{};
      };
      static constexpr Coefficients make_coefficients()
      {
        const auto g = ab_constants();
        Coefficients c;
        for (size_t m = 0; m < Steps; ++m) {
          const double g_star = m == 0 ? 1. : g[m] - g[m - 1];
          double binomial = 1.; // binom(m, j)
          for (size_t j = 0; j <= m; ++j) {
            const double sign = j % 2 == 0 ? 1. : -1.;
            c.ab[j] += g[m] * sign * binomial;
            c.am[j] += g_star * sign * binomial;
            binomial = binomial * double(m - j) / double(j + 1);
          }
        }
        return c;
      }
      static constexpr Coefficients coefficients = make_coefficients();

      /**
       * Resample the history (points t_front - k history_dt) onto t_front - k dt. Each new point is interpolated by
       * the polynomial through the (at most Steps) stored points around it, which reproduces stored points exactly;
       * new points beyond the oldest stored one are dropped rather than extrapolated.
       */
      void resample(const double dt)
      {
        if (history.size() > 1) {
          const size_t n = history.size();
          const size_t width = std::min(n, Steps);
          const double span = double(n - 1) * history_dt;
          std::deque<State> resampled;
          for (size_t i = 0; i < 2 * Steps && double(i) * dt <= span * (1. + 1e-12); ++i) {
            const double s = double(i) * dt / history_dt; // position in units of the old spacing
            const long centre = std::lround(s) - long(width / 2);
            const size_t k0 = size_t(std::clamp(centre, 0l, long(n - width)));
            State value = State::Zero(history.front().size());
            for (size_t a = k0; a < k0 + width; ++a) {
              double weight = 1.;
              for (size_t b = k0; b < k0 + width; ++b)
                if (b != a) weight *= (s - double(b)) / (double(a) - double(b));
              value += weight * history[a];
            }
            resampled.push_back(std::move(value));
          }
          history = std::move(resampled);
        }
        history_dt = dt;
      }

      std::deque<State> history; // f at t_front - k history_dt, newest first
      double t_front = 0.;
      double history_dt = 0.;
      State predicted, f_predicted, error;
    };

    /**
     * @brief Local error control for an ABMWithErrorEstimate, driven by /timestepping/explicit/error_control.
     *
     * The local error estimate, scaled by abs_tol + rel_tol |x|, must stay below 1, else the step is retaken at half
     * the size -- down to minimal_dt. After 2 * Steps steps in a row with an error below 1/4 the step doubles back
     * towards base_dt. The estimate does not fall like dt^(Steps + 1) in practice -- right-hand sides read at an EoM or
     * from interpolated spatial data are not smooth on the scale of dt -- so the growth test cannot rely on that.
     *
     * With `aligned`, the step only grows where the time is a multiple of the grown step from t0, so all steps stay
     * on the grid t0 + n base_dt / 2^k and a run whose length is a multiple of base_dt ends exactly on its last step.
     */
    class ABMStepControl
    {
    public:
      struct Parameters {
        bool enabled;
        bool aligned;
        double t0;
        double base_dt;
        double minimal_dt;
        double abs_tol;
        double rel_tol;
      };

      explicit ABMStepControl(const Parameters &parameters) : prm(parameters) {}

      /**
       * @brief Take one accepted step of `stepper` from t, retrying with smaller steps if the error control asks for
       * it. Advances t and may change dt.
       *
       * @param on_estimate called as on_estimate(t + dt, dt, err) for every step attempt.
       */
      template <typename Stepper, typename System, typename OnEstimate>
      void step(Stepper &stepper, System &&system, Eigen::VectorXd &x, double &t, double &dt, OnEstimate &&on_estimate)
      {
        for (;;) {
          if (prm.enabled) backup = x;
          stepper.do_step(system, x, t, dt);
          const auto &error = stepper.error_estimate();
          double err = 0.;
          for (Eigen::Index i = 0; i < x.size(); ++i)
            err = std::max(err, std::abs(error[i]) / (prm.abs_tol + prm.rel_tol * std::abs(x[i])));
          on_estimate(t + dt, dt, err);
          if (prm.enabled && err > 1. && dt > prm.minimal_dt) {
            x = backup;
            dt = std::max(0.5 * dt, prm.minimal_dt);
            calm_steps = 0;
            ++n_rejected;
            continue;
          }
          t += dt;
          if (prm.enabled) {
            calm_steps = err < 0.25 ? calm_steps + 1 : 0;
            if (calm_steps >= 2 * Stepper::steps && dt < prm.base_dt && may_grow(t, dt)) {
              dt = std::min(2. * dt, prm.base_dt);
              calm_steps = 0;
            }
          }
          return;
        }
      }

      /// The number of steps retaken at a smaller size so far.
      size_t rejected() const { return n_rejected; }

    private:
      bool may_grow(const double t, const double dt) const
      {
        return !prm.aligned || std::llround((t - prm.t0) / dt) % 2 == 0;
      }

      Parameters prm;
      size_t calm_steps = 0;
      size_t n_rejected = 0;
      Eigen::VectorXd backup;
    };
  } // namespace internal
} // namespace DiFfRG
