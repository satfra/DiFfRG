#pragma once

// standard library
#include <array>
#include <cmath>
#include <limits>
#include <string>

#include <functional>
#include <spdlog/spdlog.h>
#include <stdexcept>

// external libraries
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_multimin.h>

/**
 * @file minimization.hh
 * @brief Minimization of a function \f$f: \mathbb{R}^{d} \to \mathbb{R}\f$ with GSL, without derivatives.
 *
 * @code
 * GSLMinimizer1D minimizer([](double x) { return (x - 0.3) * (x - 0.3); }, 1e-8);
 * minimizer.set_bounds(0., 1.);
 * minimizer.set_x0(0.5);
 * const double x_min = minimizer.minimize(); // 0.3
 * @endcode
 *
 * @note Constructing a GSL minimizer calls `gsl_set_error_handler_off()`, which switches off GSL's abort-on-error for
 * the whole process.
 */

namespace DiFfRG
{
  /**
   * @brief Common interface of the minimizers: set tolerances, call minimize(), get the location of the minimum.
   *
   * @tparam dim number of variables \f$d\f$. For \f$d = 1\f$ the objective takes a plain `double`.
   */
  template <int dim> class AbstractMinimizer
  {
  protected:
    using FUN = std::function<double(const std::array<double, dim> &)>;

  public:
    /**
     * @param f the objective \f$f(x)\f$, taking a `std::array<double, dim>`.
     * @param abs_tol absolute tolerance on the location of the minimum (its meaning depends on the method).
     * @param max_iter maximal number of iterations.
     */
    AbstractMinimizer(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(f), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    /// Set the absolute tolerance.
    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    /// Set the maximal number of iterations.
    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    /// Number of iterations of the last minimize().
    uint get_iter() const { return iter; }

    /// Run the minimization and return the location \f$x_\text{min}\f$ of the minimum.
    std::array<double, dim> minimize() { return this->minimize_impl(); }

  protected:
    FUN f;

    double abs_tol;
    uint max_iter;
    uint iter;

    virtual std::array<double, dim> minimize_impl() = 0;
  };

  /// One variable: the objective takes a `double`, minimize() returns a `double`.
  template <> class AbstractMinimizer<1>
  {
  protected:
    using FUN = std::function<double(const double)>;
    using FUN_ARR = std::function<double(const std::array<double, 1> &)>;

  public:
    /**
     * @param f the objective \f$f(x)\f$, taking a `double`.
     * @param abs_tol absolute tolerance on the location of the minimum.
     * @param max_iter maximal number of iterations.
     */
    AbstractMinimizer(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f([f](const std::array<double, 1> &x) { return f(x[0]); }), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    /// As above, with an objective taking a `std::array<double, 1>`.
    AbstractMinimizer(const FUN_ARR &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(f), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    /// Set the absolute tolerance.
    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    /// Set the maximal number of iterations.
    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    /// Number of iterations of the last minimize().
    uint get_iter() const { return iter; }

    /// Run the minimization and return the location \f$x_\text{min}\f$ of the minimum.
    double minimize() { return this->minimize_impl()[0]; }

  protected:
    FUN_ARR f;

    double abs_tol;
    uint max_iter;
    uint iter;

    virtual std::array<double, 1> minimize_impl() = 0;
  };

  /**
   * @brief Nelder-Mead simplex minimizer (GSL `nmsimplex2`).
   *
   * Starts from a simplex around set_x0() with edge length set_step_size() (default 1). Stops when the simplex size
   * drops below `abs_tol`, when \f$f\f$ reaches set_f_target(), or after `max_iter` iterations; in the latter cases
   * minimize() returns the best point found and converged() tells which. Throws only if GSL fails.
   *
   * @tparam dim number of variables. For `dim = 1` the objective takes a plain `double`.
   */
  template <int dim> class GSLSimplexMinimizer : public AbstractMinimizer<dim>
  {
    using FUN = AbstractMinimizer<dim>::FUN;

  public:
    /**
     * @param f the objective \f$f(x)\f$.
     * @param abs_tol stop when the simplex size falls below this.
     * @param max_iter maximal number of iterations.
     */
    GSLSimplexMinimizer(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : AbstractMinimizer<dim>(f, abs_tol, max_iter), step_size(1.)
    {
      gsl_set_error_handler_off();
    }

    /// Edge length of the initial simplex (default 1).
    void set_step_size(const double step_size) { this->step_size = step_size; }

    /// The starting point; must be set before minimize().
    void set_x0(const std::array<double, dim> &x0) { this->x0 = x0; }

    /**
     * @brief Stop as soon as \f$f \leq\f$ `f_target`.
     *
     * Useful when \f$f\f$ has a meaningful scale, e.g. a residual: then reaching the target is the success
     * criterion, and a collapsed simplex counts as not converged. NaN (the default) disables the target.
     */
    void set_f_target(const double f_target) { this->f_target = f_target; }

    /// Objective value at the point returned by minimize(). NaN before it ran.
    double get_min_value() const { return min_value; }

    /**
     * @brief Whether the search met its criterion: the f-target if one was set, otherwise the
     * simplex-size test. False when it stopped because the simplex collapsed short of the target
     * or the iteration cap was hit -- in both cases minimize() still returns the best point found.
     */
    bool converged() const { return m_converged; }

    /// Why the search stopped, for logging.
    const std::string &get_stop_reason() const { return stop_reason; }

  private:
    static double gsl_wrap(const gsl_vector *v, void *params)
    {
      GSLSimplexMinimizer<dim> *self = (GSLSimplexMinimizer<dim> *)params;
      std::array<double, dim> x;
      for (int i = 0; i < dim; ++i)
        x[i] = gsl_vector_get(v, i);
      return self->f(x);
    }

  protected:
    virtual std::array<double, dim> minimize_impl() override
    {
      gsl_multimin_function gsl_f;
      gsl_f.n = dim;
      gsl_f.f = &GSLSimplexMinimizer<dim>::gsl_wrap;
      gsl_f.params = this;

      gsl_vector *x = gsl_vector_alloc(dim);
      for (int i = 0; i < dim; ++i)
        gsl_vector_set(x, i, x0[i]);

      gsl_vector *init_step = gsl_vector_alloc(dim);
      gsl_vector_set_all(init_step, step_size);

      const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
      gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(T, dim);

      gsl_multimin_fminimizer_set(s, &gsl_f, x, init_step);

      const bool have_target = std::isfinite(f_target);
      int status = GSL_CONTINUE;
      bool iterate_failed = false;
      this->m_converged = false;
      this->stop_reason = "iteration limit reached";
      this->iter = 0;
      do {
        this->iter++;
        if (gsl_multimin_fminimizer_iterate(s)) {
          iterate_failed = true;
          this->stop_reason = "GSL simplex iteration failed";
          break;
        }

        if (have_target && s->fval <= f_target) {
          this->m_converged = true;
          this->stop_reason = "objective reached the target";
          break;
        }

        const double size = gsl_multimin_fminimizer_size(s);
        status = gsl_multimin_test_size(size, this->abs_tol);
        if (status == GSL_SUCCESS) {
          // Without a target this IS the criterion; with one, the simplex ran out of room before
          // the objective got there, which is a real answer about the model, not a failure.
          this->m_converged = !have_target;
          this->stop_reason = have_target ? "simplex collapsed short of the target" : "simplex size below tolerance";
          break;
        }
      } while (this->iter < this->max_iter);

      this->min_value = s->fval;

      std::array<double, dim> result;
      for (int i = 0; i < dim; ++i)
        result[i] = gsl_vector_get(s->x, i);

      gsl_multimin_fminimizer_free(s);
      gsl_vector_free(init_step);
      gsl_vector_free(x);

      // Only a genuine solver breakdown is fatal. Stopping short of the target still yields the
      // best point seen, and the caller can ask converged()/get_stop_reason() about it.
      if (iterate_failed) throw std::runtime_error("Minimization failed: " + this->stop_reason);
      if (!this->m_converged)
        spdlog::warn("GSLSimplexMinimizer: {} after {} iterations; returning the best point with f = {:.6e}",
                     this->stop_reason, this->iter, this->min_value);

      return result;
    }

    std::array<double, dim> x0;
    double step_size;
    double f_target = std::numeric_limits<double>::quiet_NaN();
    double min_value = std::numeric_limits<double>::quiet_NaN();
    bool m_converged = false;
    std::string stop_reason;
  };

  /**
   * @brief Minimizer of a function of one variable on an interval \f$[a, b]\f$ (GSL `gsl_min_fminimizer`).
   *
   * Needs a starting point \f$x_0\f$ with \f$f(x_0) < f(a)\f$ and \f$f(x_0) < f(b)\f$, so that a minimum is bracketed;
   * otherwise minimize() throws. An \f$x_0\f$ outside \f$[a, b]\f$ is replaced by the midpoint. Stops when the
   * bracket \f$[x_\text{lo}, x_\text{hi}]\f$ is narrower than
   * \f$\mathrm{abs\_tol} + \mathrm{rel\_tol} \cdot \min(|x_\text{lo}|, |x_\text{hi}|)\f$; throws if that is not reached
   * within `max_iter` iterations.
   */
  class GSLMinimizer1D : public AbstractMinimizer<1>
  {
    using FUN = AbstractMinimizer<1>::FUN;

  public:
    /**
     * @param f the objective \f$f(x)\f$, taking a `double`.
     * @param abs_tol absolute tolerance on the location of the minimum.
     * @param max_iter maximal number of iterations.
     */
    GSLMinimizer1D(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : AbstractMinimizer<1>(f, abs_tol, max_iter), x0(0.), min_x(-1.), max_x(1.), rel_tol(0.), m(method::brent)
    {
      gsl_set_error_handler_off();
    }

    /**
     * @brief The method: golden-section search, Brent's method (default), or Brent's method with safeguarded
     * quadratic steps (GSL `quad_golden`).
     */
    enum method { golden_section, brent, quadratic };

    /// The starting point \f$x_0\f$ (default 0).
    void set_x0(const double x0) { this->x0 = x0; }

    /// The interval \f$[a, b]\f$ (default \f$[-1, 1]\f$).
    void set_bounds(const double min_x, const double max_x)
    {
      this->min_x = min_x;
      this->max_x = max_x;
    }

    /// Choose the method; default Brent.
    void set_method(const method m) { this->m = m; }

    /// Relative tolerance on the location (default 0).
    void set_rel_tol(const double rel_tol) { this->rel_tol = rel_tol; }

  protected:
    virtual std::array<double, 1> minimize_impl() override
    {
      gsl_function gsl_f;
      gsl_f.function = &GSLMinimizer1D::gsl_wrap;
      gsl_f.params = this;

      const gsl_min_fminimizer_type *T;
      switch (this->m) {
      case golden_section:
        T = gsl_min_fminimizer_goldensection;
        break;
      case brent:
        T = gsl_min_fminimizer_brent;
        break;
      case quadratic:
        T = gsl_min_fminimizer_quad_golden;
        break;
      default:
        throw std::runtime_error("Unknown minimization method.");
      }

      gsl_min_fminimizer *s = gsl_min_fminimizer_alloc(T);

      double x = this->x0 < this->min_x || this->x0 > this->max_x ? 0.5 * (this->min_x + this->max_x) : this->x0;
      double x_lo = this->min_x;
      double x_hi = this->max_x;

      if (gsl_min_fminimizer_set(s, &gsl_f, x, x_lo, x_hi) != GSL_SUCCESS) {
        gsl_min_fminimizer_free(s);
        throw std::runtime_error("Minimization failed: f(x0) must be smaller than f at both bounds.");
      }

      int status;
      this->iter = 0;
      do {
        this->iter++;
        status = gsl_min_fminimizer_iterate(s);
        if (status != GSL_SUCCESS) break;

        x = gsl_min_fminimizer_x_minimum(s);
        x_lo = gsl_min_fminimizer_x_lower(s);
        x_hi = gsl_min_fminimizer_x_upper(s);

        status = gsl_min_test_interval(x_lo, x_hi, this->abs_tol, this->rel_tol);
      } while (status == GSL_CONTINUE && this->iter < this->max_iter);

      gsl_min_fminimizer_free(s);
      if (status != GSL_SUCCESS) throw std::runtime_error("Minimization failed.");

      return {x};
    }

    static double gsl_wrap(double x, void *params)
    {
      GSLMinimizer1D *self = (GSLMinimizer1D *)params;
      return self->f({{x}});
    }

    double x0;
    double min_x, max_x;
    double rel_tol;
    method m;
  };
} // namespace DiFfRG