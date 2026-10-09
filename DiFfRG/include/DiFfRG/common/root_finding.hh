#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

namespace DiFfRG
{
  template <int dim> class AbstractRootFinder
  {
  protected:
    using FUN = std::function<bool(const std::array<double, dim> &)>;

  public:
    AbstractRootFinder(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(f), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    uint get_iter() const { return iter; }

    std::array<double, dim> search() { return this->search_impl(); }

  protected:
    FUN f;

    double abs_tol;
    uint max_iter;
    uint iter;

    virtual std::array<double, dim> search_impl() = 0;
  };

  // explicit specialization for 1D search
  template <> class AbstractRootFinder<1>
  {
  protected:
    using FUN = std::function<bool(const double)>;
    using FUN_ARR = std::function<bool(const std::array<double, 1> &)>;

  public:
    AbstractRootFinder(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f([f](const std::array<double, 1> &x) { return f(x[0]); }), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    AbstractRootFinder(const FUN_ARR &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(f), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    uint get_iter() const { return iter; }

    double search() { return this->search_impl()[0]; }

  protected:
    FUN_ARR f;

    double abs_tol;
    uint max_iter;
    uint iter;

    virtual std::array<double, 1> search_impl() = 0;
  };

  class BisectionRootFinder : public AbstractRootFinder<1>
  {
  public:
    BisectionRootFinder(const FUN &f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : AbstractRootFinder<1>(f, abs_tol, max_iter)
    {
    }

    void set_x_min(const double x_min) { this->x_min = x_min; }
    void set_x_max(const double x_max) { this->x_max = x_max; }

    void set_bounds(const double x_min, const double x_max)
    {
      this->x_min = x_min;
      this->x_max = x_max;
    }

    void set_next_x(const std::function<double(double, double)> &next_x) { this->next_x = next_x; }

  protected:
    std::array<double, 1> search_impl() override
    {
      double x_min = this->x_min;
      double x_max = this->x_max;
      double x_test = 0.;

      bool success = false;

      for (uint i = 0; i < this->max_iter; i++) {
        x_test = next_x(x_min, x_max);
        success = this->f({{x_test}});
        if (success)
          x_max = x_test;
        else
          x_min = x_test;

        this->iter = i;
        if (success && std::abs(x_max - x_min) < this->abs_tol) break;
      }

      if (!success) throw std::runtime_error("BisectionRootFinder: search did not converge");

      return {{x_test}};
    }

    double x_min;
    double x_max;

    std::function<double(double, double)> next_x = [](const double x_min, const double x_max) {
      return (x_min + x_max) / 2.;
    };
  };

  /**
   * @brief Bisection search which converges a target value rather than the search variable.
   *
   * The bracketing logic is identical to BisectionRootFinder: the callback returns `true` if the
   * tested point lies at or above the root (tightening the upper bound) and `false` if it lies
   * below (tightening the lower bound). The difference is the stopping criterion. Instead of
   * shrinking the interval in x until it falls below `abs_tol`, the callback additionally writes a
   * target value through its `double &` out-parameter, and the search stops once two consecutive
   * *successful* evaluations produce target values differing by less than `abs_tol`.
   *
   * This is the right tool when the quantity of interest is not x itself but an observable
   * computed at x, whose sensitivity to x is not known a priori.
   *
   * @code
   * BisectionRootFinderTarget search(
   *     [&](const double x, double &target) -> bool {
   *       config.set_double("/physical/m2A", x);
   *       return run_at(x, target); // writes the observable into target
   *     },
   *     tol, 100);
   * search.set_bounds(lower, upper);
   * const double x      = search.search();
   * const double target = search.get_target();
   * @endcode
   *
   * Failed evaluations need not write anything meaningful into the target; only successful ones are
   * compared. If the bracket collapses to machine precision, or `max_iter` is exhausted, before the
   * criterion is met, the search stops, warns, and returns the last successful point with
   * `converged()` reporting `false`. Only a search without any successful evaluation throws.
   *
   * Note that the criterion fires early if the target happens to be insensitive to x over part of
   * the bracket, since two neighbouring successes then look converged. In that case tighten
   * `abs_tol` or install a `next_x` which does not bisect.
   *
   * Two deliberate differences to BisectionRootFinder: the iteration counter is advanced *before*
   * the callback runs, so `get_iter()` inside the callback names the step being taken, and the
   * bounds are NaN-initialised so that a forgotten `set_bounds` throws instead of reading garbage.
   */
  class BisectionRootFinderTarget
  {
  protected:
    using FUN = std::function<bool(const double, double &)>;

  public:
    // the callback is taken by value and moved: callers pass a temporary lambda, and the copy from
    // a const reference also trips a spurious -Wmaybe-uninitialized inside std::function on GCC 16
    BisectionRootFinderTarget(FUN f, const double abs_tol = 1e-4, const int max_iter = 1000)
        : f(std::move(f)), abs_tol(abs_tol), max_iter(max_iter), iter(0)
    {
    }

    void set_abs_tol(const double abs_tol) { this->abs_tol = abs_tol; }

    void set_max_iter(const uint max_iter) { this->max_iter = max_iter; }

    void set_x_min(const double x_min) { this->x_min = x_min; }
    void set_x_max(const double x_max) { this->x_max = x_max; }

    void set_bounds(const double x_min, const double x_max)
    {
      this->x_min = x_min;
      this->x_max = x_max;
    }

    void set_next_x(const std::function<double(double, double)> &next_x) { this->next_x = next_x; }

    /**
     * @brief Width of the x-bracket below which the search gives up. Zero (the default) selects an
     * automatic, magnitude-scaled multiple of the machine epsilon.
     */
    void set_x_collapse_tol(const double x_collapse_tol) { this->x_collapse_tol = x_collapse_tol; }

    uint get_iter() const { return iter; }

    /**
     * @brief Target value belonging to the point returned by search(). NaN before search() ran.
     */
    double get_target() const { return target; }

    /**
     * @brief Whether the last search() met the target tolerance. False if it stopped early because
     * the bracket collapsed or the iteration limit was reached.
     */
    bool converged() const { return m_converged; }

    double search()
    {
      if (!(std::isfinite(this->x_min) && std::isfinite(this->x_max)))
        throw std::runtime_error("BisectionRootFinderTarget: search bounds were not set");

      double x_lo = this->x_min;
      double x_hi = this->x_max;

      double best_x = std::numeric_limits<double>::quiet_NaN();
      double best_target = std::numeric_limits<double>::quiet_NaN();
      bool have_success = false;

      this->m_converged = false;
      this->target = std::numeric_limits<double>::quiet_NaN();

      for (uint i = 0; i < this->max_iter; ++i) {
        this->iter = i;

        const double x_test = next_x(x_lo, x_hi);
        double x_target = std::numeric_limits<double>::quiet_NaN();
        const bool success = this->f(x_test, x_target);

        if (success) {
          x_hi = x_test;
          // Two consecutive successes whose targets agree to abs_tol: the observable is resolved.
          const bool target_converged = have_success && std::abs(x_target - best_target) < this->abs_tol;
          best_x = x_test;
          best_target = x_target;
          have_success = true;
          if (target_converged) {
            this->m_converged = true;
            break;
          }
        } else
          x_lo = x_test;

        const double collapse_tol =
            this->x_collapse_tol > 0.
                ? this->x_collapse_tol
                : 4. * std::numeric_limits<double>::epsilon() * std::max(1., std::max(std::abs(x_lo), std::abs(x_hi)));
        if (std::abs(x_hi - x_lo) <= collapse_tol) break;
      }

      if (!have_success)
        throw std::runtime_error("BisectionRootFinderTarget: no evaluation of the search interval succeeded");

      this->target = best_target;

      if (!this->m_converged)
        spdlog::warn("BisectionRootFinderTarget: stopped after {} iterations without reaching the target tolerance "
                     "{:.6e}; returning x = {:.12e} with target = {:.12e}",
                     this->iter + 1, this->abs_tol, best_x, best_target);

      return best_x;
    }

  protected:
    FUN f;

    double abs_tol;
    uint max_iter;
    uint iter;

    double x_min = std::numeric_limits<double>::quiet_NaN();
    double x_max = std::numeric_limits<double>::quiet_NaN();
    double x_collapse_tol = 0.;

    double target = std::numeric_limits<double>::quiet_NaN();
    bool m_converged = false;

    std::function<double(double, double)> next_x = [](const double x_min, const double x_max) {
      return (x_min + x_max) / 2.;
    };
  };

  namespace detail
  {
    /**
     * @brief Result of a three-point power-law fit @f$ y = C (x - x_c)^\beta @f$.
     *
     * @c shift is @f$ s = x_3 - x_c @f$ measured from the point closest to @f$ x_c @f$, rather
     * than @f$ x_c @f$ itself: @f$ s @f$ is the quantity the solve actually determines, it is
     * positive by construction, and it stays well-conditioned when @f$ |x_c| \gg s @f$ -- which
     * is the whole point of the fit.
     */
    struct ScalingFit {
      bool valid = false;
      double shift = std::numeric_limits<double>::quiet_NaN();         ///< s = x_3 - x_c > 0
      double exponent = std::numeric_limits<double>::quiet_NaN();      ///< beta
      double log_amplitude = std::numeric_limits<double>::quiet_NaN(); ///< ln C
    };

    /**
     * @brief Solve the three-point power-law condition for the shift @f$ s @f$.
     *
     * Three points on @f$ y = C (x - x_c)^\beta @f$, ordered @f$ x_1 > x_2 > x_3 > x_c @f$ and
     * hence @f$ y_1 > y_2 > y_3 > 0 @f$, determine @f$ x_c @f$. Writing
     * @f$ s = x_3 - x_c @f$, @f$ d_1 = x_1 - x_3 @f$, @f$ d_2 = x_2 - x_3 @f$ and eliminating
     * @f$ \beta @f$ between the three log-equations leaves
     * @f[
     *   F(s) = A \ln\!\big(1 + d_2/s\big) - B \ln\!\frac{s + d_1}{s + d_2} = 0,
     *   \qquad A = \ln(y_1/y_2),\; B = \ln(y_2/y_3).
     * @f]
     *
     * @f$ F(s) \to +\infty @f$ as @f$ s \to 0^+ @f$ and @f$ F(s) \sim [A d_2 - B(d_1 - d_2)]/s @f$
     * as @f$ s \to \infty @f$, so a positive root exists **iff**
     * @f$ A d_2 < B (d_1 - d_2) @f$. That condition is checked analytically rather than by
     * probing some arbitrary large @f$ s @f$, which overflows for widely separated triples. It is
     * also the honest test of the model: data that is log-linear in @f$ x @f$, or convex the
     * wrong way, refutes the power law and must be rejected rather than fitted.
     *
     * The bisection runs in @f$ \ln s @f$ because @f$ s @f$ legitimately spans many decades
     * between the far-from-critical and converged regimes, and @c log1p carries the first term,
     * which the naive @f$ \ln((s+d_2)/s) @f$ loses entirely once @f$ s \gg d_2 @f$.
     *
     * @param A ln(y1/y2), must be > 0. Callers on the divergent branch pass a difference of
     *          residuals directly, so no exponentials are ever formed.
     * @param B ln(y2/y3), must be > 0.
     * @param d1 x1 - x3 > d2
     * @param d2 x2 - x3 > 0
     * @return s > 0, or NaN if the data does not admit a positive root.
     */
    inline double solve_scaling_shift(const double A, const double B, const double d1, const double d2)
    {
      if (!(std::isfinite(A) && std::isfinite(B) && std::isfinite(d1) && std::isfinite(d2))) return NAN;
      if (!(A > 0. && B > 0. && d2 > 0. && d1 > d2)) return NAN;

      // A and B are log-ratios of the sampled values. Below this floor the three values are equal
      // to within rounding, both logs are noise, and every guard downstream is being applied to
      // garbage. In any regime the finder actually uses, A and B are O(0.1..1).
      if (!(A > 1e-10 && B > 1e-10)) return NAN;

      // Asymptotic sign test: without it there is no root and the bracketing below cannot
      // converge. The margin also measures the conditioning. For a power law sampled at spacing
      // d far from x_c, both sides agree at leading order and the gap is only O(d/s) relative --
      // so a vanishing margin means the data cannot distinguish the power law from a straight
      // line, and any root found is cancellation noise. Without this, a triple sampled ~1e12
      // spacings from criticality yields a confident, wholly spurious shift.
      const double lhs = A * d2, rhs = B * (d1 - d2);
      if (!(lhs < rhs)) return NAN;
      if (!(rhs - lhs > 1e-9 * std::max(std::abs(lhs), std::abs(rhs)))) return NAN;

      const auto F = [&](const double s) { return A * std::log1p(d2 / s) - B * std::log((s + d1) / (s + d2)); };

      // F is +inf at 0+ and negative far out; walk outwards from the natural scale d2.
      double s_hi = d2;
      for (int i = 0; i < 60 && F(s_hi) > 0.; ++i)
        s_hi *= 16.;
      if (F(s_hi) > 0.) return NAN;

      // Floor: below this the bisection cannot make progress. It has to be relative to the point
      // spacing, not absolute -- a probe that lands very close to criticality legitimately gives
      // s / d2 ~ 1e-14, and an absolute floor rejects exactly the most informative triple there is.
      const double s_floor = 1e-18 * d2;
      double s_lo = s_hi / 16.;
      for (int i = 0; i < 60 && s_lo > s_floor && F(s_lo) < 0.; ++i)
        s_lo /= 16.;
      if (F(s_lo) < 0.) return NAN;

      for (int i = 0; i < 200; ++i) {
        const double s_mid = std::sqrt(s_lo * s_hi);
        if (!(s_mid > s_lo && s_mid < s_hi)) break; // converged to the last representable interval
        if (F(s_mid) > 0.)
          s_lo = s_mid;
        else
          s_hi = s_mid;
      }
      const double s = std::sqrt(s_lo * s_hi);
      if (!(std::isfinite(s) && s > s_floor)) return NAN;

      // Precision ceiling. Once s >> d2 the three points sit far from x_c, the power law is
      // locally indistinguishable from a straight line, and A*d2 - B*(d1-d2) is a difference of
      // nearly equal doubles. Measured against exact data the result is good to 1e-8 relative up
      // to s ~ 1e4 d2 and degrades to tens of percent beyond it -- while still looking perfectly
      // finite. Reject rather than hand back a confident wrong critical point.
      //
      // The comparison carries a guard band because the quantity being tested is the bisection's
      // own output, and at the ceiling that output already carries O(1e-8) relative error. A bare
      // `s > 1e4 * d2` therefore decides data sitting exactly on the threshold by which side
      // rounding happens to land on -- for the canonical d1=3, d2=1 triple the root at s = 1e4
      // comes out as 1e4*(1 +/- 1.3e-8), so the verdict flips with optimisation flags rather than
      // with the data. Widening the rejection by 1e-6 relative, ~100x the noise, makes the
      // documented ceiling a hard boundary. Everything the accurate branch relies on sits at
      // s <= 1e3 * d2, a full decade below, so nothing that was accepted becomes rejected.
      if (!(s < 1e4 * d2 * (1. - 1e-6))) return NAN;

      return s;
    }

    /**
     * @brief Three-point power-law fit. Points need not be pre-sorted.
     *
     * Rejects non-monotone or tied data outright: a triple that is not strictly decreasing in
     * both x and y is not a sample of a monotone power law, and fitting it produces a confident
     * wrong answer rather than a visible failure.
     */
    inline ScalingFit fit_power_law_3(std::array<double, 3> x, std::array<double, 3> y)
    {
      std::array<std::size_t, 3> idx{{0, 1, 2}};
      std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return x[a] > x[b]; });
      const double x1 = x[idx[0]], x2 = x[idx[1]], x3 = x[idx[2]];
      const double y1 = y[idx[0]], y2 = y[idx[1]], y3 = y[idx[2]];

      ScalingFit fit;
      if (!(std::isfinite(y1) && std::isfinite(y2) && std::isfinite(y3))) return fit;
      if (!(x1 > x2 && x2 > x3)) return fit;
      if (!(y1 > y2 && y2 > y3 && y3 > 0.)) return fit;

      const double A = std::log(y1 / y2), B = std::log(y2 / y3);
      const double d1 = x1 - x3, d2 = x2 - x3;
      const double s = solve_scaling_shift(A, B, d1, d2);
      if (!std::isfinite(s)) return fit;

      const double beta = A / std::log((s + d1) / (s + d2));
      if (!std::isfinite(beta) || beta <= 0.) return fit;

      fit.valid = true;
      fit.shift = s;
      fit.exponent = beta;
      fit.log_amplitude = std::log(y3) - beta * std::log(s);
      return fit;
    }
  } // namespace detail

  /**
   * @brief Bracketed root find accelerated by the critical scaling of the observable.
   *
   * Solves obs(x) == target where obs is smooth and monotone on the convergent side of a
   * critical point x_c < x, obeying obs = C (x - x_c)^beta, and where probes below x_c fail but
   * expose a residual that grows as x_c is approached, r(x) = a - ln(x_c - x)/theta. Neither
   * exponent is assumed; both are fitted from three points, and the same solve serves both
   * branches (the divergent one via x -> -x, y -> exp(-r), which is why only the residual
   * *differences* ever enter and no exponentials are formed).
   *
   * Every model-based proposal is bracket-safeguarded Brent-style, so a wrong model costs at
   * most a factor two over plain bisection and can never move the answer.
   *
   * The callback reports three outcomes, not two:
   *   - returns true and writes a finite @c obs: a usable evaluation. Whether it lies above or
   *     below the target is the finder's business, not the callback's.
   *   - returns false and writes a finite @c residual: the evaluation ran away; the residual
   *     says how far it got.
   *   - returns false and writes nothing: it failed some other way and carries no information
   *     beyond its own x. Both @c obs and @c residual are NaN-initialised before every call, so
   *     "wrote nothing" needs no separate signal.
   *
   * @code
   * ScalingRootFinder search(
   *     [&](const double x, double &obs, double &residual) -> bool {
   *       return run_at(x, obs, residual);
   *     },
   *     zc_target, 1e-3, 40);
   * search.set_bounds(lower, upper);          // hypotheses; expanded if they do not straddle
   * search.set_residual_window(0.4 * final_time);
   * const double x = search.search();
   * @endcode
   */
  class ScalingRootFinder
  {
  public:
    /**
     * @brief What counts as an answer.
     *
     * AtTarget converges obs onto the target to within rel_tol. BelowTarget accepts the first
     * evaluation that lands anywhere strictly between the critical point and the target
     * crossing, which is a far easier thing to hit: on a cold Yang-Mills tune that interval is
     * ~600x wider in the search variable than the AtTarget window, i.e. about nine bisections
     * cheaper, and it lets a merely decent fit succeed on its first attempt instead of being
     * refined until it is accurate to rel_tol.
     */
    enum class Acceptance { AtTarget, BelowTarget };

  protected:
    using FUN = std::function<bool(const double, double &, double &)>;
    /// Compatibility shape matching BisectionRootFinderTarget. Disables the divergent-branch model.
    using FUN_NO_RESIDUAL = std::function<bool(const double, double &)>;

  public:
    ScalingRootFinder(FUN f, const double target, const double rel_tol = 1e-3, const uint max_iter = 40)
        : f(std::move(f)), target(target), rel_tol(rel_tol), max_iter(max_iter)
    {
    }

    ScalingRootFinder(FUN_NO_RESIDUAL f, const double target, const double rel_tol = 1e-3, const uint max_iter = 40)
        : f([f = std::move(f)](const double x, double &obs, double &) { return f(x, obs); }), target(target),
          rel_tol(rel_tol), max_iter(max_iter)
    {
    }

    /** @name Bracket
     * The bounds are hypotheses by default: if they turn out not to straddle, the search widens
     * them geometrically instead of throwing, which is what makes one set of bounds usable
     * across models whose critical points differ by orders of magnitude.
     * @{ */
    void set_bounds(const double x_lo, const double x_hi)
    {
      this->x_lo_init = x_lo;
      this->x_hi_init = x_hi;
    }
    void set_bounds_are_hypotheses(const bool v) { bounds_are_hypotheses = v; }
    void set_expansion_factor(const double rho) { expansion_factor = rho; }
    void set_expansion_max_iter(const uint n) { expansion_max_iter = n; }
    /** @} */

    /** @name Stopping
     * @{ */
    void set_rel_tol(const double v) { rel_tol = v; }
    /// Accept only probes at or above the target. Keeps the answer on the safe side of the
    /// critical point at a cost of at most one extra evaluation. AtTarget mode only.
    void set_one_sided(const bool v) { one_sided = v; }
    void set_acceptance(const Acceptance a) { acceptance = a; }
    /// Where inside (0, target) to aim in BelowTarget mode, as a fraction of the target. Aiming
    /// at the midpoint rather than just under the crossing leaves room for the fit to be wrong
    /// in either direction without either overshooting into the divergent region or missing.
    void set_aim_fraction(const double v) { aim_fraction = v; }
    /// Bracket width, relative to the magnitude of the bounds, below which the search gives up.
    void set_x_rel_floor(const double v) { x_rel_floor = v; }
    void set_max_iter(const uint v) { max_iter = v; }
    /** @} */

    /** @name Model trust windows
     * The power law is asymptotic. Fitting points from outside its window does not degrade
     * gracefully -- it returns a confident, badly wrong critical point that sits inside the
     * bracket, where the containment safeguard cannot catch it. These windows are load-bearing.
     * @{ */
    /// Fit only points with obs <= window * target. Ignored when target == 0.
    void set_obs_window(const double v) { obs_window = v; }
    /// Fit only failures whose residual is at least this large. Defaults to 0 (fit any three),
    /// because only the caller knows the residual's scale.
    void set_residual_window(const double v) { residual_window = v; }
    void set_exponent_bounds(const double lo, const double hi)
    {
      beta_min = lo;
      beta_max = hi;
    }
    /// Bounds on the divergent-branch exponent. theta is an inverse unstable RG eigenvalue and
    /// sits near the tuned parameter's mass dimension -- 2 for a mass squared, measured 1.92 on
    /// a cold Yang-Mills tune -- so it is far more tightly constrained a priori than beta.
    void set_theta_bounds(const double lo, const double hi)
    {
      theta_min = lo;
      theta_max = hi;
    }
    /// Fraction of the bracket width kept clear of either endpoint when clamping a proposal.
    void set_endpoint_standoff(const double v) { endpoint_standoff = v; }
    /// Fraction of the remaining distance to the critical point kept per approach step.
    void set_approach_factor(const double v) { approach_factor = v; }
    /// Seed the exponent from a nearby earlier search (another simplex step, a coarser
    /// fidelity). Exponents are far more portable than amplitudes; do not seed the latter.
    void seed_exponent(const double beta) { seeded_exponent = beta; }
    /** @} */

    /** @name Results
     * @{ */
    double get_obs() const { return best_obs; }
    /// Best available critical point: the convergent fit when it is trusted, otherwise the
    /// divergent one, which is usable much further from criticality.
    double get_x_critical() const { return model_valid ? model_x_c : (divergent_valid ? divergent_x_c : NAN); }
    double get_exponent() const { return model_beta; }
    double get_theta() const { return model_theta; }
    bool have_model() const { return model_valid; }
    /// Model extrapolation of the target point. NaN when no model was ever trusted.
    double get_x_extrapolated() const { return model_valid ? extrapolate(model_x_c) : NAN; }
    bool converged() const { return m_converged; }
    uint get_iter() const { return iter; }
    /// Live bracket width; infinite until both sides are known.
    double bracket_width() const { return x_hi - x_lo_soft; }

    const std::vector<std::pair<double, double>> &successes() const { return S; }
    const std::vector<std::pair<double, double>> &failures() const { return F; }

    struct Counts {
      uint total = 0, expand = 0, model_obs = 0, model_mixed = 0, approach = 0, model_residual = 0, secant = 0,
           bisect = 0, forced_bisect = 0;
    };
    Counts get_counts() const { return counts; }
    /** @} */

    double search();

  protected:
    /// The value actually aimed at. With one-sided acceptance the admissible window is
    /// [target, target(1+rel_tol)]; aiming at its centre rather than its lower edge keeps
    /// rounding in the extrapolation from landing just below and costing an extra evaluation.
    double effective_target() const
    {
      if (acceptance == Acceptance::BelowTarget) return aim_fraction * target;
      return one_sided ? target * (1. + 0.5 * rel_tol) : target;
    }

    /// x at which the model predicts obs == effective_target(), given a critical point.
    double extrapolate(const double x_c) const
    {
      if (!(target > 0.)) return NAN;
      return x_c + std::exp((std::log(effective_target()) - model_log_C) / model_beta);
    }

    /**
     * @brief The three candidates closest to criticality, out of n sorted by distance to it.
     *
     * Spreading the triple across the whole admissible range is better conditioned in principle
     * -- a bisecting search leaves the closest points tightly clustered -- but measures worse:
     * the far points are not asymptotic enough, the fit is rejected outright more often, and a
     * cold SP tune stopped converging inside its budget. Both laws are asymptotic, so the fit
     * stays in the asymptote.
     */
    static std::array<std::size_t, 3> select_triple(const std::size_t) { return {{0, 1, 2}}; }

    /// Refit both branches. Called exactly once per evaluation, so that the consecutive-fit
    /// agreement test sees successive iterations rather than successive calls.
    void refit();
    /// Refit from the three convergent points closest to criticality (smallest obs).
    void fit_convergent();
    /// Refit from the three divergent points closest to criticality (largest residual).
    void fit_divergent();
    /// Propose the next x.
    double propose();
    /// Record a probe outcome and update the bracket.
    void record(double x, bool ok, double obs, double residual);

    FUN f;
    double target;
    double rel_tol;
    uint max_iter;

    double x_lo_init = NAN, x_hi_init = NAN;
    bool bounds_are_hypotheses = true;
    double expansion_factor = 2.;
    uint expansion_max_iter = 24;

    bool one_sided = true;
    Acceptance acceptance = Acceptance::AtTarget;
    double aim_fraction = 0.5;
    bool answer_found = false;
    double x_rel_floor = 1e-13;

    double obs_window = 20.;
    double residual_window = 0.;
    double beta_min = 0.02, beta_max = 5.;
    double theta_min = 0.5, theta_max = 8.;
    double endpoint_standoff = 0.02;
    double approach_factor = 0.1;
    double seeded_exponent = NAN;

    // Bracket. x_lo_soft is the bisection end: the largest x known to be invalid OR below the
    // target. x_lo_hard is the largest x known to be *invalid*, which is the hard constraint on
    // x_c -- a probe that merely came in under target is still a perfectly good sample of the
    // convergent branch and must not be mistaken for evidence about where x_c lies.
    double x_lo_soft = -std::numeric_limits<double>::infinity();
    double x_lo_hard = -std::numeric_limits<double>::infinity();
    double x_hi = std::numeric_limits<double>::infinity();

    std::vector<std::pair<double, double>> S; ///< (x, obs) for every finite evaluation
    std::vector<std::pair<double, double>> F; ///< (x, residual) for every informative failure

    // Convergent-branch model: obs = exp(model_log_C) * (x - model_x_c)^model_beta.
    bool model_valid = false;
    double model_x_c = NAN, model_beta = NAN, model_log_C = NAN, model_theta = NAN;
    double model_x_c_prev = NAN;
    // Divergent-branch model: residual = a - ln(divergent_x_c - x) / model_theta. Only its
    // critical point is used; the intercept a is not identifiable and not needed.
    bool divergent_valid = false;
    double divergent_x_c = NAN, divergent_shift = NAN;

    double best_x = NAN, best_obs = NAN;
    bool have_success = false;
    bool m_converged = false;
    uint iter = 0;
    Counts counts;

    double w_prev = std::numeric_limits<double>::infinity();
    double w_prevprev = std::numeric_limits<double>::infinity();

    bool probed_hi_init = false, probed_lo_init = false;
    uint expansions = 0;
    double w_init = NAN;
  };

  inline void ScalingRootFinder::record(const double x, const bool ok, const double obs, const double residual)
  {
    counts.total++;
    (void)ok; // the outcome is decided by which output the callback wrote, not by its return

    // Branch on the observable, NOT on the return value. A probe that converges to an observable
    // below the target returns false -- it is not an answer -- but it is still a sample of the
    // convergent branch, and filing it as a divergence both discards that sample and feeds a
    // bogus residual into the divergent fit.
    if (std::isfinite(obs)) {
      S.emplace_back(x, obs);
      std::sort(S.begin(), S.end());

      if (obs >= target) {
        if (x < x_hi) x_hi = x;
        // With one_sided the answer is the admissible probe closest to the target from above,
        // which is exactly the tightest upper bracket end. In BelowTarget mode nothing at or
        // above the target is admissible at all, so the best-so-far here is only a fallback for
        // a search that runs out of budget.
        const bool by_magnitude = one_sided || acceptance == Acceptance::BelowTarget;
        const bool better =
            !have_success || (by_magnitude ? obs < best_obs : std::abs(obs - target) < std::abs(best_obs - target));
        if (better) {
          best_x = x;
          best_obs = obs;
        }
        have_success = true;
      } else if (acceptance == Acceptance::BelowTarget && target > 0.) {
        // Strictly between the critical point and the target crossing: this IS the answer.
        best_x = x;
        best_obs = obs;
        have_success = true;
        answer_found = true;
      } else {
        // Below target, but still a convergent flow: a bracket update, NOT evidence about x_c.
        if (x > x_lo_soft) x_lo_soft = x;
        if (!one_sided && (!have_success || std::abs(obs - target) < std::abs(best_obs - target))) {
          best_x = x;
          best_obs = obs;
          have_success = true;
        }
      }
    } else {
      if (x > x_lo_soft) x_lo_soft = x;
      if (x > x_lo_hard) x_lo_hard = x;
      // A failure with no residual (a different failure mode entirely) still bounds the bracket,
      // but must never enter the divergent fit -- one NaN in a triple poisons it silently.
      if (std::isfinite(residual)) {
        F.emplace_back(x, residual);
        std::sort(F.begin(), F.end());
      }
    }

    const double w = x_hi - x_lo_soft;
    w_prevprev = w_prev;
    w_prev = w;
  }

  inline void ScalingRootFinder::refit()
  {
    fit_convergent();
    fit_divergent();
  }

  inline void ScalingRootFinder::fit_convergent()
  {
    model_valid = false;

    // The power law is asymptotic: fit the three points closest to criticality, and only those
    // inside the trust window.
    std::vector<std::pair<double, double>> pts; // (obs, x)
    for (const auto &[x, obs] : S)
      if (!(target > 0. && obs_window > 0.) || obs <= obs_window * target) pts.emplace_back(obs, x);
    if (pts.size() < 3) return;
    std::sort(pts.begin(), pts.end()); // ascending in obs: closest to criticality first

    const auto k = select_triple(pts.size());
    const std::array<double, 3> xs{{pts[k[0]].second, pts[k[1]].second, pts[k[2]].second}};
    const std::array<double, 3> ys{{pts[k[0]].first, pts[k[1]].first, pts[k[2]].first}};
    const auto fit = detail::fit_power_law_3(xs, ys);
    if (!fit.valid) return;
    if (!(fit.exponent >= beta_min && fit.exponent <= beta_max)) return;

    const double x_c = *std::min_element(xs.begin(), xs.end()) - fit.shift;
    if (!(x_c > x_lo_hard && x_c < x_hi)) return;

    // Two fits that disagree by an appreciable fraction of the bracket mean the model is still
    // picking up corrections to scaling; keep the newer one but do not act on it this step.
    const double w = x_hi - x_lo_soft;
    const bool agrees = !std::isfinite(model_x_c) || !std::isfinite(w) || std::abs(x_c - model_x_c) < 0.1 * w;

    model_x_c_prev = model_x_c;
    model_x_c = x_c;
    model_beta = fit.exponent;
    model_log_C = fit.log_amplitude;
    model_valid = agrees;
  }

  inline void ScalingRootFinder::fit_divergent()
  {
    divergent_valid = false;

    std::vector<std::pair<double, double>> pts; // (residual, x)
    for (const auto &[x, r] : F)
      if (r >= residual_window) pts.emplace_back(r, x);
    if (pts.size() < 3) return;
    std::sort(pts.begin(), pts.end(), [](const auto &a, const auto &b) { return a.first > b.first; });

    // Largest residual (closest to the critical point) plus two spread across the rest.
    // Order them by x ascending, so x3 is the closest from below and r3 the largest.
    const auto k = select_triple(pts.size());
    std::array<std::pair<double, double>, 3> t{{pts[k[0]], pts[k[1]], pts[k[2]]}}; // (r, x)
    std::sort(t.begin(), t.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
    const double x1 = t[0].second, x2 = t[1].second, x3 = t[2].second;
    const double r1 = t[0].first, r2 = t[1].first, r3 = t[2].first;
    if (!(x1 < x2 && x2 < x3)) return;
    if (!(r1 < r2 && r2 < r3)) return;

    // Mirror onto the convergent form: X = -x, y = exp(-r). Only residual *differences* enter,
    // so the exponentials are never formed and a large final_time cannot underflow the fit.
    const double A = r2 - r1, B = r3 - r2;
    const double d1 = x3 - x1, d2 = x3 - x2;
    const double s = detail::solve_scaling_shift(A, B, d1, d2);
    if (!std::isfinite(s)) return;

    const double beta = A / std::log((s + d1) / (s + d2));
    if (!std::isfinite(beta) || beta <= 0.) return;
    const double theta = 1. / beta;
    if (!(theta >= theta_min && theta <= theta_max)) return;

    const double x_c = x3 + s;
    if (!(x_c > x_lo_hard && x_c < x_hi)) return;

    model_theta = theta;
    divergent_x_c = x_c;
    divergent_shift = s;
    divergent_valid = true;
  }

  inline double ScalingRootFinder::propose()
  {
    // 1. Seed with the two hypothesised bounds.
    if (!probed_hi_init) {
      probed_hi_init = true;
      return x_hi_init;
    }
    if (!probed_lo_init) {
      probed_lo_init = true;
      return x_lo_init;
    }

    const bool have_hi = std::isfinite(x_hi);
    const bool have_lo = std::isfinite(x_lo_soft);

    // 2. One-sided: widen. The divergent fit, when available, replaces blind doubling with a
    // targeted jump -- this is what pays for the approach phase.
    if (!have_hi || !have_lo) {
      if (!bounds_are_hypotheses || expansions >= expansion_max_iter) return NAN;
      if (!have_hi) {
        if (divergent_valid) {
          const double cand = divergent_x_c + divergent_shift;
          if (cand > x_lo_soft && std::isfinite(cand)) {
            counts.model_residual++;
            return cand;
          }
        }
        ++expansions;
        counts.expand++;
        return x_lo_soft + std::pow(expansion_factor, (double)expansions) * w_init;
      }
      ++expansions;
      counts.expand++;
      return x_hi - std::pow(expansion_factor, (double)expansions) * w_init;
    }

    const double w = x_hi - x_lo_soft;
    double cand = NAN;
    enum Source { NONE, MODEL_OBS, MODEL_MIXED, APPROACH, MODEL_RESIDUAL, SECANT } source = NONE;

    // 3. Convergent fit: the endgame.
    if (model_valid) {
      cand = (target > 0.) ? extrapolate(model_x_c)
                           // Deep scaling: the target IS the singular point. Approach the fitted
                           // x_c geometrically from the convergent side and never land on it.
                           : model_x_c + 0.1 * (x_hi - model_x_c);
      if (std::isfinite(cand)) source = MODEL_OBS;
    }

    // 4. Divergent x_c plus two convergent points. The divergent branch pins x_c long before the
    // convergent one does, and with x_c known the exponent and amplitude follow from any two
    // convergent points -- no third point needed. They must still be inside the scaling window:
    // an amplitude read off at obs ~ 100x the target predicts the target's location decades out.
    if (source == NONE && divergent_valid && target > 0. && S.size() >= 2) {
      const double window = (obs_window > 0.) ? obs_window * target : std::numeric_limits<double>::infinity();
      std::vector<std::pair<double, double>> in; // (obs, x), closest to criticality first
      for (const auto &[x, y] : S)
        if (y <= window && x > divergent_x_c) in.emplace_back(y, x);
      std::sort(in.begin(), in.end());
      if (in.size() >= 2) {
        const auto [ya, xa] = in[0];
        const auto [yb, xb] = in[1];
        const double da = xa - divergent_x_c, db = xb - divergent_x_c;
        if (da > 0. && db > 0. && ya > 0. && yb > 0. && da != db && ya != yb) {
          const double beta = std::log(ya / yb) / std::log(da / db);
          if (beta >= beta_min && beta <= beta_max) {
            cand = divergent_x_c + da * std::exp((std::log(effective_target()) - std::log(ya)) / beta);
            if (std::isfinite(cand)) source = MODEL_MIXED;
          }
        }
      }
    }

    // 5. A critical point is known but nothing convergent lies inside the scaling window yet.
    // This is the approach phase, and it is where a bisection is worst: the target can sit a
    // billionth of x_c away (measured: 1.4e-9 relative on a cold SP tune), so halving the bracket
    // needs ~30 flows to cover ground a geometric ladder covers in a handful. Step a fixed
    // fraction of the remaining distance to x_c -- one decade per flow -- from the safe side.
    if (source == NONE && target > 0. && std::isfinite(get_x_critical())) {
      const double x_c = get_x_critical();
      if (x_hi > x_c) {
        // Never take a step a bisection would beat: capping at the midpoint means the bracket
        // still halves every iteration, so this cannot stall and cannot trip the stagnation
        // rule, while a good critical point makes it cover a decade instead of a factor two.
        cand = std::min(x_c + approach_factor * (x_hi - x_c), 0.5 * (x_lo_soft + x_hi));
        if (std::isfinite(cand)) source = APPROACH;
      }
    }

    // 6. Divergent fit alone, no convergent point at all yet: step to the mirror of the
    // closest failing probe across x_c. Scale-free, and needs no amplitude.
    if (source == NONE && divergent_valid) {
      cand = divergent_x_c + divergent_shift;
      if (std::isfinite(cand)) source = MODEL_RESIDUAL;
    }

    // 7. Two convergent points, no critical point yet: secant in log obs.
    if (source == NONE && target > 0. && S.size() >= 2) {
      const auto &[xa, ya] = S[S.size() - 2];
      const auto &[xb, yb] = S[S.size() - 1];
      if (ya > 0. && yb > 0. && ya != yb) {
        const double la = std::log(ya / effective_target()), lb = std::log(yb / effective_target());
        cand = xb - lb * (xb - xa) / (lb - la);
        if (std::isfinite(cand)) source = SECANT;
      }
    }

    // Safeguards: containment, then standoff, then Brent stagnation.
    if (source != NONE) {
      if (!(cand > x_lo_soft && cand < x_hi) || cand <= x_lo_hard)
        source = NONE; // outside the live bracket, or contradicts a known-divergent point
      else if (std::isfinite(w_prevprev) && w > 0.5 * w_prevprev) {
        // If the bracket has not halved over the last two steps the model is not earning its
        // place. One forced bisection caps its cost at a factor two over plain bisection.
        source = NONE;
        counts.forced_bisect++;
      } else
        cand = std::min(std::max(cand, x_lo_soft + endpoint_standoff * w), x_hi - endpoint_standoff * w);
    }

    switch (source) {
    case MODEL_OBS:
      counts.model_obs++;
      return cand;
    case MODEL_MIXED:
      counts.model_mixed++;
      return cand;
    case APPROACH:
      counts.approach++;
      return cand;
    case MODEL_RESIDUAL:
      counts.model_residual++;
      return cand;
    case SECANT:
      counts.secant++;
      return cand;
    default:
      counts.bisect++;
      return 0.5 * (x_lo_soft + x_hi);
    }
  }

  inline double ScalingRootFinder::search()
  {
    if (!(std::isfinite(x_lo_init) && std::isfinite(x_hi_init)))
      throw std::runtime_error("ScalingRootFinder: search bounds were not set");
    if (!(x_lo_init < x_hi_init))
      throw std::runtime_error("ScalingRootFinder: search bounds are not ordered (x_lo < x_hi)");
    if (!(target >= 0.)) throw std::runtime_error("ScalingRootFinder: target must be non-negative");

    w_init = x_hi_init - x_lo_init;
    m_converged = false;
    answer_found = false;
    if (std::isfinite(seeded_exponent) && seeded_exponent > 0.) model_beta = seeded_exponent;

    for (iter = 0; iter < max_iter; ++iter) {
      const double x = propose();
      if (!std::isfinite(x)) {
        spdlog::warn("ScalingRootFinder: no admissible proposal at iteration {}; stopping", iter);
        break;
      }

      double obs = NAN, residual = NAN;
      const bool ok = f(x, obs, residual);
      record(x, ok, obs, residual);
      refit();

      if (answer_found) {
        m_converged = true;
        break;
      }

      if (target > 0. && acceptance == Acceptance::AtTarget) {
        if (have_success && std::isfinite(best_obs) && std::abs(best_obs - target) <= rel_tol * target &&
            (!one_sided || best_obs >= target)) {
          m_converged = true;
          break;
        }
      } else if (target <= 0. && model_valid && std::isfinite(model_x_c_prev)) {
        // Deep scaling: the observable tolerance is vacuous, so converge the fitted critical
        // point instead.
        if (std::abs(model_x_c - model_x_c_prev) <= rel_tol * std::abs(model_x_c)) {
          m_converged = true;
          break;
        }
      }

      if (std::isfinite(x_hi) && std::isfinite(x_lo_soft)) {
        const double scale = std::max({std::abs(x_hi), std::abs(x_lo_soft), 1.});
        if ((x_hi - x_lo_soft) <= x_rel_floor * scale) {
          spdlog::warn("ScalingRootFinder: bracket collapsed to [{:.12e}, {:.12e}] after {} evaluations "
                       "without reaching the tolerance {:.3e}",
                       x_lo_soft, x_hi, counts.total, rel_tol);
          break;
        }
      }

      // Three consecutive successes whose observable does not move: obs is insensitive to x
      // here, so no amount of further bracketing will resolve the target.
      if (S.size() >= 3) {
        const double y1 = S[S.size() - 1].second, y2 = S[S.size() - 2].second, y3 = S[S.size() - 3].second;
        const double scale = std::max(std::abs(y1), 1e-300);
        if (std::abs(y1 - y2) < 1e-14 * scale && std::abs(y2 - y3) < 1e-14 * scale) {
          spdlog::warn("ScalingRootFinder: observable stagnated at {:.12e}; stopping", y1);
          break;
        }
      }
    }

    if (!have_success) throw std::runtime_error("ScalingRootFinder: no evaluation of the search interval succeeded");

    if (!m_converged)
      spdlog::warn("ScalingRootFinder: stopped after {} evaluations without reaching the tolerance {:.3e}; "
                   "returning x = {:.12e} with obs = {:.12e}",
                   counts.total, rel_tol, best_x, best_obs);

    return best_x;
  }
} // namespace DiFfRG