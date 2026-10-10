#pragma once

// standard library
#include <memory>
#include <vector>

// external libraries
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <gsl/gsl_spline.h>

/**
 * @file interpolation.hh
 * @brief One-dimensional interpolation of tabulated data \f$(x_i, y_i)\f$ on the CPU.
 *
 * For interpolators that are evaluated inside GPU kernels, see physics/interpolation.hh.
 */

namespace DiFfRG
{
  /// Interpolation of tabulated one-dimensional data.
  namespace Interpolation
  {
    /**
     * @brief Barycentric rational interpolation (Floater-Hormann) of data \f$(x_i, y_i)\f$.
     *
     * Works on unevenly spaced \f$x_i\f$ and has no spurious poles. Outside \f$[\min x_i, \max x_i]\f$ the rational
     * function is extrapolated, not clamped. Uses boost::math::interpolators::barycentric_rational.
     */
    class Barycentric
    {
    public:
      /**
       * @brief Interpolate the data \f$(x_i, y_i)\f$.
       *
       * @param x the nodes \f$x_i\f$, distinct.
       * @param y the values \f$y_i\f$, same length as `x`.
       * @param order the approximation order \f$d\f$ of the Floater-Hormann scheme (\f$0 \leq d < \f$ number of
       * points); higher is smoother but less robust.
       * @param copy if false, `x` and `y` are moved into the interpolator and left empty.
       */
      Barycentric(std::vector<double> &x, std::vector<double> &y, uint order = 1, bool copy = true);

      /**
       * @brief Interpolate the data \f$(x_i, y_i)\f$; see the other constructor.
       */
      Barycentric(const std::vector<double> &x, const std::vector<double> &y, uint order = 1);

      /// An empty interpolator; must be assigned before it is evaluated.
      Barycentric();

      /// The interpolant at `x`.
      double operator()(double x) const;

      /// The interpolant at `x`.
      double value(double x) const;

      /// The derivative of the interpolant at `x`.
      double derivative(double x) const;

    private:
      std::unique_ptr<boost::math::interpolators::barycentric_rational<double>> interpolator;
    };

    /**
     * @brief Natural cubic spline through data \f$(x_i, y_i)\f$ (second derivative zero at both ends), via GSL.
     *
     * Needs at least 3 points with strictly increasing \f$x_i\f$. Evaluation outside \f$[x_0, x_{n-1}]\f$ is an error
     * reported through GSL's error handler. Not copyable; evaluation is not thread-safe (it updates a lookup cache).
     */
    class CubicSpline
    {
    public:
      /// Build the spline through \f$(x_i, y_i)\f$.
      CubicSpline(const std::vector<double> &x, const std::vector<double> &y);

      ~CubicSpline();

      CubicSpline(const CubicSpline &) = delete;
      CubicSpline &operator=(const CubicSpline &) = delete;

      /// The spline at `x`.
      double operator()(double x) const;

      /// The spline at `x`.
      double value(double x) const;

      /// The derivative of the spline at `x`.
      double derivative(double x) const;

    private:
      gsl_interp_accel *acc;
      gsl_spline *spline;
    };
  } // namespace Interpolation
} // namespace DiFfRG
