#pragma once

// standard library
#include <vector>

// external libraries
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <gsl/gsl_spline.h>

namespace DiFfRG
{
  namespace Interpolation
  {
    /**
     * @brief This class takes in x-dependent data and interpolates it to a given x on request.
     *
     * Note, that interpolations beyond the range of the data will return the value at the closest boundary.
     *
     * Internally, the boost barycentric rational interpolator is used.
     *
     */
    class Barycentric
    {
    public:
      /**
       * @brief Construct a new Interpolator object
       *
       * @param x The x values of the data.
       * @param y The y values of the data.
       * @param order The order of the interpolator.
       * @param copy Whether to copy the data or move it into the interpolator, making the original data unusable.
       */
      Barycentric(std::vector<double> &x, std::vector<double> &y, uint order = 1, bool copy = true);

      /**
       * @brief Construct a new Interpolator object
       *
       * @param x The x values of the data.
       * @param y The y values of the data.
       * @param order The order of the interpolator.
       */
      Barycentric(const std::vector<double> &x, const std::vector<double> &y, uint order = 1);

      /**
       * @brief Construct an empty Interpolator object
       */
      Barycentric();

      /**
       * @brief Check whether the interpolator is consistent with the original data.
       * The interpolator checks if it can reproduce the original data with a given relative precision.
       *
       * @param tolerance The relative precision to use.
       * @return true If the interpolator is consistent with the original data.
       */
      bool check_consistency(double tolerance) const;

      /**
       * @brief Interpolate the data to a given x.
       *
       * @param x The x to interpolate to.
       * @return double The interpolated data.
       */
      double operator()(double x) const;

      /**
       * @brief Interpolate the data to a given x.
       *
       * @param x The x to interpolate to.
       * @return double The interpolated data.
       */
      double value(double x) const;

      /**
       * @brief Interpolate the derivative of the data to a given x.
       *
       * @param x The x to interpolate to.
       * @return double The interpolated derivative.
       */
      double derivative(double x) const;

    private:
      std::unique_ptr<boost::math::interpolators::barycentric_rational<double>> interpolator;
    };

    /**
     * @brief This class takes in x-dependent data and interpolates it to a given x on request.
     * This class uses the cubic spline methods from gsl to interpolate the data.
     */
    class CubicSpline
    {
    public:
      CubicSpline(const std::vector<double> &x, const std::vector<double> &y);

      ~CubicSpline();

      double operator()(double x) const;

      double value(double x) const;

      double derivative(double x) const;

    private:
      gsl_interp_accel *acc;
      gsl_spline *spline;
    };
  } // namespace Interpolation
} // namespace DiFfRG
