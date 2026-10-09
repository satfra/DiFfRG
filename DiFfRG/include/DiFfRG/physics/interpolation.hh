#pragma once

/**
 * @file interpolation.hh
 * @brief Interpolators for data on a grid, e.g. momentum-dependent couplings fed into integration kernels.
 *
 * Available interpolators:
 * - LinearInterpolator1D, LinearInterpolator2D, LinearInterpolator3D (or LinearInterpolatorND, chosen by dimension):
 *   (multi-)linear interpolation. Supports periodic axes.
 * - PeriodicCubicInterpolator3D: as LinearInterpolator3D, but cubic (and smooth) along periodic axes.
 * - SplineInterpolator1D: cubic spline in one variable.
 * - SplineInterpolator1DStack: one cubic spline in \f$x\f$ per value of a second, stacked variable \f$s\f$ (e.g. a
 *   Matsubara frequency), linear between them; called as `f(s, x)`.
 *
 * All of them are used the same way:
 * - Construct from a coordinate system (discretization/coordinates/), which defines the grid. The data starts at zero.
 * - Set the data with `update(ptr)`: `ptr` points to `coordinates.size()` values, row-major for more than one axis
 *   (the last axis runs fastest).
 * - Evaluate with `f(x...)` on host or device. Outside the grid the boundary value is used; periodic axes wrap.
 * - To use one in an integration kernel, pass it as an argument to the integrator's get()/map(); the kernel should
 *   take it as `const auto &` and receives a lightweight read-only handle with the same call operator (see
 *   has_kernel_handle).
 *
 * @code
 * LogarithmicCoordinates1D<double> coords(64, 1e-3, 40., 2.); // 64 points, p in [1e-3, 40], bias 2
 * SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> Z(coords);
 * std::vector<double> values(coords.size());
 * for (size_t i = 0; i < coords.size(); ++i)
 *   values[i] = 1. + coords.forward(i); // data at the grid points
 * Z.update(values.data());
 * const double z = Z(0.5); // interpolated value at p = 0.5
 * @endcode
 */

#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/types.hh>

namespace DiFfRG
{
  /**
   * @brief Checks that an interpolator class provides the required type aliases (value_type, ctype).
   */
  template <typename T>
  concept has_interpolator_types = requires {
    typename T::value_type;
    typename T::ctype;
  };

  /**
   * @brief Checks that an interpolator class provides get_coordinates() and a call operator taking `T::dim`
   * coordinates.
   */
  template <typename T>
  concept has_interpolator_methods = has_interpolator_types<T> && requires(T t) {
    t.get_coordinates();
    requires has_n_call_operator<T, typename T::ctype, T::dim>;
  };

  /**
   * @brief What makes a type an interpolator.
   *
   * Required members:
   * - `value_type`: type of the interpolated values
   * - `ctype`: type of the coordinates
   * - `dim`: number of coordinates
   * - `get_coordinates()`: the coordinate system of the grid
   * - `operator()(x_1, ..., x_dim)`: the interpolated value at a point
   *
   * @tparam T The type to check
   */
  template <typename T>
  concept is_interpolator = has_interpolator_types<T> && has_interpolator_methods<T>;
} // namespace DiFfRG

#include <DiFfRG/discretization/coordinates/combined_coordinates.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/discretization/coordinates/stack_coordinates.hh>

#include <DiFfRG/physics/interpolation/linear_interpolator.hh>

#include <DiFfRG/physics/interpolation/spline_interpolator_1d.hh>
#include <DiFfRG/physics/interpolation/spline_interpolator_1d_stack.hh>