#pragma once

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
   * @brief Checks that an interpolator class provides the required methods (get_coordinates) and a
   * call operator with the correct arity.
   */
  template <typename T>
  concept has_interpolator_methods = has_interpolator_types<T> && requires(T t) {
    t.get_coordinates();
    requires has_n_call_operator<T, typename T::ctype, T::dim>;
  };

  /**
   * @brief A concept for what is an interpolator class
   *
   * An interpolator class must provide the following methods:
   * - get_coordinates(): returns the coordinates object used for the interpolation
   * - operator()(x...): interpolates at a point, callable from host AND device code
   *
   * The interpolator class must also define the following type aliases:
   * - value_type: the type of the values being interpolated
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