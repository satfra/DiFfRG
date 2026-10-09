#pragma once

// DiFfRG
#include <DiFfRG/physics/interpolation/linear_interpolator_1d.hh>
#include <DiFfRG/physics/interpolation/linear_interpolator_2d.hh>
#include <DiFfRG/physics/interpolation/linear_interpolator_3d.hh>

namespace DiFfRG
{
  namespace internal
  {
    template <typename T, int dim> struct _has_dim {
      static constexpr bool value = (T::dim == dim);
    };
    template <typename T, int dim> constexpr bool has_dim = _has_dim<T, dim>::value;
  } // namespace internal

  template <size_t dim, typename NT, typename Coordinates> struct LinearInterpolatorND_helper;

  template <typename NT, typename Coordinates> struct LinearInterpolatorND_helper<1, NT, Coordinates> {
    using type = LinearInterpolator1D<NT, Coordinates>;
  };

  template <typename NT, typename Coordinates> struct LinearInterpolatorND_helper<2, NT, Coordinates> {
    using type = LinearInterpolator2D<NT, Coordinates>;
  };

  template <typename NT, typename Coordinates> struct LinearInterpolatorND_helper<3, NT, Coordinates> {
    using type = LinearInterpolator3D<NT, Coordinates>;
  };

  /**
   * @brief A linear interpolator for ND data, callable from host and device code alike.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates>
  using LinearInterpolatorND = typename LinearInterpolatorND_helper<Coordinates::dim, NT, Coordinates>::type;
} // namespace DiFfRG
