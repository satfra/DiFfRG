#pragma once

// standard library
#include <cstddef>

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>

namespace DiFfRG
{
  /**
   * @brief The two grid indices and the interpolation weight for one axis of a linear interpolation.
   */
  template <typename CT> struct InterpolationStencil {
    size_t lower, upper;
    CT t;
  };

  /**
   * @brief Resolve a fractional grid index into the linear-interpolation stencil along one axis.
   *
   * For a non-periodic axis the index is clamped to [0, n-1] and the stencil is [lower, lower+1] with lower <= n-2,
   * i.e. evaluations outside the grid are constant-extrapolated from the boundary cell.
   *
   * For a periodic axis no clamping happens - the coordinate's backward() has already folded the index into [0, n) -
   * and the upper index wraps around to 0 in the last cell, which closes the grid across the seam.
   *
   * @tparam periodic whether the axis is periodic, see is_periodic_coordinate_v / is_periodic_axis_v
   * @param idx fractional grid index, as returned by Coordinates::backward
   * @param n number of grid points along the axis
   */
  template <bool periodic, typename CT>
  KOKKOS_FORCEINLINE_FUNCTION InterpolationStencil<CT> make_interpolation_stencil(CT idx, const size_t n)
  {
    using Kokkos::floor, Kokkos::max, Kokkos::min;
    if constexpr (periodic) {
      // backward() folds into [0, n), the min/max only guard against rounding at the seam
      const size_t lower = min(size_t(max(static_cast<CT>(0), floor(idx))), n - 1);
      return {lower, (lower + 1) % n, idx - static_cast<CT>(lower)};
    } else {
      idx = max(static_cast<CT>(0), min(idx, static_cast<CT>(n - 1)));
      const size_t lower = min(size_t(floor(idx)), n - 2);
      return {lower, lower + 1, idx - static_cast<CT>(lower)};
    }
  }
} // namespace DiFfRG
