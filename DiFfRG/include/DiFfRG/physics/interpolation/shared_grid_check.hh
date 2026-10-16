#pragma once

// std
#include <stdexcept>
#include <string>

namespace DiFfRG
{
  namespace internal
  {
    /// The coordinates of an interpolator as a string; an argument of map_points() is looked through (an
    /// interpolator can only be shared by all points there, so it is the PointArg's single value).
    template <typename T> std::string shared_grid_coordinates(const T &interpolator)
    {
      if constexpr (requires { interpolator.get_coordinates().to_string(); })
        return interpolator.get_coordinates().to_string();
      else if constexpr (requires { interpolator.value; })
        return shared_grid_coordinates(interpolator.value);
      else {
        static_assert(sizeof(T) == 0, "SharedGrids: an interpolator in this group cannot report its coordinates "
                                      "(get_coordinates().to_string()); remove it from the group.");
        return {};
      }
    }
  } // namespace internal

  /**
   * @brief Throw if the given interpolators do not all live on the same coordinate grid.
   *
   * Generated kernels with "SharedGrids" compute the coordinate transform of a lookup once and reuse it for every
   * interpolator of the group (`f.at(f.index(x))`, then `g.at(...)` with the same index), which is only correct if
   * the group really shares one grid. The generated map(), get() and map_points() call this in builds without
   * NDEBUG, or with DIFFRG_CHECK_SHARED_GRIDS defined.
   *
   * @param names the interpolators' names, for the error message
   */
  template <typename First, typename... Rest>
  void check_shared_grid(const char *names, const First &first, const Rest &...rest)
  {
    const std::string grid = internal::shared_grid_coordinates(first);
    if (((internal::shared_grid_coordinates(rest) != grid) || ...))
      throw std::runtime_error(std::string("SharedGrids: the interpolators ") + names +
                               " were declared to share one grid, but their coordinates differ.");
  }
} // namespace DiFfRG
