#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>

// std
#include <type_traits>
#include <utility>

namespace DiFfRG
{
  /**
   * @brief Whether T hands kernels a compact handle of itself, via `t.handle<ctype>()`.
   *
   * A handle is what an integrator passes to a kernel in place of the interpolator: read-only
   * views of the device and host buffers plus the coordinates, in the precision ctype the kernel
   * computes in. It carries no Kokkos bookkeeping and no copy in the other precision, which keeps
   * the launch arguments small (they go through the 32 kB of constant memory a CUDA launch has).
   *
   * A class deriving from an interpolator inherits handle(). If it changes what the interpolator
   * returns, it has to shadow handle() as well, or kernels taking handles bypass the change.
   */
  template <typename T>
  concept has_kernel_handle = requires(const T &t) {
    t.template handle<float>();
    t.template handle<double>();
  };

  /// What an integrator computing in ctype passes to its kernel in place of t: its handle, or t itself.
  template <typename ctype, typename T> decltype(auto) to_kernel_handle(const T &t)
  {
    if constexpr (has_kernel_handle<T>)
      return t.template handle<ctype>();
    else
      return (t);
  }

  template <typename T, typename ctype>
  using kernel_handle_t = std::remove_cvref_t<decltype(to_kernel_handle<ctype>(std::declval<const T &>()))>;

  namespace internal
  {
    /// The coordinates in single precision, where rebind_ctype knows them; otherwise unchanged.
    template <typename Coordinates> struct _single_precision_coordinates {
      using type = Coordinates;
    };
    template <typename Coordinates>
      requires requires { typename rebind_ctype<Coordinates, float>::type; }
    struct _single_precision_coordinates<Coordinates> {
      using type = rebind_ctype_t<Coordinates, float>;
    };

    /// Read-only access to a contiguous 1D buffer, as stored in a handle: a bare pointer, unlike a
    /// Kokkos::View, which costs 32 B of launch arguments even when unmanaged.
    template <typename NT> struct RawView1D {
      const NT *data;
      KOKKOS_FORCEINLINE_FUNCTION NT operator()(const size_t i) const { return data[i]; }
    };

    /// Read-only access to a contiguous 2D buffer in Kokkos layout Layout, see RawView1D.
    template <typename NT, typename Layout> struct RawView2D {
      const NT *data;
      device::array<size_t, 2> sizes;
      KOKKOS_FORCEINLINE_FUNCTION NT operator()(const size_t i, const size_t j) const
      {
        if constexpr (std::is_same_v<Layout, Kokkos::LayoutLeft>)
          return data[i + sizes[0] * j];
        else {
          static_assert(std::is_same_v<Layout, Kokkos::LayoutRight>, "RawView2D: unsupported layout");
          return data[i * sizes[1] + j];
        }
      }
    };

    /// Read-only access to a contiguous 3D buffer in Kokkos layout Layout, see RawView1D.
    template <typename NT, typename Layout> struct RawView3D {
      const NT *data;
      device::array<size_t, 3> sizes;
      KOKKOS_FORCEINLINE_FUNCTION NT operator()(const size_t i, const size_t j, const size_t k) const
      {
        if constexpr (std::is_same_v<Layout, Kokkos::LayoutLeft>)
          return data[i + sizes[0] * (j + sizes[1] * k)];
        else {
          static_assert(std::is_same_v<Layout, Kokkos::LayoutRight>, "RawView3D: unsupported layout");
          return data[(i * sizes[1] + j) * sizes[2] + k];
        }
      }
    };
  } // namespace internal

  /**
   * @brief An interpolator's single-precision twin: value type and coordinates, see
   * SplineInterpolator1D. It exists only for double (or complex double) data on coordinates that
   * rebind_ctype can turn into float.
   */
  template <typename NT, typename Coordinates> struct SinglePrecisionTwin {
    using value_type = get_type::single_precision<NT>;
    using coordinates_type = typename internal::_single_precision_coordinates<Coordinates>::type;
    static constexpr bool exists =
        !std::is_same_v<value_type, NT> && !std::is_same_v<coordinates_type, Coordinates>;
  };
} // namespace DiFfRG
