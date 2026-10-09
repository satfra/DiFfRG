#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/interpolation/interpolation_stencil.hh>
#include <DiFfRG/physics/interpolation/interpolator_handle.hh>

// std
#include <optional>
#include <stdexcept>

namespace DiFfRG
{
  namespace internal
  {
    /// The bilinear evaluation at fractional grid indices, shared by LinearInterpolator2D and its handle.
    template <typename NT, typename Coordinates, typename CT, typename View>
    KOKKOS_FORCEINLINE_FUNCTION NT linear_2d_at(const View &data, const device::array<size_t, 2> &sizes,
                                                const device::array<CT, 2> &idx)
    {
      // Clamped [i, i+1] stencil on bounded axes, wrapping [i, (i+1) % n] on periodic ones
      const auto sx = make_interpolation_stencil<is_periodic_axis_v<Coordinates, 0>>(idx[0], sizes[0]);
      const auto sy = make_interpolation_stencil<is_periodic_axis_v<Coordinates, 1>>(idx[1], sizes[1]);

      const size_t x0 = sx.lower, x1 = sx.upper;
      const size_t y0 = sy.lower, y1 = sy.upper;

      const NT corner00 = data(x0, y0);
      const NT corner01 = data(x0, y1);
      const NT corner10 = data(x1, y0);
      const NT corner11 = data(x1, y1);

      const auto tx = sx.t;
      const auto ty = sy.t;

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(ty, Kokkos::fma(tx, corner11, Kokkos::fma(-tx, corner01, corner01)),
                           (1 - ty) * Kokkos::fma(tx, corner10, Kokkos::fma(-tx, corner00, corner00)));
      else
        return corner00 * (1 - tx) * (1 - ty) + corner01 * (1 - tx) * ty + corner10 * tx * (1 - ty) +
               corner11 * tx * ty;
    }
  } // namespace internal

  /**
   * @brief Read-only view of a LinearInterpolator2D, as passed to integration kernels.
   *
   * Obtained from LinearInterpolator2D::handle(); offers the same index(), at() and operator(). See has_kernel_handle.
   */
  template <typename NT, typename Coordinates, typename Layout> class LinearInterpolator2DHandle
  {
  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 2;

    LinearInterpolator2DHandle(const NT *device_data, const NT *host_data, const device::array<size_t, 2> &sizes,
                               const Coordinates &coordinates)
        : device_data(device_data), host_data(host_data), sizes(sizes), coordinates(coordinates)
    {
    }

    device::array<ctype, 2> KOKKOS_FUNCTION index(const ctype x, const ctype y) const
    {
      return coordinates.backward(x, y);
    }

    NT KOKKOS_FUNCTION at(const device::array<ctype, 2> &idx) const
    {
      KOKKOS_IF_ON_DEVICE((return internal::linear_2d_at<NT, Coordinates>(View{device_data, sizes}, sizes, idx);))
      KOKKOS_IF_ON_HOST((return internal::linear_2d_at<NT, Coordinates>(View{host_data, sizes}, sizes, idx);))
    }

    NT KOKKOS_FUNCTION operator()(const ctype x, const ctype y) const { return at(index(x, y)); }

    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    using View = internal::RawView2D<NT, Layout>;
    const NT *device_data, *host_data;
    device::array<size_t, 2> sizes;
    Coordinates coordinates;
  };

  /**
   * @brief Bilinear interpolation of data given on a 2D grid, usable in host and device code.
   *
   * Each axis is interpolated linearly in its fractional grid index (Coordinates::backward). Outside the grid the
   * value at the nearest boundary point is used (constant extrapolation); periodic axes wrap around instead. Every
   * bounded axis needs at least 2 points.
   *
   * The data is zero after construction; set it with update(). Host and device copies are kept in sync. A
   * double-precision interpolator also keeps a single-precision copy for single-precision kernels (see handle()).
   *
   * @tparam NT value type of the data (real, complex or autodiff)
   * @tparam Coordinates 2D coordinate system of the grid, e.g. CoordinatePackND of two 1D coordinate systems
   */
  template <typename NT, typename Coordinates> class LinearInterpolator2D
  {
    static_assert(Coordinates::dim == 2, "LinearInterpolator2D requires 2D coordinates");

    using ViewType = Kokkos::View<NT **, GPU_memory, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::host_mirror_type;

    static constexpr bool has_separate_device =
        !std::is_same_v<typename ViewType::memory_space, typename HostViewType::memory_space>;

    using Twin = SinglePrecisionTwin<NT, Coordinates>;
    using NT32 = typename Twin::value_type;
    using Coordinates32 = typename Twin::coordinates_type;
    using ViewType32 = Kokkos::View<NT32 **, GPU_memory>;
    using HostViewType32 = typename ViewType32::host_mirror_type;

  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 2;

    /**
     * @brief Allocate zeroed data on the grid of `coordinates`.
     *
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator2D(const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()),
          device_data("LinearInterpolator2D_data", coordinates.sizes()[0], coordinates.sizes()[1]),
          host_data(Kokkos::create_mirror_view(device_data))
    {
      // the handles index the buffers by hand, which needs them unpadded
      if (device_data.span() != sizes[0] * sizes[1] || host_data.span() != device_data.span())
        throw std::runtime_error("LinearInterpolator2D: padded data buffer");
      if constexpr (Twin::exists) {
        // SequentialHostInit: Single holds device views, which must not be created or destroyed
        // inside a host parallel region (the default for a view's element), or teardown deadlocks.
        single = Kokkos::View<Single, Kokkos::HostSpace>(
            Kokkos::view_alloc("LinearInterpolator2D_single", Kokkos::SequentialHostInit));
        auto &s = single();
        s.coordinates.emplace(coordinates);
        s.device_data = ViewType32("LinearInterpolator2D_data32", sizes[0], sizes[1]);
        s.host_data = Kokkos::create_mirror_view(s.device_data);
        if (s.device_data.span() != device_data.span() || s.host_data.span() != device_data.span())
          throw std::runtime_error("LinearInterpolator2D: padded data buffer");
      }
    }

    /// Shallow copy: the copy shares the data, so later update() calls are seen by both.
    KOKKOS_DEFAULTED_FUNCTION LinearInterpolator2D(const LinearInterpolator2D &) = default;

    /**
     * @brief Replace the data. Host and device copies are both current when this returns.
     *
     * @param in_data values at the grid points in row-major order, `in_data[i * n1 + j]` for grid point (i, j), with
     * `n1 = get_coordinates().sizes()[1]`; converted to NT
     */
    template <typename NT2> void update(const NT2 *in_data)
    {
      // Filled element-wise: the views' storage layout need not be row-major. See
      // LinearInterpolator1D::update() for the copy and fence.
      for (size_t i = 0; i < sizes[0]; ++i)
        for (size_t j = 0; j < sizes[1]; ++j)
          host_data(i, j) = in_data[i * sizes[1] + j];

      if constexpr (Twin::exists)
        for (size_t i = 0; i < sizes[0]; ++i)
          for (size_t j = 0; j < sizes[1]; ++j)
            single().host_data(i, j) = static_cast<NT32>(host_data(i, j));

      if constexpr (has_separate_device) {
        typename ViewType::execution_space exec;
        Kokkos::deep_copy(exec, device_data, host_data);
        if constexpr (Twin::exists) Kokkos::deep_copy(exec, single().device_data, single().host_data);
        exec.fence();
      }
    }

    /**
     * @brief The read-only view a kernel computing in precision CT receives (see has_kernel_handle).
     *
     * For CT = float and double data this reads the single-precision copy, otherwise the data itself.
     */
    template <typename CT> auto handle() const
    {
      if constexpr (Twin::exists && std::is_same_v<CT, float>) {
        const auto &s = single();
        return LinearInterpolator2DHandle<NT32, Coordinates32, typename ViewType32::array_layout>(
            s.device_data.data(), s.host_data.data(), sizes, *s.coordinates);
      } else
        return LinearInterpolator2DHandle<NT, Coordinates, typename ViewType::array_layout>(
            device_data.data(), host_data.data(), sizes, coordinates);
    }

    /**
     * @brief Value at flat row-major index i, the order update() takes (host only).
     */
    NT operator[](size_t i) const { return host_data(i / sizes[1], i % sizes[1]); }

    /**
     * @brief Fractional grid indices of the point (x, y), i.e. Coordinates::backward(x, y).
     *
     * `f(x, y) == f.at(f.index(x, y))`. Several interpolators on the same coordinate system can share one index()
     * call, which saves the (possibly expensive) coordinate transform.
     *
     * @param x first physical coordinate
     * @param y second physical coordinate
     */
    // Return type spelled out: nvcc loses decltype(var) of a deduced return type inside a class-template member.
    device::array<typename Coordinates::ctype, 2> KOKKOS_FUNCTION
    index(const typename Coordinates::ctype x, const typename Coordinates::ctype y) const
    {
      return coordinates.backward(x, y);
    }

    /**
     * @brief Interpolate at fractional grid indices, as returned by index().
     *
     * @param idx fractional grid indices; out-of-range values are clamped (or wrapped, on periodic axes)
     */
    NT KOKKOS_FUNCTION at(const device::array<typename Coordinates::ctype, 2> &idx) const
    {
      KOKKOS_IF_ON_DEVICE((return internal::linear_2d_at<NT, Coordinates>(device_data, sizes, idx);))
      KOKKOS_IF_ON_HOST((return internal::linear_2d_at<NT, Coordinates>(host_data, sizes, idx);))
    }

    /**
     * @brief Interpolate the data at the point (x, y).
     *
     * @param x first physical coordinate
     * @param y second physical coordinate
     */
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype x,
                                  const typename Coordinates::ctype y) const
    {
      return at(index(x, y));
    }

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

    /**
     * @brief Pointer to the host copy of the values (read-only).
     *
     * The storage order depends on the backend and need not be row-major; use operator[] for row-major access.
     */
    const NT *data() const { return host_data.data(); }

  private:
    const Coordinates coordinates;
    const device::array<size_t, 2> sizes;

    ViewType device_data;
    HostViewType host_data;

    // The single-precision copy, allocated only if Twin::exists. Behind one host-side handle, so it
    // adds 24 B to this object, which kernels naming the interpolator type still receive in full.
    struct Single {
      std::optional<Coordinates32> coordinates;
      ViewType32 device_data;
      HostViewType32 host_data;
    };
    Kokkos::View<Single, Kokkos::HostSpace> single;
  };
} // namespace DiFfRG
