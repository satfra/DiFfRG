#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/interpolation/interpolation_stencil.hh>
#include <DiFfRG/physics/interpolation/interpolator_handle.hh>

// std
#include <cstring>
#include <optional>

namespace DiFfRG
{
  namespace internal
  {
    /// The linear evaluation at a fractional grid index, shared by LinearInterpolator1D and its handle.
    template <typename NT, typename Coordinates, typename CT, typename View>
    KOKKOS_FORCEINLINE_FUNCTION NT linear_1d_at(const View &data, const size_t size, const CT idx)
    {
      // Resolve the stencil: clamped [i, i+1] for a bounded axis, wrapping [i, (i+1) % size] for a periodic one
      const auto stencil = make_interpolation_stencil<is_periodic_coordinate_v<Coordinates>>(idx, size);
      const auto t = stencil.t;
      const NT lower = data(stencil.lower);
      const NT upper = data(stencil.upper);
      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower));
      else
        return t * upper + (1 - t) * lower;
    }
  } // namespace internal

  /**
   * @brief Read-only view of a LinearInterpolator1D, as passed to integration kernels.
   *
   * Obtained from LinearInterpolator1D::handle(); offers the same index(), at() and operator(). See has_kernel_handle.
   */
  template <typename NT, typename Coordinates> class LinearInterpolator1DHandle
  {
  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 1;

    LinearInterpolator1DHandle(const NT *device_data, const NT *host_data, const size_t size,
                               const Coordinates &coordinates)
        : device_data(device_data), host_data(host_data), size(size), coordinates(coordinates)
    {
    }

    ctype KOKKOS_FUNCTION index(const ctype x) const { return coordinates.backward(x); }

    NT KOKKOS_FUNCTION at(const ctype idx) const
    {
      KOKKOS_IF_ON_DEVICE((return internal::linear_1d_at<NT, Coordinates>(View{device_data}, size, idx);))
      KOKKOS_IF_ON_HOST((return internal::linear_1d_at<NT, Coordinates>(View{host_data}, size, idx);))
    }

    NT KOKKOS_FUNCTION operator()(const ctype x) const { return at(index(x)); }

    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    using View = internal::RawView1D<NT>;
    const NT *device_data, *host_data;
    size_t size;
    Coordinates coordinates;
  };

  /**
   * @brief Linear interpolation of data given on a 1D grid, usable in host and device code.
   *
   * With \f$i(x)\f$ the fractional grid index of \f$x\f$ (Coordinates::backward), \f$i_0 = \lfloor i \rfloor\f$
   * and \f$t = i - i_0\f$,
   * \f[ f(x) = (1-t)\, f_{i_0} + t\, f_{i_0+1}\,. \f]
   * The interpolation is linear in the grid index, i.e. linear in \f$x\f$ only for linear coordinates.
   *
   * Outside the grid the boundary value is returned (constant extrapolation). A periodic coordinate system wraps
   * around instead. The grid needs at least 2 points (bounded axis).
   *
   * The data is zero after construction; set it with update(). Host and device copies are kept in sync, so the same
   * object can be evaluated on either side. A double-precision interpolator also keeps a single-precision copy, used
   * by single-precision kernels (see handle()).
   *
   * @tparam NT value type of the data (real, complex or autodiff)
   * @tparam Coordinates 1D coordinate system of the grid, e.g. LinearCoordinates1D or LogarithmicCoordinates1D
   */
  // Host/device dispatch in at() uses KOKKOS_IF_ON_DEVICE/HOST, a per-compilation-pass selection. If it is ever
  // wrong, Kokkos aborts with "attempt to access inaccessible memory space" (see host_device_dispatch.cc in tests).
  template <typename NT, typename Coordinates> class LinearInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "LinearInterpolator1D requires 1D coordinates");

    using ViewType = Kokkos::View<NT *, GPU_memory, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::host_mirror_type;

    // Without a distinct device space create_mirror_view aliases the source: the two members are
    // then one allocation, and the upload must not be compiled at all so a host-only build pays no
    // fence.
    static constexpr bool has_separate_device =
        !std::is_same_v<typename ViewType::memory_space, typename HostViewType::memory_space>;

    using Twin = SinglePrecisionTwin<NT, Coordinates>;
    using NT32 = typename Twin::value_type;
    using Coordinates32 = typename Twin::coordinates_type;
    using ViewType32 = Kokkos::View<NT32 *, GPU_memory>;
    using HostViewType32 = typename ViewType32::host_mirror_type;

  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 1;

    /**
     * @brief Allocate zeroed data on the grid of `coordinates`.
     *
     * @param coordinates coordinate system of the data; its size() is the number of grid points
     */
    LinearInterpolator1D(const Coordinates &coordinates)
        : coordinates(coordinates), size(coordinates.size()),
          device_data("LinearInterpolator1D_data", coordinates.size()),
          host_data(Kokkos::create_mirror_view(device_data))
    {
      if constexpr (Twin::exists) {
        // SequentialHostInit: Single holds device views, which must not be created or destroyed
        // inside a host parallel region (the default for a view's element), or teardown deadlocks.
        single = Kokkos::View<Single, Kokkos::HostSpace>(
            Kokkos::view_alloc("LinearInterpolator1D_single", Kokkos::SequentialHostInit));
        auto &s = single();
        s.coordinates.emplace(coordinates);
        s.device_data = ViewType32("LinearInterpolator1D_data32", size);
        s.host_data = Kokkos::create_mirror_view(s.device_data);
      }
    }

    /**
     * @brief Shallow copy: the copy shares the data, so later update() calls are seen by both.
     */
    // Defaulted so that capture into a KOKKOS_LAMBDA stays well-formed; as in Kokkos::DualView, the host view is
    // copied into device closures but never dereferenced there.
    KOKKOS_DEFAULTED_FUNCTION LinearInterpolator1D(const LinearInterpolator1D &) = default;

    /**
     * @brief Replace the data. Host and device copies are both current when this returns.
     *
     * @param in_data values at the grid points, `get_coordinates().size()` entries, converted to NT
     */
    template <typename NT2> void update(const NT2 *in_data)
    {
      // Host fill by memcpy/loop (complete on return), then one H2D copy on a named execution space instance: the
      // space-less deep_copy would fence every backend globally. The trailing fence lets the next update() refill
      // host_data without racing the copy.
      if constexpr (std::is_same_v<NT, NT2> && std::is_trivially_copyable_v<NT>)
        std::memcpy(host_data.data(), in_data, size * sizeof(NT));
      else
        for (size_t i = 0; i < size; ++i)
          host_data(i) = in_data[i];

      if constexpr (Twin::exists)
        for (size_t i = 0; i < size; ++i)
          single().host_data(i) = static_cast<NT32>(host_data(i));

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
        return LinearInterpolator1DHandle<NT32, Coordinates32>(s.device_data.data(), s.host_data.data(), size,
                                                               *s.coordinates);
      } else
        return LinearInterpolator1DHandle<NT, Coordinates>(device_data.data(), host_data.data(), size, coordinates);
    }

    /**
     * @brief Value at grid point i (host only).
     */
    NT operator[](size_t i) const { return host_data(i); }

    /**
     * @brief Fractional grid index of the point x, i.e. Coordinates::backward(x).
     *
     * `f(x) == f.at(f.index(x))`. Several interpolators on the same coordinate system can share one index() call,
     * which saves the (possibly expensive) coordinate transform.
     *
     * @param x physical coordinate
     */
    typename Coordinates::ctype KOKKOS_FUNCTION index(const typename Coordinates::ctype x) const
    {
      return coordinates.backward(x);
    }

    /**
     * @brief Interpolate at a fractional grid index, as returned by index().
     *
     * @param idx fractional grid index; values outside [0, size-1] are clamped (or wrapped, if periodic)
     */
    NT KOKKOS_FUNCTION at(const typename Coordinates::ctype idx) const
    {
      KOKKOS_IF_ON_DEVICE((return internal::linear_1d_at<NT, Coordinates>(device_data, size, idx);))
      KOKKOS_IF_ON_HOST((return internal::linear_1d_at<NT, Coordinates>(host_data, size, idx);))
    }

    /**
     * @brief Interpolate the data at the point x.
     *
     * @param x physical coordinate
     */
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype x) const { return at(index(x)); }

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

    /**
     * @brief Pointer to the host copy of the values (read-only; change the data with update()).
     */
    const NT *data() const { return host_data.data(); }

  private:
    const Coordinates coordinates;
    const size_t size;

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
