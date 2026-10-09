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
   * @brief What a kernel receives in place of a LinearInterpolator2D: read-only views of its device
   * and host buffers plus its coordinates, in one precision. See has_kernel_handle.
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
   * @brief A linear interpolator for 2D data, callable from host AND device code.
   *
   * See LinearInterpolator1D for the host/device dispatch rationale.
   *
   * Like SplineInterpolator1D, a double-precision interpolator keeps a single-precision copy for
   * single-precision kernels.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
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
     * @brief Construct a LinearInterpolator2D with internal, zeroed data and a coordinate system.
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

    /// Shallow copy of BOTH views, valid in host and in device code. See LinearInterpolator1D.
    KOKKOS_DEFAULTED_FUNCTION LinearInterpolator2D(const LinearInterpolator2D &) = default;

    /**
     * @brief Replace the data, leaving host AND device current. The only mutator.
     *
     * `in_data` is row-major. The fill indexes the mirror through operator() rather than
     * memcpy'ing into it, which makes that contract explicit and independent of the mirror's own
     * layout -- the device view is pinned to GPU_memory, whose default layout is LayoutLeft, so a
     * flat copy would transpose the data.
     *
     * See LinearInterpolator1D::update() for why the host fill is a plain loop and why the single
     * trailing fence is not optional.
     */
    template <typename NT2> void update(const NT2 *in_data)
    {
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
     * @brief The compact view of this interpolator a kernel computing in CT receives, see
     * has_kernel_handle: the single-precision copy for a float kernel, the data itself otherwise.
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
     * @brief Host-side element access, in the row-major order update() takes its input in.
     */
    NT operator[](size_t i) const { return host_data(i / sizes[1], i % sizes[1]); }

    /**
     * @brief Map physical coordinates onto grid indices.
     *
     * Split out of operator() because it is the expensive half: Coordinates::backward is a fp64
     * log/log1p per logarithmic axis, ~200 fp64 instructions each on current NVIDIA parts against
     * ~10 for the interpolation. A generated kernel evaluating several dressings at the SAME point
     * otherwise pays it once per dressing -- the compiler cannot CSE it, because each interpolator
     * owns its own `coordinates` members and cannot be proven to agree with another's.
     *
     * Depends only on the coordinate system, so the result may be shared across interpolators that
     * share one. Clamping and stencil resolution stay in at(), since they are size dependent and
     * would otherwise make an index untransferable between interpolators of different extent.
     *
     * The return type is spelled out rather than deduced: nvcc loses `decltype(var)` when the
     * initializer has a deduced return type inside a class-template member, and the consumer stores
     * this in a `const auto`.
     */
    device::array<typename Coordinates::ctype, 2> KOKKOS_FUNCTION
    index(const typename Coordinates::ctype x, const typename Coordinates::ctype y) const
    {
      return coordinates.backward(x, y);
    }

    /**
     * @brief Interpolate at grid indices previously obtained from index().
     */
    NT KOKKOS_FUNCTION at(const device::array<typename Coordinates::ctype, 2> &idx) const
    {
      KOKKOS_IF_ON_DEVICE((return internal::linear_2d_at<NT, Coordinates>(device_data, sizes, idx);))
      KOKKOS_IF_ON_HOST((return internal::linear_2d_at<NT, Coordinates>(host_data, sizes, idx);))
    }

    /**
     * @brief Interpolate the data at a given point.
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
     * @brief Read-only handle to the host values, in the mirror's storage order.
     *
     * NOTE the storage order is NOT the row-major order update() takes: the device view is pinned
     * to GPU_memory, hence LayoutLeft. Use operator[] for row-major access.
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
