#pragma once

// DiFfRG
#include "DiFfRG/common/utils.hh"
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
    /// The trilinear evaluation at fractional grid indices, shared by LinearInterpolator3D and its handle.
    template <typename NT, typename Coordinates, typename CT, typename View>
    KOKKOS_FORCEINLINE_FUNCTION NT linear_3d_at(const View &data, const device::array<size_t, 3> &sizes,
                                                const device::array<CT, 3> &idx)
    {
      // Clamped [i, i+1] stencil on bounded axes, wrapping [i, (i+1) % n] on periodic ones
      const auto sx = make_interpolation_stencil<is_periodic_axis_v<Coordinates, 0>>(idx[0], sizes[0]);
      const auto sy = make_interpolation_stencil<is_periodic_axis_v<Coordinates, 1>>(idx[1], sizes[1]);
      const auto sz = make_interpolation_stencil<is_periodic_axis_v<Coordinates, 2>>(idx[2], sizes[2]);

      const size_t x0 = sx.lower, x1 = sx.upper;
      const size_t y0 = sy.lower, y1 = sy.upper;
      const size_t z0 = sz.lower, z1 = sz.upper;

      const NT corner000 = data(x0, y0, z0);
      const NT corner010 = data(x0, y1, z0);
      const NT corner100 = data(x1, y0, z0);
      const NT corner110 = data(x1, y1, z0);
      const NT corner001 = data(x0, y0, z1);
      const NT corner011 = data(x0, y1, z1);
      const NT corner101 = data(x1, y0, z1);
      const NT corner111 = data(x1, y1, z1);

      const auto tx = sx.t;
      const auto ty = sy.t;
      const auto tz = sz.t;

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(
            tx,
            Kokkos::fma(ty, Kokkos::fma(tz, corner111, Kokkos::fma(-tz, corner110, corner110)),
                        (1 - ty) * Kokkos::fma(tz, corner101, Kokkos::fma(-tz, corner100, corner100))),
            (1 - tx) * Kokkos::fma(ty, Kokkos::fma(tz, corner011, Kokkos::fma(-tz, corner010, corner010)),
                                   (1 - ty) * Kokkos::fma(tz, corner001, Kokkos::fma(-tz, corner000, corner000))));
      else
        return corner000 * (1 - tx) * (1 - ty) * (1 - tz) + corner001 * (1 - tx) * (1 - ty) * tz +
               corner010 * (1 - tx) * ty * (1 - tz) + corner011 * (1 - tx) * ty * tz +
               corner100 * tx * (1 - ty) * (1 - tz) + corner101 * tx * (1 - ty) * tz +
               corner110 * tx * ty * (1 - tz) + corner111 * tx * ty * tz;
    }

    /// The stencil of one axis of periodic_cubic_3d_at: two points with linear weights on a bounded
    /// axis, four points with Catmull-Rom weights on a periodic one.
    template <bool periodic, typename CT> struct AxisStencil {
      static constexpr int n = periodic ? 4 : 2;
      size_t idx[n];
      CT w[n];
    };

    template <bool periodic, typename CT>
    KOKKOS_FORCEINLINE_FUNCTION AxisStencil<periodic, CT> make_axis_stencil(CT idx, const size_t n)
    {
      const auto s = make_interpolation_stencil<periodic>(idx, n);
      AxisStencil<periodic, CT> a;
      if constexpr (periodic) {
        // Cubic Hermite with central-difference node slopes (f[i+1] - f[i-1]) / 2: C^1 across nodes
        const CT t = s.t, t2 = t * t, t3 = t2 * t;
        a.idx[0] = (s.lower + n - 1) % n;
        a.idx[1] = s.lower;
        a.idx[2] = s.upper;
        a.idx[3] = (s.upper + 1) % n;
        a.w[0] = CT(0.5) * (-t + 2 * t2 - t3);
        a.w[1] = CT(0.5) * (2 - 5 * t2 + 3 * t3);
        a.w[2] = CT(0.5) * (t + 4 * t2 - 3 * t3);
        a.w[3] = CT(0.5) * (t3 - t2);
      } else {
        a.idx[0] = s.lower;
        a.idx[1] = s.upper;
        a.w[0] = 1 - s.t;
        a.w[1] = s.t;
      }
      return a;
    }

    /// Linear on bounded axes, Catmull-Rom on periodic ones, see PeriodicCubicInterpolator3D.
    template <typename NT, typename Coordinates, typename CT, typename View>
    KOKKOS_FORCEINLINE_FUNCTION NT periodic_cubic_3d_at(const View &data, const device::array<size_t, 3> &sizes,
                                                        const device::array<CT, 3> &idx)
    {
      const auto sx = make_axis_stencil<is_periodic_axis_v<Coordinates, 0>>(idx[0], sizes[0]);
      const auto sy = make_axis_stencil<is_periodic_axis_v<Coordinates, 1>>(idx[1], sizes[1]);
      const auto sz = make_axis_stencil<is_periodic_axis_v<Coordinates, 2>>(idx[2], sizes[2]);

      NT res = NT(0);
      for (int a = 0; a < sx.n; ++a)
        for (int b = 0; b < sy.n; ++b) {
          NT line = NT(0);
          for (int c = 0; c < sz.n; ++c)
            line += sz.w[c] * data(sx.idx[a], sy.idx[b], sz.idx[c]);
          res += (sx.w[a] * sy.w[b]) * line;
        }
      return res;
    }

    template <bool periodic_cubic, typename NT, typename Coordinates, typename CT, typename View>
    KOKKOS_FORCEINLINE_FUNCTION NT interpolate_3d_at(const View &data, const device::array<size_t, 3> &sizes,
                                                     const device::array<CT, 3> &idx)
    {
      if constexpr (periodic_cubic)
        return periodic_cubic_3d_at<NT, Coordinates>(data, sizes, idx);
      else
        return linear_3d_at<NT, Coordinates>(data, sizes, idx);
    }
  } // namespace internal

  /**
   * @brief What a kernel receives in place of a LinearInterpolator3D: read-only views of its device
   * and host buffers plus its coordinates, in one precision. See has_kernel_handle.
   */
  template <typename NT, typename Coordinates, typename Layout, bool periodic_cubic = false>
  class LinearInterpolator3DHandle
  {
  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 3;

    LinearInterpolator3DHandle(const NT *device_data, const NT *host_data, const device::array<size_t, 3> &sizes,
                               const Coordinates &coordinates)
        : device_data(device_data), host_data(host_data), sizes(sizes), coordinates(coordinates)
    {
    }

    device::array<ctype, 3> KOKKOS_FUNCTION index(const ctype &x, const ctype &y, const ctype &z) const
    {
      return coordinates.backward(x, y, z);
    }

    NT KOKKOS_FUNCTION at(const device::array<ctype, 3> &idx) const
    {
      KOKKOS_IF_ON_DEVICE(
          (return internal::interpolate_3d_at<periodic_cubic, NT, Coordinates>(View{device_data, sizes}, sizes, idx);))
      KOKKOS_IF_ON_HOST(
          (return internal::interpolate_3d_at<periodic_cubic, NT, Coordinates>(View{host_data, sizes}, sizes, idx);))
    }

    NT KOKKOS_FUNCTION operator()(const ctype &x, const ctype &y, const ctype &z) const { return at(index(x, y, z)); }

    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    using View = internal::RawView3D<NT, Layout>;
    const NT *device_data, *host_data;
    device::array<size_t, 3> sizes;
    Coordinates coordinates;
  };

  /**
   * @brief A linear interpolator for 3D data, callable from host AND device code.
   *
   * See LinearInterpolator1D for the host/device dispatch rationale. Like SplineInterpolator1D, a
   * double-precision interpolator keeps a single-precision copy for single-precision kernels.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   * @tparam periodic_cubic cubic instead of linear along the periodic axes, see PeriodicCubicInterpolator3D
   */
  template <typename NT, typename Coordinates, bool periodic_cubic = false> class LinearInterpolator3D
  {
    static_assert(Coordinates::dim == 3, "LinearInterpolator3D requires 3D coordinates");

    using ViewType = Kokkos::View<NT ***, GPU_memory, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using HostViewType = typename ViewType::host_mirror_type;

    static constexpr bool has_separate_device =
        !std::is_same_v<typename ViewType::memory_space, typename HostViewType::memory_space>;

    using Twin = SinglePrecisionTwin<NT, Coordinates>;
    using NT32 = typename Twin::value_type;
    using Coordinates32 = typename Twin::coordinates_type;
    using ViewType32 = Kokkos::View<NT32 ***, GPU_memory>;
    using HostViewType32 = typename ViewType32::host_mirror_type;

  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 3;

    /**
     * @brief Construct a LinearInterpolator3D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()),
          device_data("LinearInterpolator3D_data", coordinates.sizes()[0], coordinates.sizes()[1],
                      coordinates.sizes()[2]),
          host_data(Kokkos::create_mirror_view(device_data))
    {
      // the handles index the buffers by hand, which needs them unpadded
      if (device_data.span() != sizes[0] * sizes[1] * sizes[2] || host_data.span() != device_data.span())
        throw std::runtime_error("LinearInterpolator3D: padded data buffer");
      if constexpr (Twin::exists) {
        // SequentialHostInit: Single holds device views, which must not be created or destroyed
        // inside a host parallel region (the default for a view's element), or teardown deadlocks.
        single = Kokkos::View<Single, Kokkos::HostSpace>(
            Kokkos::view_alloc("LinearInterpolator3D_single", Kokkos::SequentialHostInit));
        auto &s = single();
        s.coordinates.emplace(coordinates);
        s.device_data = ViewType32("LinearInterpolator3D_data32", sizes[0], sizes[1], sizes[2]);
        s.host_data = Kokkos::create_mirror_view(s.device_data);
        if (s.device_data.span() != device_data.span() || s.host_data.span() != device_data.span())
          throw std::runtime_error("LinearInterpolator3D: padded data buffer");
      }
    }

    /// Shallow copy of BOTH views, valid in host and in device code. See LinearInterpolator1D.
    KOKKOS_DEFAULTED_FUNCTION LinearInterpolator3D(const LinearInterpolator3D &) = default;

    /**
     * @brief Replace the data, leaving host AND device current. The only mutator.
     *
     * `in_data` is row-major; see LinearInterpolator2D::update() for why the fill indexes the
     * mirror through operator() instead of copying flat, and LinearInterpolator1D::update() for
     * why the single trailing fence is not optional.
     */
    template <typename NT2> void update(const NT2 *in_data)
    {
      for (size_t i = 0; i < sizes[0]; ++i)
        for (size_t j = 0; j < sizes[1]; ++j)
          for (size_t k = 0; k < sizes[2]; ++k)
            host_data(i, j, k) = in_data[(i * sizes[1] + j) * sizes[2] + k];

      if constexpr (Twin::exists) {
        auto &s = single();
        for (size_t i = 0; i < sizes[0]; ++i)
          for (size_t j = 0; j < sizes[1]; ++j)
            for (size_t k = 0; k < sizes[2]; ++k)
              s.host_data(i, j, k) = static_cast<NT32>(host_data(i, j, k));
      }

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
        return LinearInterpolator3DHandle<NT32, Coordinates32, typename ViewType32::array_layout, periodic_cubic>(
            s.device_data.data(), s.host_data.data(), sizes, *s.coordinates);
      } else
        return LinearInterpolator3DHandle<NT, Coordinates, typename ViewType::array_layout, periodic_cubic>(
            device_data.data(), host_data.data(), sizes, coordinates);
    }

    /**
     * @brief Host-side element access, in the row-major order update() takes its input in.
     */
    NT operator[](size_t i) const
    {
      return host_data(i / (sizes[1] * sizes[2]), (i / sizes[2]) % sizes[1], i % sizes[2]);
    }

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
    device::array<ctype, 3> KOKKOS_FUNCTION index(const ctype &x, const ctype &y, const ctype &z) const
    {
      return coordinates.backward(x, y, z);
    }

    /**
     * @brief Interpolate at grid indices previously obtained from index().
     */
    NT KOKKOS_FUNCTION at(const device::array<ctype, 3> &idx) const
    {
      KOKKOS_IF_ON_DEVICE(
          (return internal::interpolate_3d_at<periodic_cubic, NT, Coordinates>(device_data, sizes, idx);))
      KOKKOS_IF_ON_HOST((return internal::interpolate_3d_at<periodic_cubic, NT, Coordinates>(host_data, sizes, idx);))
    }

    /**
     * @brief Interpolate the data at a given point.
     */
    NT KOKKOS_FUNCTION operator()(const ctype &x, const ctype &y, const ctype &z) const
    {
      return at(index(x, y, z));
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
     * NOTE the storage order is NOT the row-major order update() takes; see LinearInterpolator2D.
     */
    const NT *data() const { return host_data.data(); }

  private:
    const Coordinates coordinates;
    const device::array<size_t, 3> sizes;

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

  /**
   * @brief LinearInterpolator3D with a cubic instead of a linear stencil along its periodic axes.
   *
   * A piecewise-linear interpolant has a slope jump at every node, so a read at a node offset by
   * +-delta picks up a term in |delta|. Where delta is linear in a physical momentum -- e.g. a shape
   * angle approaching a soft-leg point that sits on a node -- this is a spurious non-analytic term,
   * and loop integrals that amplify the dressing (a tadpole ~ k^2) turn it into a |p| term in a
   * self-energy. The Catmull-Rom stencil takes the node slope from the central difference, so the
   * interpolant is C^1 and data even about a node have exactly zero slope there. It is exact for
   * quadratics in the grid index, i.e. O(h^3) for smooth data, and reads 4 instead of 2 points per
   * periodic axis. Bounded axes stay linear.
   */
  template <typename NT, typename Coordinates>
  using PeriodicCubicInterpolator3D = LinearInterpolator3D<NT, Coordinates, true>;
} // namespace DiFfRG
