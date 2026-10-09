#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/interpolation/interpolator_handle.hh>

// std
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace DiFfRG
{
  namespace internal
  {
    /// Map physical coordinates onto the grid indices of a stack, shared by SplineInterpolator1DStack and its handle.
    template <typename Coordinates>
    KOKKOS_FORCEINLINE_FUNCTION device::array<typename Coordinates::ctype, 2>
    spline_stack_index(const Coordinates &coordinates, const typename Coordinates::ctype s,
                       const typename Coordinates::ctype x)
    {
      using ctype = typename Coordinates::ctype;
      // backward() returns device::array for CoordinatePackND but std::tuple<Idx, NT> for the
      // combined finite-T systems; the structured binding normalises both onto what at() expects.
      auto [sidx, xidx] = coordinates.backward(s, x);
      return {{static_cast<ctype>(sidx), static_cast<ctype>(xidx)}};
    }

    /// The evaluation at grid indices, shared by SplineInterpolator1DStack and its handle.
    template <typename NT, typename CT, typename View>
    KOKKOS_FORCEINLINE_FUNCTION NT spline_stack_at(const View &values, const View &coeffs,
                                                   const device::array<size_t, 2> &sizes,
                                                   const device::array<CT, 2> &raw)
    {
      auto _sidx = raw[0];
      auto xidx = raw[1];
      // Clamp indices to the range [0, sizes[i] - 1]
      xidx = Kokkos::max(static_cast<decltype(xidx)>(0), Kokkos::min(xidx, static_cast<decltype(xidx)>(sizes[1] - 1)));
      _sidx =
          Kokkos::max(static_cast<decltype(_sidx)>(0), Kokkos::min(_sidx, static_cast<decltype(_sidx)>(sizes[0] - 1)));
      // for the x part
      const size_t lidx = Kokkos::min(size_t(Kokkos::floor(xidx)), sizes[1] - 2);
      const size_t uidx = lidx + 1;
      // t is the fractional part of the index
      const CT t = xidx - lidx;

      // the s part is linear between the two neighbouring rows, exact on each row
      const size_t s_lo = Kokkos::min(size_t(Kokkos::floor(_sidx)), sizes[0] - 1);
      const size_t s_hi = Kokkos::min(s_lo + 1, sizes[0] - 1);
      const CT ts = _sidx - CT(s_lo);

      const CT tm1 = t - 1;
      const auto row = [&](const size_t sidx) -> NT {
        const NT lower = values(sidx, lidx);
        const NT upper = values(sidx, uidx);
        const NT cl = coeffs(sidx, lidx);
        const NT cu = coeffs(sidx, uidx);
        const NT cubic = t * tm1 * ((t + 1) * cl - (t - 2) * cu);
        if constexpr (std::is_arithmetic_v<NT>)
          return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower)) + cubic; // linear + cubic
        else
          return t * upper + (1 - t) * lower + cubic; // linear + cubic
      };
      const NT f_lo = row(s_lo);
      if (s_hi == s_lo || ts == CT(0)) return f_lo;
      return f_lo + ts * (row(s_hi) - f_lo);
    }
  } // namespace internal

  /**
   * @brief What a kernel receives in place of a SplineInterpolator1DStack: read-only views of its
   * device and host buffers plus its coordinates, in one precision. See has_kernel_handle.
   */
  template <typename NT, typename Coordinates, typename Layout> class SplineInterpolator1DStackHandle
  {
  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 2;

    SplineInterpolator1DStackHandle(const NT *device_values, const NT *device_coeffs, const NT *host_values,
                                    const NT *host_coeffs, const device::array<size_t, 2> &sizes,
                                    const Coordinates &coordinates)
        : device_values(device_values), device_coeffs(device_coeffs), host_values(host_values),
          host_coeffs(host_coeffs), sizes(sizes), coordinates(coordinates)
    {
    }

    device::array<ctype, 2> KOKKOS_FUNCTION index(const ctype s, const ctype x) const
    {
      return internal::spline_stack_index(coordinates, s, x);
    }

    NT KOKKOS_FUNCTION at(const device::array<ctype, 2> &raw) const
    {
      KOKKOS_IF_ON_DEVICE(
          (return internal::spline_stack_at<NT>(View{device_values, sizes}, View{device_coeffs, sizes}, sizes, raw);))
      KOKKOS_IF_ON_HOST(
          (return internal::spline_stack_at<NT>(View{host_values, sizes}, View{host_coeffs, sizes}, sizes, raw);))
    }

    NT KOKKOS_FUNCTION operator()(const ctype s, const ctype x) const { return at(index(s, x)); }

    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    using View = internal::RawView2D<NT, Layout>;
    const NT *device_values, *device_coeffs, *host_values, *host_coeffs;
    device::array<size_t, 2> sizes;
    Coordinates coordinates;
  };

  /**
   * @brief A stack of 1D splines, callable from host AND device code.
   *
   * Linear in the stack axis, spline in the data axis. See LinearInterpolator1D for the
   * host/device dispatch rationale. Like SplineInterpolator1D, a double-precision interpolator keeps
   * a single-precision copy for single-precision kernels.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class SplineInterpolator1DStack
  {
    static_assert(Coordinates::dim == 2, "SplineInterpolator1DStack requires 2D coordinates");
    // The spline coefficients come from a non-cyclic tridiagonal solve with boundary conditions at the grid edges,
    // which cannot close a periodic axis. Use LinearInterpolator2D there instead.
    static_assert(!is_periodic_axis_v<Coordinates, 0> && !is_periodic_axis_v<Coordinates, 1>,
                  "SplineInterpolator1DStack does not support periodic coordinates; use LinearInterpolator2D.");

    // SoA layout: separate views for values and spline coefficients, for coalesced GPU access
    using ValueViewType = Kokkos::View<NT **, GPU_memory, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using CoeffViewType = ValueViewType;
    using HostValueViewType = typename ValueViewType::host_mirror_type;
    using HostCoeffViewType = typename CoeffViewType::host_mirror_type;

    static constexpr bool has_separate_device =
        !std::is_same_v<typename ValueViewType::memory_space, typename HostValueViewType::memory_space>;

    using Twin = SinglePrecisionTwin<NT, Coordinates>;
    using NT32 = typename Twin::value_type;
    using Coordinates32 = typename Twin::coordinates_type;
    using ValueViewType32 = Kokkos::View<NT32 **, GPU_memory>;
    using HostValueViewType32 = typename ValueViewType32::host_mirror_type;

  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 2;

    /**
     * @brief Construct a SplineInterpolator1DStack with zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    SplineInterpolator1DStack(const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()),
          device_values("SplineInterpolator1DStack_values", coordinates.sizes()[0], coordinates.sizes()[1]),
          device_coeffs("SplineInterpolator1DStack_coeffs", coordinates.sizes()[0], coordinates.sizes()[1]),
          host_values(Kokkos::create_mirror_view(device_values)),
          host_coeffs(Kokkos::create_mirror_view(device_coeffs))
    {
      // the handles index the buffers by hand, which needs them unpadded
      if (device_values.span() != sizes[0] * sizes[1] || host_values.span() != device_values.span())
        throw std::runtime_error("SplineInterpolator1DStack: padded data buffer");
      if constexpr (Twin::exists) {
        // SequentialHostInit: Single holds device views, which must not be created or destroyed
        // inside a host parallel region (the default for a view's element), or teardown deadlocks.
        single = Kokkos::View<Single, Kokkos::HostSpace>(
            Kokkos::view_alloc("SplineInterpolator1DStack_single", Kokkos::SequentialHostInit));
        auto &s = single();
        s.coordinates.emplace(coordinates);
        s.device_values = ValueViewType32("SplineInterpolator1DStack_values32", sizes[0], sizes[1]);
        s.device_coeffs = ValueViewType32("SplineInterpolator1DStack_coeffs32", sizes[0], sizes[1]);
        s.host_values = Kokkos::create_mirror_view(s.device_values);
        s.host_coeffs = Kokkos::create_mirror_view(s.device_coeffs);
        if (s.device_values.span() != device_values.span() || s.host_values.span() != device_values.span())
          throw std::runtime_error("SplineInterpolator1DStack: padded data buffer");
      }
    }

    /// Shallow copy of ALL views, valid in host and in device code. See LinearInterpolator1D.
    KOKKOS_DEFAULTED_FUNCTION SplineInterpolator1DStack(const SplineInterpolator1DStack &) = default;

    /**
     * @brief Replace the data, leaving host AND device current. The only mutator.
     *
     * `in_data` is row-major; the fill indexes the mirror through operator(), which keeps that
     * contract independent of the mirror's own layout. See LinearInterpolator1D::update() for why
     * the host fill is a plain loop and why the single trailing fence is not optional.
     */
    template <typename NT2>
    void update(const NT2 *in_data, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Copy values from input data (row-major)
      for (size_t i = 0; i < sizes[0]; ++i)
        for (size_t j = 0; j < sizes[1]; ++j)
          host_values(i, j) = in_data[i * sizes[1] + j];

      // Build the spline coefficients
      for (size_t i = 0; i < sizes[0]; ++i)
        build_y2(i, lower_y1, upper_y1);

      if constexpr (Twin::exists) {
        auto &s = single();
        for (size_t i = 0; i < sizes[0]; ++i)
          for (size_t j = 0; j < sizes[1]; ++j) {
            s.host_values(i, j) = static_cast<NT32>(host_values(i, j));
            s.host_coeffs(i, j) = static_cast<NT32>(host_coeffs(i, j));
          }
      }

      if constexpr (has_separate_device) {
        typename ValueViewType::execution_space exec;
        Kokkos::deep_copy(exec, device_values, host_values);
        Kokkos::deep_copy(exec, device_coeffs, host_coeffs);
        if constexpr (Twin::exists) {
          auto &s = single();
          Kokkos::deep_copy(exec, s.device_values, s.host_values);
          Kokkos::deep_copy(exec, s.device_coeffs, s.host_coeffs);
        }
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
        return SplineInterpolator1DStackHandle<NT32, Coordinates32, typename ValueViewType32::array_layout>(
            s.device_values.data(), s.device_coeffs.data(), s.host_values.data(), s.host_coeffs.data(), sizes,
            *s.coordinates);
      } else
        return SplineInterpolator1DStackHandle<NT, Coordinates, typename ValueViewType::array_layout>(
            device_values.data(), device_coeffs.data(), host_values.data(), host_coeffs.data(), sizes, coordinates);
    }

    /**
     * @brief Host-side element access, in the row-major order update() takes its input in.
     */
    NT operator[](size_t i) const { return host_values(i / sizes[1], i % sizes[1]); }

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
    index(const typename Coordinates::ctype s, const typename Coordinates::ctype x) const
    {
      return internal::spline_stack_index(coordinates, s, x);
    }

    /**
     * @brief Interpolate at grid indices previously obtained from index().
     */
    NT KOKKOS_FUNCTION at(const device::array<typename Coordinates::ctype, 2> &raw) const
    {
      KOKKOS_IF_ON_DEVICE((return internal::spline_stack_at<NT>(device_values, device_coeffs, sizes, raw);))
      KOKKOS_IF_ON_HOST((return internal::spline_stack_at<NT>(host_values, host_coeffs, sizes, raw);))
    }

    /**
     * @brief Interpolate the data at a given point.
     */
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype s,
                                  const typename Coordinates::ctype x) const
    {
      return at(index(s, x));
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
    const NT *data() const { return host_values.data(); }

  private:
    const Coordinates coordinates;
    const device::array<size_t, 2> sizes;

    ValueViewType device_values;
    CoeffViewType device_coeffs;
    HostValueViewType host_values;
    HostCoeffViewType host_coeffs;

    // The single-precision copy, allocated only if Twin::exists. Behind one host-side handle, so it
    // adds 24 B to this object, which kernels naming the interpolator type still receive in full.
    struct Single {
      std::optional<Coordinates32> coordinates;
      ValueViewType32 device_values, device_coeffs;
      HostValueViewType32 host_values, host_coeffs;
    };
    Kokkos::View<Single, Kokkos::HostSpace> single;

    void build_y2(const size_t sidx, const ctype lower_y1, const ctype upper_y1)
    {
      const auto &size = sizes[1];

      NT p, qn, sig, un;
      std::vector<NT> u(size - 1);

      if (!std::isfinite(lower_y1) || lower_y1 >= std::numeric_limits<ctype>::max() / 2)
        host_coeffs(sidx, 0) = u[0] = 0.0;
      else {
        host_coeffs(sidx, 0) = -0.5;
        u[0] = 3.0 * ((host_values(sidx, 1) - host_values(sidx, 0)) - lower_y1);
      }
      for (size_t i = 1; i < size - 1; i++) {
        sig = 0.5;
        p = sig * host_coeffs(sidx, i - 1) + 2.0;
        host_coeffs(sidx, i) = (sig - 1.0) / p;
        u[i] = (host_values(sidx, i + 1) - host_values(sidx, i)) - (host_values(sidx, i) - host_values(sidx, i - 1));
        u[i] = (6.0 * u[i] / 2. - sig * u[i - 1]) / p;
      }
      if (!std::isfinite(upper_y1) || upper_y1 >= std::numeric_limits<ctype>::max() / 2)
        qn = un = 0.0;
      else {
        qn = 0.5;
        un = 3.0 * (upper_y1 - (host_values(sidx, size - 1) - host_values(sidx, size - 2)));
      }
      host_coeffs(sidx, size - 1) = (un - qn * u[size - 2]) / (qn * host_coeffs(sidx, size - 2) + 1);
      for (int k = size - 2; k >= 0; k--)
        host_coeffs(sidx, k) = host_coeffs(sidx, k) * host_coeffs(sidx, k + 1) + u[k];

      // Precompute division by 6 so operator() avoids per-call divides
      for (size_t k = 0; k < size; ++k)
        host_coeffs(sidx, k) /= (ctype)6;
    }
  };
} // namespace DiFfRG
