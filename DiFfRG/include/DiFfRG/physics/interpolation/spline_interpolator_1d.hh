#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/interpolation/interpolator_handle.hh>

// std
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace DiFfRG
{
  namespace internal
  {
    /// The spline evaluation at a fractional grid index, shared by SplineInterpolator1D and its handle.
    template <typename NT, typename CT, typename View>
    KOKKOS_FORCEINLINE_FUNCTION NT spline_at(const View &values, const View &coeffs, const size_t size, const CT raw_idx)
    {
      // Clamp the index to the range [0, size - 1]
      const CT idx = Kokkos::max(static_cast<CT>(0), Kokkos::min(raw_idx, static_cast<CT>(size - 1)));
      const size_t lidx = Kokkos::min(size_t(Kokkos::floor(idx)), size - 2);
      const size_t uidx = lidx + 1;
      // t is the fractional part of the index
      const CT t = idx - lidx;

      const NT lower = values(lidx);
      const NT upper = values(uidx);
      const NT cl = coeffs(lidx);
      const NT cu = coeffs(uidx);

      const CT tm1 = t - 1;
      const NT cubic = t * tm1 * ((t + 1) * cl - (t - 2) * cu);

      if constexpr (std::is_arithmetic_v<NT>)
        return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower)) + cubic; // linear + cubic
      else
        return t * upper + (1 - t) * lower + cubic; // linear + cubic
    }
  } // namespace internal

  /**
   * @brief What a kernel receives in place of a SplineInterpolator1D: read-only views of its device
   * and host buffers plus its coordinates, in one precision. See has_kernel_handle.
   */
  template <typename NT, typename Coordinates> class SplineInterpolator1DHandle
  {
  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 1;

    SplineInterpolator1DHandle(const NT *device_values, const NT *device_coeffs, const NT *host_values,
                               const NT *host_coeffs, const size_t size, const Coordinates &coordinates)
        : device_values(device_values), device_coeffs(device_coeffs), host_values(host_values),
          host_coeffs(host_coeffs), size(size), coordinates(coordinates)
    {
    }

    ctype KOKKOS_FUNCTION index(const ctype x) const { return coordinates.backward(x); }

    NT KOKKOS_FUNCTION at(const ctype raw_idx) const
    {
      KOKKOS_IF_ON_DEVICE(
          (return internal::spline_at<NT>(View{device_values}, View{device_coeffs}, size, raw_idx);))
      KOKKOS_IF_ON_HOST((return internal::spline_at<NT>(View{host_values}, View{host_coeffs}, size, raw_idx);))
    }

    NT KOKKOS_FUNCTION operator()(const ctype x) const { return at(index(x)); }

    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    using View = internal::RawView1D<NT>;
    const NT *device_values, *device_coeffs, *host_values, *host_coeffs;
    size_t size;
    Coordinates coordinates;
  };

  /**
   * @brief A spline interpolator for 1D data, callable from host AND device code.
   *
   * See LinearInterpolator1D for the host/device dispatch rationale.
   *
   * A double-precision interpolator also keeps a single-precision copy of its data and coordinates,
   * refreshed by update() from the double data (so the spline itself is always solved in double).
   * A single-precision kernel that takes handles (see has_kernel_handle) reads that copy.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class SplineInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "SplineInterpolator1D requires 1D coordinates");
    // The spline coefficients come from a non-cyclic tridiagonal solve with boundary conditions at the grid edges,
    // which cannot close a periodic axis. Use LinearInterpolator1D there instead.
    static_assert(!is_periodic_coordinate_v<Coordinates>,
                  "SplineInterpolator1D does not support periodic coordinates; use LinearInterpolator1D.");

    // SoA layout: separate views for values and spline coefficients, for coalesced GPU access
    using ValueViewType = Kokkos::View<NT *, GPU_memory, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
    using CoeffViewType = ValueViewType;
    using HostValueViewType = typename ValueViewType::host_mirror_type;
    using HostCoeffViewType = typename CoeffViewType::host_mirror_type;

    static constexpr bool has_separate_device =
        !std::is_same_v<typename ValueViewType::memory_space, typename HostValueViewType::memory_space>;

    using Twin = SinglePrecisionTwin<NT, Coordinates>;
    using NT32 = typename Twin::value_type;
    using Coordinates32 = typename Twin::coordinates_type;
    using ValueViewType32 = Kokkos::View<NT32 *, GPU_memory>;
    using HostValueViewType32 = typename ValueViewType32::host_mirror_type;

  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 1;

    /**
     * @brief Construct a SplineInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    SplineInterpolator1D(const Coordinates &coordinates)
        : coordinates(coordinates), size(coordinates.size()),
          device_values("SplineInterpolator1D_values", coordinates.size()),
          device_coeffs("SplineInterpolator1D_coeffs", coordinates.size()),
          host_values(Kokkos::create_mirror_view(device_values)),
          host_coeffs(Kokkos::create_mirror_view(device_coeffs))
    {
      if constexpr (Twin::exists) {
        // SequentialHostInit: Single holds device views, which must not be created or destroyed
        // inside a host parallel region (the default for a view's element), or teardown deadlocks.
        single = Kokkos::View<Single, Kokkos::HostSpace>(
            Kokkos::view_alloc("SplineInterpolator1D_single", Kokkos::SequentialHostInit));
        auto &s = single();
        s.coordinates.emplace(coordinates);
        s.device_values = ValueViewType32("SplineInterpolator1D_values32", size);
        s.device_coeffs = ValueViewType32("SplineInterpolator1D_coeffs32", size);
        s.host_values = Kokkos::create_mirror_view(s.device_values);
        s.host_coeffs = Kokkos::create_mirror_view(s.device_coeffs);
      }
    }

    /// Shallow copy of ALL views, valid in host and in device code. See LinearInterpolator1D.
    KOKKOS_DEFAULTED_FUNCTION SplineInterpolator1D(const SplineInterpolator1D &) = default;

    /**
     * @brief Replace the data, leaving host AND device current. The only mutator.
     *
     * See LinearInterpolator1D::update() for why the host fill is a plain copy and why the single
     * trailing fence is not optional.
     */
    template <typename NT2>
    void update(const NT2 *in_data, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Copy values from input data
      for (size_t i = 0; i < size; ++i)
        host_values(i) = in_data[i];

      // Build the spline coefficients
      build_y2(lower_y1, upper_y1);

      if constexpr (Twin::exists) {
        auto &s = single();
        for (size_t i = 0; i < size; ++i) {
          s.host_values(i) = static_cast<NT32>(host_values(i));
          s.host_coeffs(i) = static_cast<NT32>(host_coeffs(i));
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
        return SplineInterpolator1DHandle<NT32, Coordinates32>(s.device_values.data(), s.device_coeffs.data(),
                                                               s.host_values.data(), s.host_coeffs.data(), size,
                                                               *s.coordinates);
      } else
        return SplineInterpolator1DHandle<NT, Coordinates>(device_values.data(), device_coeffs.data(),
                                                           host_values.data(), host_coeffs.data(), size, coordinates);
    }

    /**
     * @brief Host-side element access. Always valid, including on a copy.
     */
    NT operator[](size_t i) const { return host_values(i); }

    /**
     * @brief Map a physical coordinate onto the (clamped) grid index.
     *
     * Split out of operator() because it is the expensive half: for a logarithmic axis
     * Coordinates::backward is a fp64 log1p, which costs ~200 fp64 instructions on current NVIDIA
     * parts, against ~10 for the spline evaluation itself. A generated kernel that evaluates many
     * dressings at the SAME momentum pays that log1p once per dressing, because each interpolator
     * owns its own `coordinates` members and the compiler cannot prove they are equal across
     * objects -- so it cannot CSE the transform. Hoisting index() lets the caller pay it once.
     *
     * The returned index is only meaningful for interpolators sharing this coordinate system.
     */
    ctype KOKKOS_FUNCTION index(const typename Coordinates::ctype x) const { return coordinates.backward(x); }

    /**
     * @brief Interpolate at a grid index previously obtained from index().
     *
     * Clamping lives here, not in index(), so that index() depends ONLY on the coordinate system.
     * That is exactly the precondition under which a caller may share one index across several
     * interpolators; folding a size-dependent clamp into it would silently break sharing between
     * interpolators of different length.
     */
    NT KOKKOS_FUNCTION at(const ctype raw_idx) const
    {
      KOKKOS_IF_ON_DEVICE((return internal::spline_at<NT>(device_values, device_coeffs, size, raw_idx);))
      KOKKOS_IF_ON_HOST((return internal::spline_at<NT>(host_values, host_coeffs, size, raw_idx);))
    }

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    NT KOKKOS_FUNCTION operator()(const typename Coordinates::ctype x) const { return at(index(x)); }

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

    /**
     * @brief Read-only handle to the host values.
     *
     * Deliberately const: a host-side write would leave both the device buffers and the spline
     * coefficients stale, and there is no public way to push it. Route mutations through update().
     */
    const NT *data() const { return host_values.data(); }

  private:
    const Coordinates coordinates;
    const size_t size;

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

    void build_y2(const ctype lower_y1, const ctype upper_y1)
    {
      NT p, qn, sig, un;
      std::vector<NT> u(size - 1);

      if (!std::isfinite(lower_y1) || lower_y1 >= std::numeric_limits<ctype>::max() / 2)
        host_coeffs(0) = u[0] = 0.0;
      else {
        host_coeffs(0) = -0.5;
        u[0] = 3.0 * ((host_values(1) - host_values(0)) - lower_y1);
      }
      for (size_t i = 1; i < size - 1; i++) {
        sig = 0.5;
        p = sig * host_coeffs(i - 1) + 2.0;
        host_coeffs(i) = (sig - 1.0) / p;
        u[i] = (host_values(i + 1) - host_values(i)) - (host_values(i) - host_values(i - 1));
        u[i] = (6.0 * u[i] / 2. - sig * u[i - 1]) / p;
      }
      if (!std::isfinite(upper_y1) || upper_y1 >= std::numeric_limits<ctype>::max() / 2)
        qn = un = 0.0;
      else {
        qn = 0.5;
        un = 3.0 * (upper_y1 - (host_values(size - 1) - host_values(size - 2)));
      }
      host_coeffs(size - 1) = (un - qn * u[size - 2]) / (qn * host_coeffs(size - 2) + 1);
      for (int k = size - 2; k >= 0; k--)
        host_coeffs(k) = host_coeffs(k) * host_coeffs(k + 1) + u[k];

      // Precompute division by 6 so operator() avoids per-call divides
      for (size_t k = 0; k < size; ++k)
        host_coeffs(k) /= (ctype)6;
    }
  };
} // namespace DiFfRG
