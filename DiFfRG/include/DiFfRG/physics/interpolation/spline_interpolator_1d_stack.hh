#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/interpolation/data_epoch.hh>
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

    /**
     * @brief The evaluation at grid indices, shared by SplineInterpolator1DStack and its handle.
     *
     * A tensor-product cubic spline. Along x each row is a cubic spline in the grid index (values, coeffs = f''/6).
     * Along s the same holds with coefficients s2 = (d^2 f / ds^2) / 6 at every grid point, which are themselves
     * splined in x (s2x); since a spline is linear in its data, the spline of s2 in x IS the s-coefficient of the
     * rows evaluated at x, so the result is exact on every grid point and twice continuously differentiable in both
     * directions. Four rows are read instead of two.
     */
    template <typename NT, typename CT, typename View>
    KOKKOS_FORCEINLINE_FUNCTION NT spline_stack_at(const View &values, const View &coeffs, const View &s2,
                                                   const View &s2x, const device::array<size_t, 2> &sizes,
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

      const size_t s_lo = Kokkos::min(size_t(Kokkos::floor(_sidx)), sizes[0] - 1);
      const size_t s_hi = Kokkos::min(s_lo + 1, sizes[0] - 1);
      const CT ts = _sidx - CT(s_lo);

      const CT tm1 = t - 1;
      // One row of a spline in x: the values (y, c) or the s-coefficients (s2, s2x).
      const auto row = [&](const View &y, const View &c, const size_t sidx) -> NT {
        const NT lower = y(sidx, lidx);
        const NT upper = y(sidx, uidx);
        const NT cl = c(sidx, lidx);
        const NT cu = c(sidx, uidx);
        const NT cubic = t * tm1 * ((t + 1) * cu - (t - 2) * cl);
        if constexpr (std::is_arithmetic_v<NT>)
          return Kokkos::fma(t, upper, Kokkos::fma(-t, lower, lower)) + cubic; // linear + cubic
        else
          return t * upper + (1 - t) * lower + cubic; // linear + cubic
      };
      const NT f_lo = row(values, coeffs, s_lo);
      if (s_hi == s_lo || ts == CT(0)) return f_lo;
      const NT f_hi = row(values, coeffs, s_hi);
      const NT c_lo = row(s2, s2x, s_lo);
      const NT c_hi = row(s2, s2x, s_hi);
      return f_lo + ts * (f_hi - f_lo) + ts * (ts - 1) * ((ts + 1) * c_hi - (ts - 2) * c_lo);
    }
  } // namespace internal

  /**
   * @brief Read-only view of a SplineInterpolator1DStack, as passed to integration kernels.
   *
   * Obtained from SplineInterpolator1DStack::handle(); offers the same index(), at() and operator(). See
   * has_kernel_handle.
   */
  template <typename NT, typename Coordinates, typename Layout> class SplineInterpolator1DStackHandle
  {
  public:
    using ctype = typename Coordinates::ctype;
    using value_type = NT;
    static constexpr size_t dim = 2;

    /// Pointers to {values, coeffs, s2, s2x} on the device and on the host; see internal::spline_stack_at.
    SplineInterpolator1DStackHandle(const device::array<const NT *, 4> &device_data,
                                    const device::array<const NT *, 4> &host_data, const device::array<size_t, 2> &sizes,
                                    const Coordinates &coordinates)
        : device_data(device_data), host_data(host_data), sizes(sizes), coordinates(coordinates)
    {
    }

    device::array<ctype, 2> KOKKOS_FUNCTION index(const ctype s, const ctype x) const
    {
      return internal::spline_stack_index(coordinates, s, x);
    }

    NT KOKKOS_FUNCTION at(const device::array<ctype, 2> &raw) const
    {
      KOKKOS_IF_ON_DEVICE((return internal::spline_stack_at<NT>(View{device_data[0], sizes}, View{device_data[1], sizes},
                                                                View{device_data[2], sizes}, View{device_data[3], sizes},
                                                                sizes, raw);))
      KOKKOS_IF_ON_HOST((return internal::spline_stack_at<NT>(View{host_data[0], sizes}, View{host_data[1], sizes},
                                                              View{host_data[2], sizes}, View{host_data[3], sizes},
                                                              sizes, raw);))
    }

    NT KOKKOS_FUNCTION operator()(const ctype s, const ctype x) const { return at(index(s, x)); }

    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    using View = internal::RawView2D<NT, Layout>;
    device::array<const NT *, 4> device_data, host_data;
    device::array<size_t, 2> sizes;
    Coordinates coordinates;
  };

  /**
   * @brief A stack of 1D cubic splines \f$f(s, x)\f$, usable in host and device code.
   *
   * Axis 0 is the stack variable \f$s\f$ (e.g. a Matsubara frequency), axis 1 the spline variable \f$x\f$ (e.g. a
   * spatial momentum); calls take them in that order, `f(s, x)`. The data are a tensor-product cubic spline in the
   * grid indices of both: each row of fixed \f$s\f$ as in SplineInterpolator1D, and the rows joined by a cubic
   * spline in \f$s\f$ (natural ends; for a finite-temperature stack that starts at the zero frequency and folds
   * \f$s \to |s|\f$, zero slope at that row, so the spline is even and smooth through it). Exact on every grid point,
   * twice continuously differentiable in both directions. A frequency sum evaluated between the Matsubara rows, as
   * a Gauss rule for sums or a frequency integral does, thus sees a smooth summand; a linear join put a kink at every
   * row, and an error into such sums that more nodes do not remove.
   *
   * Outside the grid the boundary values are used; how an \f$s\f$ beyond the stack is handled is up to the
   * coordinate system (the finite-temperature systems, e.g. BosonicCoordinates1DFiniteT, fold \f$s \to |s|\f$ when the
   * stack starts at 0 and move the excess frequency into the momentum). The spline axis needs at least 2 points;
   * periodic axes are not supported.
   *
   * The data is zero after construction; set it with update(). Host and device copies are kept in sync. A
   * double-precision interpolator also keeps a single-precision copy for single-precision kernels (see handle()).
   *
   * @tparam NT value type of the data (real, complex or autodiff)
   * @tparam Coordinates 2D coordinate system: a CoordinatePackND of two 1D systems or a finite-temperature system
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
     * @brief Allocate zeroed data on the grid of `coordinates`.
     *
     * @param coordinates coordinate system of the data
     */
    SplineInterpolator1DStack(const Coordinates &coordinates)
        : coordinates(coordinates), sizes(coordinates.sizes()),
          device_values("SplineInterpolator1DStack_values", coordinates.sizes()[0], coordinates.sizes()[1]),
          device_coeffs("SplineInterpolator1DStack_coeffs", coordinates.sizes()[0], coordinates.sizes()[1]),
          device_s2("SplineInterpolator1DStack_s2", coordinates.sizes()[0], coordinates.sizes()[1]),
          device_s2x("SplineInterpolator1DStack_s2x", coordinates.sizes()[0], coordinates.sizes()[1]),
          host_values(Kokkos::create_mirror_view(device_values)),
          host_coeffs(Kokkos::create_mirror_view(device_coeffs)), host_s2(Kokkos::create_mirror_view(device_s2)),
          host_s2x(Kokkos::create_mirror_view(device_s2x))
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
        s.device_s2 = ValueViewType32("SplineInterpolator1DStack_s2_32", sizes[0], sizes[1]);
        s.device_s2x = ValueViewType32("SplineInterpolator1DStack_s2x_32", sizes[0], sizes[1]);
        s.host_values = Kokkos::create_mirror_view(s.device_values);
        s.host_coeffs = Kokkos::create_mirror_view(s.device_coeffs);
        s.host_s2 = Kokkos::create_mirror_view(s.device_s2);
        s.host_s2x = Kokkos::create_mirror_view(s.device_s2x);
        if (s.device_values.span() != device_values.span() || s.host_values.span() != device_values.span())
          throw std::runtime_error("SplineInterpolator1DStack: padded data buffer");
      }
    }

    /// Shallow copy: the copy shares the data, so later update() calls are seen by both.
    KOKKOS_DEFAULTED_FUNCTION SplineInterpolator1DStack(const SplineInterpolator1DStack &) = default;

    /**
     * @brief Replace the data and recompute the splines. Host and device copies are both current when this returns.
     *
     * The end conditions are first derivatives \f$df/di\f$ with respect to the grid index of \f$x\f$ (not to
     * \f$x\f$), the same for every row. A value left at its default (or not finite) gives a natural boundary
     * condition, \f$f'' = 0\f$, at that end. Along \f$s\f$ the ends are fixed by the coordinates (see the class).
     *
     * @param in_data values in row-major order `[s][x]`: `in_data[i * n1 + j]` for stack index i and spline grid point
     * j, with `n1 = get_coordinates().sizes()[1]`; converted to NT
     * @param lower_y1 derivative \f$df/di\f$ at the first spline grid point
     * @param upper_y1 derivative \f$df/di\f$ at the last spline grid point
     */
    // Filled element-wise: the views' storage layout need not be row-major. See
    // LinearInterpolator1D::update() for the copy and fence.
    template <typename NT2>
    void update(const NT2 *in_data, const ctype lower_y1 = std::numeric_limits<ctype>::max(),
                const ctype upper_y1 = std::numeric_limits<ctype>::max())
    {
      // Identical data and end conditions leave the splines as they are: no rebuild, no copy to the device, and
      // caches built from them (see data_epoch.hh) stay valid.
      // Only for plain numbers: == on an autodiff value ignores its derivatives.
      if (std::is_arithmetic_v<NT> && has_data && lower_y1 == last_y1[0] && upper_y1 == last_y1[1]) {
        bool same = true;
        for (size_t i = 0; i < sizes[0] && same; ++i)
          for (size_t j = 0; j < sizes[1] && same; ++j)
            same = host_values(i, j) == NT(in_data[i * sizes[1] + j]);
        if (same) return;
      }
      has_data = true;
      last_y1 = {lower_y1, upper_y1};
      internal::bump_interpolator_data_epoch();

      // Copy values from input data (row-major)
      for (size_t i = 0; i < sizes[0]; ++i)
        for (size_t j = 0; j < sizes[1]; ++j)
          host_values(i, j) = in_data[i * sizes[1] + j];

      build_splines(lower_y1, upper_y1);

      if constexpr (Twin::exists) {
        auto &s = single();
        for (size_t i = 0; i < sizes[0]; ++i)
          for (size_t j = 0; j < sizes[1]; ++j) {
            s.host_values(i, j) = static_cast<NT32>(host_values(i, j));
            s.host_coeffs(i, j) = static_cast<NT32>(host_coeffs(i, j));
            s.host_s2(i, j) = static_cast<NT32>(host_s2(i, j));
            s.host_s2x(i, j) = static_cast<NT32>(host_s2x(i, j));
          }
      }

      if constexpr (has_separate_device) {
        typename ValueViewType::execution_space exec;
        Kokkos::deep_copy(exec, device_values, host_values);
        Kokkos::deep_copy(exec, device_coeffs, host_coeffs);
        Kokkos::deep_copy(exec, device_s2, host_s2);
        Kokkos::deep_copy(exec, device_s2x, host_s2x);
        if constexpr (Twin::exists) {
          auto &s = single();
          Kokkos::deep_copy(exec, s.device_values, s.host_values);
          Kokkos::deep_copy(exec, s.device_coeffs, s.host_coeffs);
          Kokkos::deep_copy(exec, s.device_s2, s.host_s2);
          Kokkos::deep_copy(exec, s.device_s2x, s.host_s2x);
        }
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
        return SplineInterpolator1DStackHandle<NT32, Coordinates32, typename ValueViewType32::array_layout>(
            {{s.device_values.data(), s.device_coeffs.data(), s.device_s2.data(), s.device_s2x.data()}},
            {{s.host_values.data(), s.host_coeffs.data(), s.host_s2.data(), s.host_s2x.data()}}, sizes,
            *s.coordinates);
      } else
        return SplineInterpolator1DStackHandle<NT, Coordinates, typename ValueViewType::array_layout>(
            {{device_values.data(), device_coeffs.data(), device_s2.data(), device_s2x.data()}},
            {{host_values.data(), host_coeffs.data(), host_s2.data(), host_s2x.data()}}, sizes, coordinates);
    }

    /**
     * @brief Value at flat row-major index i, the order update() takes (host only).
     */
    NT operator[](size_t i) const { return host_values(i / sizes[1], i % sizes[1]); }

    /**
     * @brief Fractional grid indices of the point (s, x), i.e. Coordinates::backward(s, x).
     *
     * `f(s, x) == f.at(f.index(s, x))`. Several interpolators on the same coordinate system can share one index()
     * call, which saves the (possibly expensive) coordinate transform.
     *
     * @param s stack variable
     * @param x spline variable
     */
    // Return type spelled out: nvcc loses decltype(var) of a deduced return type inside a class-template member.
    device::array<typename Coordinates::ctype, 2> KOKKOS_FUNCTION
    index(const typename Coordinates::ctype s, const typename Coordinates::ctype x) const
    {
      return internal::spline_stack_index(coordinates, s, x);
    }

    /**
     * @brief Interpolate at fractional grid indices, as returned by index().
     *
     * @param raw fractional grid indices {stack, spline}; out-of-range values are clamped
     */
    NT KOKKOS_FUNCTION at(const device::array<typename Coordinates::ctype, 2> &raw) const
    {
      KOKKOS_IF_ON_DEVICE(
          (return internal::spline_stack_at<NT>(device_values, device_coeffs, device_s2, device_s2x, sizes, raw);))
      KOKKOS_IF_ON_HOST((return internal::spline_stack_at<NT>(host_values, host_coeffs, host_s2, host_s2x, sizes, raw);))
    }

    /**
     * @brief Interpolate the data at the point (s, x).
     *
     * @param s stack variable
     * @param x spline variable
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
     * @brief Pointer to the host copy of the values (read-only).
     *
     * The storage order depends on the backend and need not be row-major; use operator[] for row-major access.
     */
    const NT *data() const { return host_values.data(); }

  private:
    const Coordinates coordinates;
    const device::array<size_t, 2> sizes;

    // values, x-spline coefficients f''/6, s-spline coefficients (d^2f/ds^2)/6 and their x-spline coefficients
    ValueViewType device_values;
    CoeffViewType device_coeffs, device_s2, device_s2x;
    HostValueViewType host_values;
    HostCoeffViewType host_coeffs, host_s2, host_s2x;

    // What the last update() was given, so that an identical update can be skipped.
    bool has_data = false;
    device::array<ctype, 2> last_y1{};

    // The single-precision copy, allocated only if Twin::exists. Behind one host-side handle, so it
    // adds 24 B to this object, which kernels naming the interpolator type still receive in full.
    struct Single {
      std::optional<Coordinates32> coordinates;
      ValueViewType32 device_values, device_coeffs, device_s2, device_s2x;
      HostValueViewType32 host_values, host_coeffs, host_s2, host_s2x;
    };
    Kokkos::View<Single, Kokkos::HostSpace> single;

    /// True if the end condition is a natural one (f'' = 0): the derivative was left at its default or is not finite.
    static bool natural(const ctype y1) { return !std::isfinite(y1) || y1 >= std::numeric_limits<ctype>::max() / 2; }

    /**
     * @brief Second derivatives / 6 of the cubic spline through y on a unit-spaced grid, into y2.
     *
     * End conditions: first derivatives lower_y1, upper_y1, or natural (see natural()).
     */
    static void spline_y2(const std::vector<NT> &y, std::vector<NT> &y2, const ctype lower_y1, const ctype upper_y1)
    {
      const size_t size = y.size();
      y2.assign(size, NT(0));
      if (size < 2) return;

      NT p, qn, sig, un;
      std::vector<NT> u(size - 1);

      if (natural(lower_y1))
        y2[0] = u[0] = 0.0;
      else {
        y2[0] = -0.5;
        u[0] = 3.0 * ((y[1] - y[0]) - lower_y1);
      }
      for (size_t i = 1; i < size - 1; i++) {
        sig = 0.5;
        p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        u[i] = (y[i + 1] - y[i]) - (y[i] - y[i - 1]);
        u[i] = (6.0 * u[i] / 2. - sig * u[i - 1]) / p;
      }
      if (natural(upper_y1))
        qn = un = 0.0;
      else {
        qn = 0.5;
        un = 3.0 * (upper_y1 - (y[size - 1] - y[size - 2]));
      }
      y2[size - 1] = (un - qn * u[size - 2]) / (qn * y2[size - 2] + 1);
      for (int k = size - 2; k >= 0; k--)
        y2[k] = y2[k] * y2[k + 1] + u[k];

      // Precompute division by 6 so the evaluation avoids per-call divides
      for (size_t k = 0; k < size; ++k)
        y2[k] /= (ctype)6;
    }

    /**
     * @brief All coefficients from host_values: x-splines of the rows, s-splines of the columns, and x-splines of
     * the s-coefficients.
     *
     * The s-coefficients are linear in the values, so the x-spline of them equals the s-coefficient of the
     * x-interpolated rows exactly, provided its end conditions are the image of the rows' ones: natural where the
     * rows are natural, and zero slope where the rows share a slope (the s-curvature of equal slopes vanishes).
     */
    void build_splines(const ctype lower_y1, const ctype upper_y1)
    {
      const size_t ns = sizes[0], nx = sizes[1];
      std::vector<NT> y, y2;

      for (size_t i = 0; i < ns; ++i) {
        y.resize(nx);
        for (size_t j = 0; j < nx; ++j)
          y[j] = host_values(i, j);
        spline_y2(y, y2, lower_y1, upper_y1);
        for (size_t j = 0; j < nx; ++j)
          host_coeffs(i, j) = y2[j];
      }

      // A finite-temperature stack starting at the zero frequency is folded, s -> |s|: the data are even there, so
      // the spline has zero slope at the first row. Natural ends otherwise.
      bool mirrored = false;
      if constexpr (requires { Coordinates::folds_frequency; }) mirrored = coordinates.m_start == 0;
      const ctype natural_end = std::numeric_limits<ctype>::max();
      for (size_t j = 0; j < nx; ++j) {
        y.resize(ns);
        for (size_t i = 0; i < ns; ++i)
          y[i] = host_values(i, j);
        spline_y2(y, y2, mirrored ? ctype(0) : natural_end, natural_end);
        for (size_t i = 0; i < ns; ++i)
          host_s2(i, j) = y2[i];
      }

      const ctype lower = natural(lower_y1) ? natural_end : ctype(0);
      const ctype upper = natural(upper_y1) ? natural_end : ctype(0);
      for (size_t i = 0; i < ns; ++i) {
        y.resize(nx);
        for (size_t j = 0; j < nx; ++j)
          y[j] = host_s2(i, j);
        spline_y2(y, y2, lower, upper);
        for (size_t j = 0; j < nx; ++j)
          host_s2x(i, j) = y2[j];
      }
    }
  };
} // namespace DiFfRG
