#pragma once

// standard library
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  /**
   * @brief A coordinate system: a map from grid indices to physical coordinates.
   *
   * Requires `ctype` (scalar type), `dim` (number of axes), size() (number of grid points), from_linear_index(s)
   * (the multi-index of the s-th point) and forward(multi-index) (its physical coordinates).
   */
  template <typename T>
  concept is_coordinates = requires(T t) {
    typename T::ctype;
    T::dim;
    t.size();
    t.forward(device::array<size_t, T::dim>{});
    { t.from_linear_index(size_t{}) } -> std::same_as<device::array<size_t, T::dim>>;
  };

  namespace internal
  {
    template <typename T, typename = void> struct coord_periodic : std::false_type {
    };
    template <typename T>
    struct coord_periodic<T, std::void_t<decltype(T::periodic)>> : std::bool_constant<T::periodic> {
    };
  } // namespace internal

  /**
   * @brief Whether a 1D coordinate class describes a periodic axis, i.e. one where the last grid point is followed
   * again by the first one. Detected through a `static constexpr bool periodic` member, defaulting to false.
   */
  template <typename T> inline constexpr bool is_periodic_coordinate_v = internal::coord_periodic<T>::value;

  namespace internal
  {
    template <typename C, size_t i, typename = void> struct axis_periodic : std::false_type {
    };
    template <typename C, size_t i>
    struct axis_periodic<C, i, std::void_t<typename C::template coordinate_type<i>>>
        : std::bool_constant<is_periodic_coordinate_v<typename C::template coordinate_type<i>>> {
    };
  } // namespace internal

  /**
   * @brief Whether axis i of a (possibly multi-dimensional) coordinate system is periodic. Falls back to false for
   * coordinate systems which do not expose their axes as separate types, e.g. BosonicCoordinates1DFiniteT.
   */
  template <typename C, size_t i> inline constexpr bool is_periodic_axis_v = internal::axis_periodic<C, i>::value;

  namespace internal
  {
    /// @p value with the fewest digits (at least 6) that read back exactly; used by the to_string() methods.
    template <typename NT> std::string round_trip(const NT value)
    {
      char buffer[40];
      for (int precision = 6; precision < std::numeric_limits<NT>::max_digits10; ++precision) {
        std::snprintf(buffer, sizeof(buffer), "%.*g", precision, double(value));
        if (NT(std::strtod(buffer, nullptr)) == value) return std::string(buffer);
      }
      std::snprintf(buffer, sizeof(buffer), "%.*g", std::numeric_limits<NT>::max_digits10, double(value));
      return std::string(buffer);
    }
  } // namespace internal

  /**
   * @brief Product grid of one-dimensional coordinate systems.
   *
   * A point has the multi-index \f$(i_0, \dots, i_{d-1})\f$ and coordinates
   * \f$(x_0(i_0), \dots, x_{d-1}(i_{d-1}))\f$, with \f$x_k\f$ the map of the k-th axis. The linear index is
   * row-major (last axis fastest): \f$s = (\dots(i_0 N_1 + i_1) N_2 + \dots) N_{d-1} + i_{d-1}\f$, \f$N_k\f$ the
   * size of axis k.
   *
   * @tparam Coordinates The one-dimensional axes, all with the same ctype.
   */
  template <typename... Coordinates> class CoordinatePackND
  {
    // Assert that all coordinates have the same ctype
    static_assert((std::is_same<typename Coordinates::ctype,
                                typename device::tuple_element<0, device::tuple<Coordinates...>>::type::ctype>::value &&
                   ...),
                  "CoordinatePackND: all coordinate systems must use the same ctype");
    static_assert(sizeof...(Coordinates) > 0, "CoordinatePackND requires at least one coordinate system");
    // Assert that all coordinates have the dim 1
    static_assert(((Coordinates::dim == 1) && ...), "CoordinatePackND requires all coordinates to have dim 1");

  public:
    /// Scalar type.
    using ctype = typename device::tuple_element<0, device::tuple<Coordinates...>>::type::ctype;
    /// Number of axes.
    static constexpr size_t dim = sizeof...(Coordinates);

    /// From the axes.
    CoordinatePackND(Coordinates... coordinates) : coordinates(coordinates...) {}

    /**
     * @brief Convert from the same axes in another precision, e.g. float from double.
     */
    template <typename... Coordinates2>
      requires(sizeof...(Coordinates2) == sizeof...(Coordinates) &&
               !std::is_same_v<CoordinatePackND<Coordinates2...>, CoordinatePackND> &&
               (std::is_constructible_v<Coordinates, const Coordinates2 &> && ...))
    CoordinatePackND(const CoordinatePackND<Coordinates2...> &other)
        : CoordinatePackND(other, std::index_sequence_for<Coordinates...>{})
    {
    }

    /// The coordinates of the point with indices @p i (one per axis; may be fractional).
    template <typename... I>
      requires(std::is_convertible_v<std::decay_t<I>, size_t> && ...)
    KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> forward(I &&...i) const
    {
      static_assert(sizeof...(I) == sizeof...(Coordinates));
      return forward_impl(device::make_integer_sequence<int, sizeof...(I)>(), device::forward<I>(i)...);
    }
    /// The coordinates of the point with multi-index @p coords.
    template <typename IT>
    KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> forward(const device::array<IT, dim> &coords) const
    {
      return device::apply([&](const auto &...iargs) { return forward(iargs...); }, coords);
    }

    template <typename... I, int... Is>
    device::array<ctype, dim> KOKKOS_FORCEINLINE_FUNCTION forward_impl(device::integer_sequence<int, Is...>,
                                                                       I &&...i) const
    {
      return {{device::get<Is>(coordinates).forward(device::get<Is>(device::tie(i...)))...}};
    }

    /// The (fractional) grid indices of the coordinates @p i, axis by axis.
    template <typename... I> KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, sizeof...(I)> backward(I &&...i) const
    {
      static_assert(sizeof...(I) == sizeof...(Coordinates));
      return backward_impl(device::make_integer_sequence<int, sizeof...(I)>(), device::forward<I>(i)...);
    }

    template <typename... I, int... Is>
    device::array<ctype, sizeof...(I)> KOKKOS_FORCEINLINE_FUNCTION backward_impl(device::integer_sequence<int, Is...>,
                                                                                 I &&...i) const
    {
      return {{device::get<Is>(coordinates).backward(device::get<Is>(device::tie(i...)))...}};
    }

    /// Axis @p i.
    template <size_t i> const auto &get_coordinates() const { return device::get<i>(coordinates); }

    /**
     * @brief The type of the i-th axis. Used e.g. to query per-axis periodicity, see is_periodic_axis_v.
     */
    template <size_t i>
    using coordinate_type = typename device::tuple_element<i, device::tuple<Coordinates...>>::type;

    /// Total number of points, the product of the axis sizes.
    size_t KOKKOS_FORCEINLINE_FUNCTION size() const
    {
      size_t size = 1;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { size *= device::get<i>(coordinates).size(); });
      return size;
    }

    /// The size of each axis.
    device::array<size_t, sizeof...(Coordinates)> KOKKOS_FORCEINLINE_FUNCTION sizes() const
    {
      device::array<size_t, sizeof...(Coordinates)> sizes;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { sizes[i] = device::get<i>(coordinates).size(); });
      return sizes;
    }

    /// The multi-index of the point with linear index @p s (row-major, last axis fastest).
    device::array<size_t, sizeof...(Coordinates)> KOKKOS_INLINE_FUNCTION from_linear_index(size_t s) const
    {
      device::array<size_t, sizeof...(Coordinates)> idx;
      // calculate the index for each coordinate system
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) {
        idx[sizeof...(Coordinates) - 1 - i] = s % device::get<sizeof...(Coordinates) - 1 - i>(coordinates).size();
        s = s / device::get<sizeof...(Coordinates) - 1 - i>(coordinates).size();
      });

      return idx;
    }

    /// Equal if the axes have the same types and are pairwise equal; false for different axis types.
    template <typename... Coordinates2>
    friend bool operator==(const CoordinatePackND<Coordinates...> &r, const CoordinatePackND<Coordinates2...> &l)
    {
      if constexpr (!std::is_same_v<device::tuple<Coordinates...>, device::tuple<Coordinates2...>>)
        return false; // different axes
      else {
        // Check if all coordinates are equal
        bool equal = true;
        constexpr_for<0, sizeof...(Coordinates), 1>(
            [&](auto i) { equal &= (device::get<i>(r.coordinates) == device::get<i>(l.coordinates)); });
        return equal;
      }
    }

    /// "CoordinatePackND(<axis 0>, <axis 1>, ...)"; identifies the grid, e.g. in HDF5 output.
    std::string to_string() const
    {
      std::string ret = "CoordinatePackND(";
      constexpr_for<0, dim, 1>([&](auto i) {
        ret += device::get<i>(coordinates).to_string();
        if (i + 1 < dim) ret += ", ";
      });
      ret += ")";
      return ret;
    }

  protected:
    const device::tuple<Coordinates...> coordinates;

  private:
    template <typename Other, size_t... Is>
    CoordinatePackND(const Other &other, std::index_sequence<Is...>)
        : coordinates(Coordinates(other.template get_coordinates<Is>())...)
    {
    }
  };

  /**
   * @brief The points with linear indices [offset, offset + size) of another coordinate system @p Base.
   *
   * from_linear_index(s) returns Base's multi-index of point offset + s, and forward() is Base's, so positions are
   * identical to those on the full grid. Used to split map() evaluations between MPI ranks.
   */
  template <typename Base> class SubCoordinates : public Base
  {
  public:
    using ctype = typename Base::ctype;
    static constexpr size_t dim = Base::dim;

    /// The window [@p offset, @p offset + @p size) of @p base; throws if empty or out of range.
    SubCoordinates(const Base &base, size_t offset, size_t size) : Base(base), m_offset(offset), m_size(size)
    {
      if (size == 0) throw std::runtime_error("SubCoordinates: size must be > 0");
      if (offset + size > base.size()) throw std::runtime_error("SubCoordinates: offset + size must be <= base.size()");
    }

    /// Per-axis extents; only defined for dim == 1 (a linear window is not a box otherwise).
    device::array<size_t, dim> KOKKOS_FORCEINLINE_FUNCTION sizes() const
    {
      static_assert(dim == 1, "SubCoordinates is a linear index window, which is a per-axis box only for dim == 1.");
      return device::array<size_t, dim>{{m_size}};
    }

    /// Number of points in the window.
    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return m_size; }

    /// Base's multi-index of the window's @p s-th point.
    device::array<size_t, dim> KOKKOS_INLINE_FUNCTION from_linear_index(size_t s) const
    {
      return Base::from_linear_index(m_offset + s);
    }

    /// Base::forward of the Base indices @p i.
    template <typename... I>
      requires(std::is_convertible_v<std::decay_t<I>, size_t> && ...)
    KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> forward(I &&...i) const
    {
      static_assert(sizeof...(I) == dim);
      return forward(device::array<size_t, dim>{{static_cast<size_t>(i)...}});
    }

    /// Base::forward of a Base multi-index (as returned by from_linear_index()).
    template <typename IT> KOKKOS_INLINE_FUNCTION device::array<ctype, dim> forward(device::array<IT, dim> i) const
    {
      return Base::forward(i);
    }

    // backward() is Base's.

    /// "SubCoordinates(<base>, offset, size)"; identifies the window.
    std::string to_string() const
    {
      return "SubCoordinates(" + Base::to_string() + ", " + std::to_string(m_offset) + ", " + std::to_string(m_size) +
             ")";
    }

  private:
    size_t m_offset;
    size_t m_size;
  };

  /**
   * @brief Equally spaced points on [start, stop], including both ends.
   *
   * With N = grid_extent ≥ 2 and grid index i (real values allowed):
   * \f[ x(i) = \mathrm{start} + i\,\frac{\mathrm{stop} - \mathrm{start}}{N - 1}\,. \f]
   *
   * @tparam NT Scalar type of the coordinates.
   */
  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class LinearCoordinates1D
  {
  public:
    /// Scalar type.
    using ctype = NT;
    /// One axis.
    static constexpr size_t dim = 1;

    /// @p grid_extent points (≥ 2, throws otherwise) from @p start to @p stop.
    LinearCoordinates1D(size_t grid_extent, double start, double stop)
        : start(start), stop(stop), grid_extent(grid_extent), parent_extent(grid_extent)
    {
      if (grid_extent < 2) throw std::runtime_error("LinearCoordinates1D: grid_extent must be > 1");
      a = (stop - start) / (grid_extent - 1.);
    }

    /// Convert from another precision.
    template <typename NT2>
    LinearCoordinates1D(const LinearCoordinates1D<NT2> &other)
        : LinearCoordinates1D(LinearCoordinates1D(other.parent_size(), other.start, other.stop), other.size())
    {
    }

    /**
     * @brief The first @p n points of this grid, as a grid of the same type.
     *
     * forward(i) is bit-identical to that of the full grid for i < n, and backward() is unchanged; use it to skip the
     * upper end of a grid without changing the coordinate type. The prefix keeps the parameters and point count of the
     * full grid (parent_size()); to_string() appends ".prefix(n)".
     *
     * @param n Number of points kept, 1 ≤ n ≤ size().
     */
    LinearCoordinates1D prefix(size_t n) const
    {
      if (n < 1 || n > grid_extent) throw std::runtime_error("LinearCoordinates1D::prefix: need 1 <= n <= size()");
      return LinearCoordinates1D(*this, n);
    }

    /// Number of points of the grid this one is a prefix() of; size() unless it is a prefix.
    size_t KOKKOS_FORCEINLINE_FUNCTION parent_size() const { return parent_extent; }

    /// The multi-index {i} of point i.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /// The coordinate \f$x(i)\f$ at grid index @p x (real values allowed).
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const { return start + a * x; }

    /// As forward(x[0]).
    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /// The inverse map: the (fractional) grid index of the coordinate @p y.
    NT KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const { return (y - start) / a; }

    /// Number of points N.
    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    /// {N}.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    /// First and last point.
    const NT start, stop;

    /// Same precision, parameters and number of points (and the same parent, for a prefix).
    template <typename NT2>
    friend bool operator==(const LinearCoordinates1D<NT> &lhs, const LinearCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.size() == rhs.size() &&
             lhs.parent_size() == rhs.parent_size();
    }

    /// "LinearCoordinates1D(N, start, stop)"; identifies the grid, e.g. in HDF5 output.
    std::string to_string() const
    {
      return "LinearCoordinates1D(" + std::to_string(parent_extent) + ", " + internal::round_trip(start) + ", " +
             internal::round_trip(stop) + ")" +
             (grid_extent == parent_extent ? std::string() : ".prefix(" + std::to_string(grid_extent) + ")");
    }

  private:
    // prefix(): the parent's parameters and spacing, fewer points
    LinearCoordinates1D(const LinearCoordinates1D &parent, size_t n)
        : start(parent.start), stop(parent.stop), grid_extent(n), parent_extent(parent.parent_extent), a(parent.a)
    {
    }

    const size_t grid_extent;
    size_t parent_extent;
    NT a;
  };

  /**
   * @brief Equally spaced points on a periodic axis of period stop - start; stop itself is excluded.
   *
   * With N = grid_extent ≥ 1: \f$ x(i) = \mathrm{start} + i\,(\mathrm{stop} - \mathrm{start})/N \f$, i.e. the
   * points start, ..., stop - (stop - start)/N, since start and stop are the same point (e.g. an angle).
   * backward() folds into [0, N); interpolators wrap around between the last and the first point.
   *
   * @tparam NT Scalar type of the coordinates.
   */
  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class LinearPeriodicCoordinates1D
  {
  public:
    /// Scalar type.
    using ctype = NT;
    /// One axis.
    static constexpr size_t dim = 1;
    /// Marks the axis as periodic.
    static constexpr bool periodic = true;

    /// @p grid_extent points (≥ 1, throws otherwise) on the period [@p start, @p stop).
    LinearPeriodicCoordinates1D(size_t grid_extent, double start, double stop)
        : start(start), stop(stop), grid_extent(grid_extent), extent(static_cast<NT>(grid_extent))
    {
      if (grid_extent == 0) throw std::runtime_error("LinearPeriodicCoordinates1D: grid_extent must be > 0");
      a = (stop - start) / NT(grid_extent);
    }

    /// Convert from another precision.
    template <typename NT2>
    LinearPeriodicCoordinates1D(const LinearPeriodicCoordinates1D<NT2> &other)
        : LinearPeriodicCoordinates1D(other.size(), other.start, other.stop)
    {
    }

    /// The multi-index {i} of point i.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /// The coordinate \f$x(i)\f$ at grid index @p x (real values allowed).
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const { return start + a * x; }

    /// As forward(x[0]).
    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /**
     * @brief The inverse map: the (fractional) grid index of the coordinate @p y, folded into [0, N).
     */
    NT KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const
    {
      using Kokkos::floor;
      // fmod would keep the sign of idx, so fold explicitly into [0, grid_extent)
      NT idx = (y - start) / a;
      idx -= floor(idx / extent) * extent;
      // right at the seam the division can round such that the fold leaves idx just outside the interval
      if (idx >= extent) idx -= extent;
      if (idx < NT(0)) idx = NT(0);
      return idx;
    }

    /// Number of points N.
    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    /// {N}.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    /// Ends of the period.
    const NT start, stop;

    /// Same precision, parameters and number of points.
    template <typename NT2>
    friend bool operator==(const LinearPeriodicCoordinates1D<NT> &lhs, const LinearPeriodicCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.grid_extent == rhs.grid_extent;
    }

    /// "LinearPeriodicCoordinates1D(N, start, stop)"; identifies the grid.
    std::string to_string() const
    {
      return "LinearPeriodicCoordinates1D(" + std::to_string(grid_extent) + ", " + internal::round_trip(start) + ", " +
             internal::round_trip(stop) + ")";
    }

  private:
    const size_t grid_extent;
    const NT extent;
    NT a;
  };

  /**
   * @brief Points on [start, stop], including both ends, with spacing growing exponentially in the index.
   *
   * With N = grid_extent ≥ 2, bias b ≠ 0 and grid index i (real values allowed):
   * \f[ x(i) = \mathrm{start} + (\mathrm{stop} - \mathrm{start})\,\frac{e^{b\,i/(N-1)} - 1}{e^{b} - 1}\,. \f]
   * For b > 0 the points cluster near start (consecutive spacings grow by the factor \f$e^{b/(N-1)}\f$), for b < 0
   * near stop; b → 0 approaches LinearCoordinates1D.
   *
   * @tparam NT Scalar type of the coordinates.
   */
  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class LogarithmicCoordinates1D
  {
  public:
    /// Scalar type.
    using ctype = NT;
    /// One axis.
    static constexpr size_t dim = 1;

    /// @p grid_extent points (≥ 2) from @p start to @p stop with @p bias (≠ 0); throws otherwise. Constants are
    /// computed in double.
    LogarithmicCoordinates1D(size_t grid_extent, double start, double stop, double bias)
        : start(start), stop(stop), bias(bias), grid_extent(grid_extent), parent_extent(grid_extent),
          gem1(grid_extent - 1.), gem1inv(1. / (grid_extent - 1.)), a(bias), b((stop - start) / std::expm1(bias)),
          c(start)
    {
      if (grid_extent < 2) throw std::runtime_error("LogarithmicCoordinates1D: grid_extent must be > 1");
      if (bias == 0.) throw std::runtime_error("LogarithmicCoordinates1D: bias must be != 0");
    }

    /// Convert from another precision.
    template <typename NT2>
    LogarithmicCoordinates1D(const LogarithmicCoordinates1D<NT2> &other)
        : LogarithmicCoordinates1D(LogarithmicCoordinates1D(other.parent_size(), other.start, other.stop, other.bias),
                                   other.size())
    {
    }

    /**
     * @brief The first @p n points of this grid, as a grid of the same type.
     *
     * forward(i) is bit-identical to that of the full grid for i < n, and backward() is unchanged; use it to skip the
     * upper end of a grid without changing the coordinate type. The prefix keeps the parameters and point count of the
     * full grid (parent_size()); to_string() appends ".prefix(n)".
     *
     * @param n Number of points kept, 1 ≤ n ≤ size().
     */
    LogarithmicCoordinates1D prefix(size_t n) const
    {
      if (n < 1 || n > grid_extent) throw std::runtime_error("LogarithmicCoordinates1D::prefix: need 1 <= n <= size()");
      return LogarithmicCoordinates1D(*this, n);
    }

    /// Number of points of the grid this one is a prefix() of; size() unless it is a prefix.
    size_t KOKKOS_FORCEINLINE_FUNCTION parent_size() const { return parent_extent; }

    /// The multi-index {i} of point i.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /// The coordinate \f$x(i)\f$ at grid index @p x (real values allowed).
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const
    {
      using Kokkos::expm1;
      return b * expm1(a * x * gem1inv) + c;
    }

    /// As forward(x[0]).
    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /// The inverse map: the (fractional) grid index of the coordinate @p y.
    NT KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const
    {
      using Kokkos::log1p;
      return log1p((y - c) / b) * gem1 / a;
    }

    /// \f$di/dx\f$ at the coordinate @p y.
    NT KOKKOS_FORCEINLINE_FUNCTION backward_derivative(const NT &y) const { return gem1 / (a * (b + y - c)); }

    /// Number of points N.
    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    /// {N}.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    /// The parameters.
    const NT start, stop, bias;

    /// Same precision, parameters and number of points (and the same parent, for a prefix).
    template <typename NT2>
    friend bool operator==(const LogarithmicCoordinates1D<NT> &lhs, const LogarithmicCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.bias == rhs.bias && lhs.size() == rhs.size() &&
             lhs.parent_size() == rhs.parent_size();
    }

    /// "LogarithmicCoordinates1D(start, stop, bias, N)"; identifies the grid.
    std::string to_string() const
    {
      return "LogarithmicCoordinates1D(" + internal::round_trip(start) + ", " + internal::round_trip(stop) + ", " +
             internal::round_trip(bias) + ", " + std::to_string(parent_extent) + ")" +
             (grid_extent == parent_extent ? std::string() : ".prefix(" + std::to_string(grid_extent) + ")");
    }

  private:
    // prefix(): the parent's parameters and spacing (gem1 is the parent's N - 1), fewer points
    LogarithmicCoordinates1D(const LogarithmicCoordinates1D &parent, size_t n)
        : start(parent.start), stop(parent.stop), bias(parent.bias), grid_extent(n),
          parent_extent(parent.parent_extent), gem1(parent.gem1), gem1inv(parent.gem1inv), a(parent.a), b(parent.b),
          c(parent.c)
    {
    }

    const size_t grid_extent;
    size_t parent_extent;
    const NT gem1, gem1inv;
    NT a, b, c;
  };

  /**
   * @brief Points on [start, stop] (start > 0), logarithmically spaced and clustered around an interior scale
   * `center`.
   *
   * In \f$u = \ln x\f$, with \f$u_0 = \ln(\mathrm{center})\f$, \f$c\f$ = focus and grid index i (real values
   * allowed):
   * \f[
   *   u(i) = u_0 + \frac{\sinh(s_{\min} + a\,i)}{c}\,, \qquad s(u) = \operatorname{asinh}\big(c\,(u - u_0)\big)\,,
   * \f]
   * with \f$s_{\min} = s(\ln \mathrm{start})\f$ and \f$a = (s(\ln \mathrm{stop}) - s_{\min})/(N - 1)\f$, N =
   * grid_extent. The point density per unit \f$u\f$ is \f$\propto 1/\sqrt{1 + c^2 (u - u_0)^2}\f$: highest at
   * center, falling off like \f$1/(c\,|u - u_0|)\f$. c = 0 gives a geometric grid; a center outside [start, stop]
   * gives a one-sided stretch. forward(0) returns start exactly.
   *
   * Example: 64 points on [0.005, 250] with center = 1, focus = 2 have a spacing of 0.021 decades at 1 and 32 points
   * in [0.3, 3].
   */
  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class FocusedLogCoordinates1D
  {
  public:
    /// Scalar type.
    using ctype = NT;
    /// One axis.
    static constexpr size_t dim = 1;

    /**
     * @brief The grid; throws if a parameter is out of range.
     *
     * @param grid_extent number of grid points, must be at least 2
     * @param start first grid point, must be positive
     * @param stop last grid point, must be larger than start
     * @param center the scale to cluster around, in the same units as start and stop
     * @param focus clustering strength; 0 gives a pure logarithmic grid
     *
     * The map constants are computed in double whatever NT is, and only then rounded to NT.
     */
    FocusedLogCoordinates1D(size_t grid_extent, double start, double stop, double center, double focus)
        : start(start), stop(stop), center(center), focus(focus), grid_extent(grid_extent), parent_extent(grid_extent)
    {
      if (grid_extent < 2) throw std::runtime_error("FocusedLogCoordinates1D: grid_extent must be > 1");
      if (!(start > 0.)) throw std::runtime_error("FocusedLogCoordinates1D: start must be > 0");
      if (!(stop > start)) throw std::runtime_error("FocusedLogCoordinates1D: stop must be > start");
      if (!(center > 0.)) throw std::runtime_error("FocusedLogCoordinates1D: center must be > 0");
      if (!(focus >= 0.)) throw std::runtime_error("FocusedLogCoordinates1D: focus must be >= 0");

      const double u_start = std::log(start);
      const double u_stop = std::log(stop);
      const double u0_d = std::log(center);
      // sinh(s)/c is 0/0 at c == 0; below this threshold the map is a pure logarithmic grid to
      // within round-off anyway, so switch to it explicitly.
      pure_log = !(focus > 1e-8);

      double s_min_d, a_d;
      if (pure_log) {
        // in this branch s *is* u, so forward() degenerates to start * exp(u - log(start))
        s_min_d = u_start;
        a_d = (u_stop - u_start) / double(grid_extent - 1);
      } else {
        s_min_d = s_of_u(u_start, focus, u0_d);
        a_d = (s_of_u(u_stop, focus, u0_d) - s_min_d) / double(grid_extent - 1);
      }
      u0 = u0_d;
      c = focus;
      c_inv = pure_log ? NT(1) : NT(1. / focus);
      s_min = s_min_d;
      a = a_d;
      a_inv = NT(1. / a_d);
      // Store sinh(s_min), NOT sinh(s_min) * c_inv: forward() subtracts this BEFORE scaling by
      // c_inv, which is what makes x == 0 cancel exactly. See the note in forward(). For the same
      // reason it is evaluated in NT on the ROUNDED s_min, i.e. on exactly what forward() sees.
      using Kokkos::sinh;
      g_min = pure_log ? s_min : sinh(s_min);
    }

    /// Convert from another precision.
    template <typename NT2>
    FocusedLogCoordinates1D(const FocusedLogCoordinates1D<NT2> &other)
        : FocusedLogCoordinates1D(
              FocusedLogCoordinates1D(other.parent_size(), other.start, other.stop, other.center, other.focus),
              other.size())
    {
    }

    /**
     * @brief The first @p n points of this grid, as a grid of the same type.
     *
     * forward(i) is bit-identical to that of the full grid for i < n, and backward() is unchanged; use it to skip the
     * upper end of a grid without changing the coordinate type. The prefix keeps the parameters and point count of the
     * full grid (parent_size()); to_string() appends ".prefix(n)".
     *
     * @param n Number of points kept, 1 ≤ n ≤ size().
     */
    FocusedLogCoordinates1D prefix(size_t n) const
    {
      if (n < 1 || n > grid_extent) throw std::runtime_error("FocusedLogCoordinates1D::prefix: need 1 <= n <= size()");
      return FocusedLogCoordinates1D(*this, n);
    }

    /// Number of points of the grid this one is a prefix() of; size() unless it is a prefix.
    size_t KOKKOS_FORCEINLINE_FUNCTION parent_size() const { return parent_extent; }

    /// The multi-index {i} of point i.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /// The coordinate \f$x(i)\f$ at grid index @p x (real values allowed).
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const
    {
      using Kokkos::exp, Kokkos::sinh;
      const NT s = s_min + a * x;
      // start * exp(...) and subtracting g_min before scaling by c_inv (no FMA contraction) make x == 0 give start
      // exactly.
      const NT d = pure_log ? (s - g_min) : (sinh(s) - g_min) * c_inv;
      return start * exp(d);
    }

    /// As forward(x[0]).
    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /// The inverse map: the (fractional) grid index of the coordinate @p y.
    NT KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const
    {
      using Kokkos::log;
      const NT u = log(y);
      return ((pure_log ? u : s_of_u(u, c, u0)) - s_min) * a_inv;
    }

    /// \f$di/dx\f$ at the coordinate @p y.
    NT KOKKOS_FORCEINLINE_FUNCTION backward_derivative(const NT &y) const
    {
      using Kokkos::log, Kokkos::sqrt;
      if (pure_log) return a_inv / y;
      const NT v = c * (log(y) - u0);
      return a_inv * c / (sqrt(v * v + NT(1)) * y);
    }

    /// Number of points N.
    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    /// {N}.
    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    /// The parameters.
    const NT start, stop, center, focus;

    /// Same precision, parameters and number of points (and the same parent, for a prefix).
    template <typename NT2>
    friend bool operator==(const FocusedLogCoordinates1D<NT> &lhs, const FocusedLogCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.center == rhs.center && lhs.focus == rhs.focus &&
             lhs.size() == rhs.size() && lhs.parent_size() == rhs.parent_size();
    }

    /// "FocusedLogCoordinates1D(N, start, stop, center, focus)"; identifies the grid.
    std::string to_string() const
    {
      return "FocusedLogCoordinates1D(" + std::to_string(parent_extent) + ", " + internal::round_trip(start) + ", " +
             internal::round_trip(stop) + ", " + internal::round_trip(center) + ", " + internal::round_trip(focus) +
             ")" + (grid_extent == parent_extent ? std::string() : ".prefix(" + std::to_string(grid_extent) + ")");
    }

  private:
    /// asinh(c (u - u0)), in a form without cancellation for negative arguments.
    template <typename T> static KOKKOS_FORCEINLINE_FUNCTION T s_of_u(const T u, const T c, const T u0)
    {
      using Kokkos::log, Kokkos::sqrt;
      const T v = c * (u - u0);
      const T D = sqrt(v * v + T(1));
      return v >= T(0) ? log(v + D) : -log(D - v);
    }

    // prefix(): the parent's parameters and spacing, fewer points
    FocusedLogCoordinates1D(const FocusedLogCoordinates1D &parent, size_t n)
        : start(parent.start), stop(parent.stop), center(parent.center), focus(parent.focus), grid_extent(n),
          parent_extent(parent.parent_extent), u0(parent.u0), s_min(parent.s_min), a(parent.a), a_inv(parent.a_inv),
          c(parent.c), c_inv(parent.c_inv), g_min(parent.g_min), pure_log(parent.pure_log)
    {
    }

    const size_t grid_extent;
    size_t parent_extent;
    NT u0, s_min, a, a_inv, c, c_inv, g_min;
    bool pure_log;
  };

  /// All grid points, in linear index order, as arrays of dim coordinates.
  template <typename Coordinates> auto make_grid(const Coordinates &coordinates)
  {
    using ctype = typename Coordinates::ctype;
    using cortype = device::array<ctype, Coordinates::dim>;
    std::vector<cortype> grid(coordinates.size());
    for (size_t i = 0; i < coordinates.size(); ++i) {
      const auto forwarded = coordinates.forward(coordinates.from_linear_index(i));
      for (size_t j = 0; j < Coordinates::dim; ++j) {
        grid[i][j] = forwarded[j];
      }
    }
    return grid;
  }

  /// All grid points, in linear index order, flattened: point i occupies entries [i dim, (i + 1) dim).
  template <typename Coordinates> std::vector<typename Coordinates::ctype> dump_grid(const Coordinates &coordinates)
  {
    using ctype = typename Coordinates::ctype;
    std::vector<typename Coordinates::ctype> grid(coordinates.size() * Coordinates::dim);
    for (size_t i = 0; i < coordinates.size(); ++i) {
      for (size_t j = 0; j < Coordinates::dim; ++j) {
        const auto forwarded = coordinates.forward(coordinates.from_linear_index(i));
        grid[i * coordinates.dim + j] = forwarded[j];
      }
    }
    return grid;
  }

  /**
   * @brief The same coordinate system with ctype NT, e.g. the float twin of a double grid:
   * rebind_ctype_t<CoordinatePackND<FocusedLogCoordinates1D<double>, LinearCoordinates1D<double>>, float>.
   */
  template <typename Coordinates, typename NT> struct rebind_ctype;
  template <typename NT0, typename NT> struct rebind_ctype<LinearCoordinates1D<NT0>, NT> {
    using type = LinearCoordinates1D<NT>;
  };
  template <typename NT0, typename NT> struct rebind_ctype<LinearPeriodicCoordinates1D<NT0>, NT> {
    using type = LinearPeriodicCoordinates1D<NT>;
  };
  template <typename NT0, typename NT> struct rebind_ctype<LogarithmicCoordinates1D<NT0>, NT> {
    using type = LogarithmicCoordinates1D<NT>;
  };
  template <typename NT0, typename NT> struct rebind_ctype<FocusedLogCoordinates1D<NT0>, NT> {
    using type = FocusedLogCoordinates1D<NT>;
  };
  template <typename... Cs, typename NT>
    requires(requires { typename rebind_ctype<Cs, NT>::type; } && ...)
  struct rebind_ctype<CoordinatePackND<Cs...>, NT> {
    using type = CoordinatePackND<typename rebind_ctype<Cs, NT>::type...>;
  };
  template <typename Coordinates, typename NT> using rebind_ctype_t = typename rebind_ctype<Coordinates, NT>::type;

  // Common coordinate systems (double precision)
  using LogCoordinates = LogarithmicCoordinates1D<double>;
  using LinCoordinates = LinearCoordinates1D<double>;
  using LogLogCoordinates = CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>>;
  using LogLinCoordinates = CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>;
  using LinLogCoordinates = CoordinatePackND<LinearCoordinates1D<double>, LogarithmicCoordinates1D<double>>;
  using LinLinCoordinates = CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>>;
  using LogLogLinCoordinates =
      CoordinatePackND<LogarithmicCoordinates1D<double>, LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>>;
  using LogLinLinCoordinates =
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>;
  using LinLinLinCoordinates =
      CoordinatePackND<LinearCoordinates1D<double>, LinearCoordinates1D<double>, LinearCoordinates1D<double>>;

  // Logarithmic coordinates focused on an interior scale, see FocusedLogCoordinates1D
  using FocusedLogCoordinates = FocusedLogCoordinates1D<double>;
  using FocusedLogLinCoordinates = CoordinatePackND<FocusedLogCoordinates1D<double>, LinearCoordinates1D<double>>;
  using FocusedLogLinLinCoordinates = CoordinatePackND<FocusedLogCoordinates1D<double>, LinearCoordinates1D<double>,
                                                       LinearCoordinates1D<double>>;

  // Periodic (angular) coordinates
  using LinPeriodicCoordinates = LinearPeriodicCoordinates1D<double>;
  using LogLinPeriodicCoordinates =
      CoordinatePackND<LogarithmicCoordinates1D<double>, LinearPeriodicCoordinates1D<double>>;
  using LogLinLinPeriodicCoordinates = CoordinatePackND<LogarithmicCoordinates1D<double>, LinearCoordinates1D<double>,
                                                        LinearPeriodicCoordinates1D<double>>;
  using FocusedLogLinPeriodicCoordinates =
      CoordinatePackND<FocusedLogCoordinates1D<double>, LinearPeriodicCoordinates1D<double>>;
  using FocusedLogLinLinPeriodicCoordinates =
      CoordinatePackND<FocusedLogCoordinates1D<double>, LinearCoordinates1D<double>,
                       LinearPeriodicCoordinates1D<double>>;
} // namespace DiFfRG
