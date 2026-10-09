#pragma once

// standard library
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  // A concept for what is a coordinate class
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

  /**
   * @brief Utility class for combining multiple coordinate systems into one
   *
   * @tparam Coordinates... the coordinate systems to combine
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
    using ctype = typename device::tuple_element<0, device::tuple<Coordinates...>>::type::ctype;
    static constexpr size_t dim = sizeof...(Coordinates);

    /**
     * @brief Construct a new CoordinatePackND object
     *
     * @param coordinates the coordinate systems to combine
     */
    CoordinatePackND(Coordinates... coordinates) : coordinates(coordinates...) {}

    /**
     * @brief Convert axis by axis from a pack of the same coordinate systems in another precision,
     * e.g. the float twin of a double grid. Implicit, like the converting constructors of the axes.
     */
    template <typename... Coordinates2>
      requires(sizeof...(Coordinates2) == sizeof...(Coordinates) &&
               !std::is_same_v<CoordinatePackND<Coordinates2...>, CoordinatePackND> &&
               (std::is_constructible_v<Coordinates, const Coordinates2 &> && ...))
    CoordinatePackND(const CoordinatePackND<Coordinates2...> &other)
        : CoordinatePackND(other, std::index_sequence_for<Coordinates...>{})
    {
    }

    template <typename... I>
      requires(std::is_convertible_v<std::decay_t<I>, size_t> && ...)
    KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> forward(I &&...i) const
    {
      static_assert(sizeof...(I) == sizeof...(Coordinates));
      return forward_impl(device::make_integer_sequence<int, sizeof...(I)>(), device::forward<I>(i)...);
    }
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

    template <size_t i> const auto &get_coordinates() const { return device::get<i>(coordinates); }

    /**
     * @brief The type of the i-th axis. Used e.g. to query per-axis periodicity, see is_periodic_axis_v.
     */
    template <size_t i>
    using coordinate_type = typename device::tuple_element<i, device::tuple<Coordinates...>>::type;

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const
    {
      // multiply all sizes
      size_t size = 1;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { size *= device::get<i>(coordinates).size(); });
      return size;
    }

    device::array<size_t, sizeof...(Coordinates)> KOKKOS_FORCEINLINE_FUNCTION sizes() const
    {
      device::array<size_t, sizeof...(Coordinates)> sizes;
      constexpr_for<0, sizeof...(Coordinates), 1>([&](auto i) { sizes[i] = device::get<i>(coordinates).size(); });
      return sizes;
    }

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

    template <typename... Coordinates2>
    friend bool operator==(const CoordinatePackND<Coordinates...> &r, const CoordinatePackND<Coordinates2...> &l)
    {
      if constexpr (sizeof...(Coordinates) != sizeof...(Coordinates2)) return false; // Different number of coordinates
      if constexpr ((!std::is_same_v<Coordinates, Coordinates2> && ...))
        return false; // Different types of coordinates
      else if constexpr (sizeof...(Coordinates) == 0)
        return true; // Empty coordinate pack, always equal
      else {
        // Check if all coordinates are equal
        bool equal = true;
        constexpr_for<0, sizeof...(Coordinates), 1>(
            [&](auto i) { equal &= (device::get<i>(r.coordinates) == device::get<i>(l.coordinates)); });
        return equal;
      }
    }

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
   * @brief A contiguous window into the *linear* index range of another coordinate system.
   *
   * The window is `[offset, offset + size)` of `Base`'s flattened grid, which is exactly the shape
   * MapScheduler hands out (`part(G, r, j)`), and it is deliberately **not** a per-axis box. An
   * earlier version derived one, by running `Base::from_linear_index()` on both ends and
   * subtracting; that is only the same set when `dim == 1`, which is why splitting used to be
   * restricted to one-dimensional grids and the expensive multi-angle vertex flows were assigned
   * whole to a single rank.
   *
   * Carrying the linear offset instead makes the window exact in any dimension, at no cost:
   * `from_linear_index(s)` returns the **base** multi-index of global point `offset + s`, and
   * `forward()` is then `Base::forward` unchanged. So the position computed for a windowed point is
   * the same fp64 value as the unwindowed run computes for that global point, which is what keeps a
   * distributed result bitwise identical to a serial one.
   *
   * Only MapScheduler's four `map()` call sites construct this.
   */
  template <typename Base> class SubCoordinates : public Base
  {
  public:
    using ctype = typename Base::ctype;
    static constexpr size_t dim = Base::dim;

    SubCoordinates(const Base &base, size_t offset, size_t size) : Base(base), m_offset(offset), m_size(size)
    {
      if (size == 0) throw std::runtime_error("SubCoordinates: size must be > 0");
      if (offset + size > base.size()) throw std::runtime_error("SubCoordinates: offset + size must be <= base.size()");
    }

    /// Per-axis extents, which a linear window only has when the grid is one-dimensional. Nothing
    /// on the map() path calls this -- the integrators drive everything off size() and
    /// from_linear_index() -- so a caller that needs a box should say so at compile time rather
    /// than receive a plausible-looking wrong answer.
    device::array<size_t, dim> KOKKOS_FORCEINLINE_FUNCTION sizes() const
    {
      static_assert(dim == 1, "SubCoordinates is a linear index window, which is a per-axis box only for dim == 1.");
      return device::array<size_t, dim>{{m_size}};
    }

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return m_size; }

    /// The base multi-index of the window's s-th point. Note this is a *base* index: it is fed
    /// straight back into Base::forward(), which is what makes the positions identical to the
    /// unwindowed ones.
    device::array<size_t, dim> KOKKOS_INLINE_FUNCTION from_linear_index(size_t s) const
    {
      return Base::from_linear_index(m_offset + s);
    }

    /// Spelled-out array rather than `forward({{i...}})`: a braced-init-list cannot deduce the
    /// element type of the overload below, so the old spelling was ill-formed the moment anyone
    /// called it. Nothing on the map() path does, which is why it never showed up.
    template <typename... I>
      requires(std::is_convertible_v<std::decay_t<I>, size_t> && ...)
    KOKKOS_FORCEINLINE_FUNCTION device::array<ctype, dim> forward(I &&...i) const
    {
      static_assert(sizeof...(I) == dim);
      return forward(device::array<size_t, dim>{{static_cast<size_t>(i)...}});
    }

    /// Takes a *base* multi-index, i.e. what from_linear_index() above returns.
    template <typename IT> KOKKOS_INLINE_FUNCTION device::array<ctype, dim> forward(device::array<IT, dim> i) const
    {
      return Base::forward(i);
    }

    // backward() is deliberately not overridden: it is the inverse of forward(), which is now
    // Base's unchanged, so Base::backward is already the right answer and is inherited as is.

    /// Identifies the window, not just the base grid: the position cache in QuadratureIntegrator
    /// keys on this, and two windows into one base grid must not collide.
    std::string to_string() const
    {
      return "SubCoordinates(" + Base::to_string() + ", " + std::to_string(m_offset) + ", " + std::to_string(m_size) +
             ")";
    }

  private:
    size_t m_offset;
    size_t m_size;
  };

  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class LinearCoordinates1D
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 1;

    LinearCoordinates1D(size_t grid_extent, double start, double stop)
        : start(start), stop(stop), grid_extent(grid_extent), parent_extent(grid_extent)
    {
      if (grid_extent == 0) throw std::runtime_error("LinearCoordinates1D: grid_extent must be > 0");
      a = (stop - start) / (grid_extent - 1.);
    }

    template <typename NT2>
    LinearCoordinates1D(const LinearCoordinates1D<NT2> &other)
        : LinearCoordinates1D(LinearCoordinates1D(other.parent_size(), other.start, other.stop), other.size())
    {
    }

    /**
     * @brief The first n points of this grid, as a grid of the same type.
     *
     * forward(i) of the result is bit-identical to forward(i) here for every i < n, and backward()
     * is unchanged, so a kernel mapped over prefix(n) fills exactly the first n entries it would
     * fill over the full grid. That is how to skip the upper end of a grid -- e.g. momenta p >> k
     * whose loops have decoupled -- without a different coordinate type, and hence without
     * regenerating kernels that take this type by name. (SubCoordinates is the general window,
     * but a different type.)
     *
     * The spacing is copied, not re-derived: a grid rebuilt from n points and the prefix's last
     * point would put its points elsewhere (by an ulp, or -- where the spacing depends on the
     * point count -- entirely). A prefix therefore keeps its parent's parameters and point count
     * (parent_size()), and to_string() renders it as "<parent>.prefix(n)", so the position caches
     * and HDF5 datasets keyed on it never confuse it with the parent or with a genuine n-point
     * grid. For a grid that is not a prefix, to_string() and operator== are unchanged.
     *
     * @param n number of points kept, 1 <= n <= size()
     */
    LinearCoordinates1D prefix(size_t n) const
    {
      if (n < 1 || n > grid_extent) throw std::runtime_error("LinearCoordinates1D::prefix: need 1 <= n <= size()");
      return LinearCoordinates1D(*this, n);
    }

    /// Number of points of the grid this one is a prefix() of; size() unless it is a prefix.
    size_t KOKKOS_FORCEINLINE_FUNCTION parent_size() const { return parent_extent; }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const { return start + a * x; }

    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    NT KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const { return (y - start) / a; }

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    const NT start, stop;

    template <typename NT2>
    friend bool operator==(const LinearCoordinates1D<NT> &lhs, const LinearCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.size() == rhs.size() &&
             lhs.parent_size() == rhs.parent_size();
    }

    std::string to_string() const
    {
      return "LinearCoordinates1D(" + std::to_string(parent_extent) + ", " + std::to_string(start) + ", " +
             std::to_string(stop) + ")" + (grid_extent == parent_extent ? std::string() : ".prefix(" + std::to_string(grid_extent) + ")");
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
   * @brief Linear coordinates on a periodic axis of period (stop - start).
   *
   * In contrast to LinearCoordinates1D, the endpoint is *excluded*: the grid points are
   * start, start + a, ..., stop - a with a = (stop - start) / grid_extent, because stop and start describe the same
   * physical point. For an angle this means one full plane of grid points less than the naive [start, stop] grid, and
   * no duplicated, unconstrained degrees of freedom.
   *
   * backward() folds the returned fractional index into [0, grid_extent); interpolators aware of
   * is_periodic_coordinate_v additionally wrap the upper stencil index around to 0.
   */
  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class LinearPeriodicCoordinates1D
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 1;
    static constexpr bool periodic = true;

    LinearPeriodicCoordinates1D(size_t grid_extent, double start, double stop)
        : start(start), stop(stop), grid_extent(grid_extent), extent(static_cast<NT>(grid_extent))
    {
      if (grid_extent == 0) throw std::runtime_error("LinearPeriodicCoordinates1D: grid_extent must be > 0");
      a = (stop - start) / NT(grid_extent);
    }

    template <typename NT2>
    LinearPeriodicCoordinates1D(const LinearPeriodicCoordinates1D<NT2> &other)
        : LinearPeriodicCoordinates1D(other.size(), other.start, other.stop)
    {
    }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const { return start + a * x; }

    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /**
     * @brief Transform from the physical space to the grid, folded into [0, grid_extent)
     *
     * @param y physical coordinate
     * @return double grid coordinate
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

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    const NT start, stop;

    template <typename NT2>
    friend bool operator==(const LinearPeriodicCoordinates1D<NT> &lhs, const LinearPeriodicCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.grid_extent == rhs.grid_extent;
    }

    std::string to_string() const
    {
      return "LinearPeriodicCoordinates1D(" + std::to_string(grid_extent) + ", " + std::to_string(start) + ", " +
             std::to_string(stop) + ")";
    }

  private:
    const size_t grid_extent;
    const NT extent;
    NT a;
  };

  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class LogarithmicCoordinates1D
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 1;

    /// The map constants are computed in double whatever NT is, and only then rounded to NT.
    LogarithmicCoordinates1D(size_t grid_extent, double start, double stop, double bias)
        : start(start), stop(stop), bias(bias), grid_extent(grid_extent), parent_extent(grid_extent),
          gem1(grid_extent - 1.), gem1inv(1. / (grid_extent - 1.)), a(bias), b((stop - start) / std::expm1(bias)),
          c(start)
    {
      if (grid_extent == 0) throw std::runtime_error("LogarithmicCoordinates1D: grid_extent must be > 0");
    }

    template <typename NT2>
    LogarithmicCoordinates1D(const LogarithmicCoordinates1D<NT2> &other)
        : LogarithmicCoordinates1D(LogarithmicCoordinates1D(other.parent_size(), other.start, other.stop, other.bias),
                                   other.size())
    {
    }

    /**
     * @brief The first n points of this grid, as a grid of the same type.
     *
     * forward(i) of the result is bit-identical to forward(i) here for every i < n, and backward()
     * is unchanged, so a kernel mapped over prefix(n) fills exactly the first n entries it would
     * fill over the full grid. That is how to skip the upper end of a grid -- e.g. momenta p >> k
     * whose loops have decoupled -- without a different coordinate type, and hence without
     * regenerating kernels that take this type by name. (SubCoordinates is the general window,
     * but a different type.)
     *
     * The spacing is copied, not re-derived: a grid rebuilt from n points and the prefix's last
     * point would put its points elsewhere (by an ulp, or -- where the spacing depends on the
     * point count -- entirely). A prefix therefore keeps its parent's parameters and point count
     * (parent_size()), and to_string() renders it as "<parent>.prefix(n)", so the position caches
     * and HDF5 datasets keyed on it never confuse it with the parent or with a genuine n-point
     * grid. For a grid that is not a prefix, to_string() and operator== are unchanged.
     *
     * @param n number of points kept, 1 <= n <= size()
     */
    LogarithmicCoordinates1D prefix(size_t n) const
    {
      if (n < 1 || n > grid_extent) throw std::runtime_error("LogarithmicCoordinates1D::prefix: need 1 <= n <= size()");
      return LogarithmicCoordinates1D(*this, n);
    }

    /// Number of points of the grid this one is a prefix() of; size() unless it is a prefix.
    size_t KOKKOS_FORCEINLINE_FUNCTION parent_size() const { return parent_extent; }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const
    {
      using Kokkos::expm1;
      return b * expm1(a * x * gem1inv) + c;
    }

    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    NT KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const
    {
      using Kokkos::log1p;
      return log1p((y - c) / b) * gem1 / a;
    }

    NT KOKKOS_FORCEINLINE_FUNCTION backward_derivative(const NT &y) const { return 1. / (y - c) * gem1 / a; }

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    const NT start, stop, bias;

    template <typename NT2>
    friend bool operator==(const LogarithmicCoordinates1D<NT> &lhs, const LogarithmicCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.bias == rhs.bias && lhs.size() == rhs.size() &&
             lhs.parent_size() == rhs.parent_size();
    }

    std::string to_string() const
    {
      return "LogarithmicCoordinates1D(" + std::to_string(start) + ", " + std::to_string(stop) + ", " +
             std::to_string(bias) + ", " + std::to_string(parent_extent) + ")" + (grid_extent == parent_extent ? std::string() : ".prefix(" + std::to_string(grid_extent) + ")");
    }

  private:
    // prefix(): the parent's parameters and spacing -- gem1/gem1inv included, which is what fixes
    // the spacing here, since forward() divides the index by the PARENT's point count -- fewer points
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
   * @brief Logarithmic coordinates which cluster grid points around an interior scale.
   *
   * In contrast to LogarithmicCoordinates1D, whose single `bias` knob can only pack points towards
   * `start`, this coordinate spends its resolution around a freely chosen `center` and lets it fall
   * off smoothly to both sides. That is what is needed when the interesting structure sits at an
   * intermediate scale -- e.g. the ~1 GeV peak of a gluon dressing on a grid spanning 0.005 to 250
   * GeV -- while the deep IR and the deep UV are flat.
   *
   * The construction works in u = log(p) and stretches it with a sinh, the standard interior-point
   * clustering map. With u0 = log(center) and c = focus,
   *
   *     forward:  u(s) = u0 + sinh(s) / c,          s = s_min + a * i uniform in the index
   *     backward: s(u) = asinh(c * (u - u0))
   *
   * where s_min = s(log(start)) and a = (s(log(stop)) - s_min) / (grid_extent - 1). forward(0)
   * returns `start` exactly and forward(grid_extent - 1) returns `stop` to a few ulp, which matters
   * for the common idiom of pairing grid element 0 with `start` directly rather than going through
   * forward(). The node density is dindex/du ~ 1 / sqrt(1 + c^2 (u - u0)^2), i.e. peaked at
   * `center` and decaying like 1/(c |u - u0|), so resolution is redistributed smoothly rather than
   * cut off at the edge of a window.
   *
   * `focus == 0` reproduces a pure logarithmic (geometric) grid; larger values cluster harder. As a
   * calibration point, 64 points on [0.005, 250] with center = 1 and focus = 2 give a cell width of
   * 0.021 decades at the center and 32 points in [0.3, 3], against 0.076 decades and 13 points for
   * LogarithmicCoordinates1D with bias = 11 on the same interval.
   *
   * `center` outside [start, stop] is allowed and degenerates to a one-sided stretch.
   */
  template <typename NT = double>
    requires std::is_floating_point_v<NT>
  class FocusedLogCoordinates1D
  {
  public:
    using ctype = NT;
    static constexpr size_t dim = 1;

    /**
     * @brief Construct a new FocusedLogCoordinates1D object
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

    template <typename NT2>
    FocusedLogCoordinates1D(const FocusedLogCoordinates1D<NT2> &other)
        : FocusedLogCoordinates1D(
              FocusedLogCoordinates1D(other.parent_size(), other.start, other.stop, other.center, other.focus),
              other.size())
    {
    }

    /**
     * @brief The first n points of this grid, as a grid of the same type.
     *
     * forward(i) of the result is bit-identical to forward(i) here for every i < n, and backward()
     * is unchanged, so a kernel mapped over prefix(n) fills exactly the first n entries it would
     * fill over the full grid. That is how to skip the upper end of a grid -- e.g. momenta p >> k
     * whose loops have decoupled -- without a different coordinate type, and hence without
     * regenerating kernels that take this type by name. (SubCoordinates is the general window,
     * but a different type.)
     *
     * The spacing is copied, not re-derived: a grid rebuilt from n points and the prefix's last
     * point would put its points elsewhere (by an ulp, or -- where the spacing depends on the
     * point count -- entirely). A prefix therefore keeps its parent's parameters and point count
     * (parent_size()), and to_string() renders it as "<parent>.prefix(n)", so the position caches
     * and HDF5 datasets keyed on it never confuse it with the parent or with a genuine n-point
     * grid. For a grid that is not a prefix, to_string() and operator== are unchanged.
     *
     * @param n number of points kept, 1 <= n <= size()
     */
    FocusedLogCoordinates1D prefix(size_t n) const
    {
      if (n < 1 || n > grid_extent) throw std::runtime_error("FocusedLogCoordinates1D::prefix: need 1 <= n <= size()");
      return FocusedLogCoordinates1D(*this, n);
    }

    /// Number of points of the grid this one is a prefix() of; size() unless it is a prefix.
    size_t KOKKOS_FORCEINLINE_FUNCTION parent_size() const { return parent_extent; }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION from_linear_index(size_t i) const
    {
      return device::array<size_t, 1>{i};
    }

    /**
     * @brief Transform from the grid to the physical space
     *
     * @param x grid coordinate
     * @return NumberType physical coordinate
     */
    template <typename IT> NT KOKKOS_FORCEINLINE_FUNCTION forward(const IT &x) const
    {
      using Kokkos::exp, Kokkos::sinh;
      const NT s = s_min + a * x;
      // Factored as start * exp(...) rather than exp(log(start) + ...) so that x == 0 gives back
      // start bit-for-bit: the models pair grid element 0 with the p_grid_min parameter directly.
      // The upper end is then accurate to a handful of ulp, which nothing reads that way.
      //
      // The subtraction MUST happen before the multiplication by c_inv. Written the other way
      // round, `sinh(s) * c_inv - g_min` is a multiply-add, which the compiler is free to
      // contract into an FMA: the product is then kept at infinite precision and the ALREADY
      // ROUNDED g_min subtracted from it, so x == 0 yields that rounding error instead of zero
      // and start comes back off by an ulp. As written, `sinh(s) - g_min` is exactly zero at
      // x == 0 (identical argument, same function), and 0 * c_inv is exactly zero.
      const NT d = pure_log ? (s - g_min) : (sinh(s) - g_min) * c_inv;
      return start * exp(d);
    }

    template <typename IT> device::array<NT, 1> KOKKOS_FORCEINLINE_FUNCTION forward(const device::array<IT, 1> &x) const
    {
      return {forward(x[0])};
    }

    /**
     * @brief Transform from the physical space to the grid
     *
     * @param y physical coordinate
     * @return double grid coordinate
     */
    NT KOKKOS_FORCEINLINE_FUNCTION backward(const NT &y) const
    {
      using Kokkos::log;
      const NT u = log(y);
      return ((pure_log ? u : s_of_u(u, c, u0)) - s_min) * a_inv;
    }

    NT KOKKOS_FORCEINLINE_FUNCTION backward_derivative(const NT &y) const
    {
      using Kokkos::log, Kokkos::sqrt;
      if (pure_log) return a_inv / y;
      const NT v = c * (log(y) - u0);
      return a_inv * c / (sqrt(v * v + NT(1)) * y);
    }

    size_t KOKKOS_FORCEINLINE_FUNCTION size() const { return grid_extent; }

    device::array<size_t, 1> KOKKOS_FORCEINLINE_FUNCTION sizes() const { return {grid_extent}; }

    const NT start, stop, center, focus;

    template <typename NT2>
    friend bool operator==(const FocusedLogCoordinates1D<NT> &lhs, const FocusedLogCoordinates1D<NT2> &rhs)
    {
      if constexpr (!std::is_same_v<NT, NT2>) return false; // Different types, cannot be equal
      return lhs.start == rhs.start && lhs.stop == rhs.stop && lhs.center == rhs.center && lhs.focus == rhs.focus &&
             lhs.size() == rhs.size() && lhs.parent_size() == rhs.parent_size();
    }

    std::string to_string() const
    {
      return "FocusedLogCoordinates1D(" + std::to_string(parent_extent) + ", " + round_trip(start) + ", " +
             round_trip(stop) + ", " + round_trip(center) + ", " + round_trip(focus) + ")" + (grid_extent == parent_extent ? std::string() : ".prefix(" + std::to_string(grid_extent) + ")");
    }

  private:
    /**
     * @brief Format a parameter with the fewest digits that still identify it uniquely.
     *
     * to_string() is the identity key of a coordinate dataset in HDF5 output, and the key that
     * HDF5Input::load_map compares against, so two grids that differ must render differently.
     * std::to_string would not do: it emits six decimals, under which every start below 1e-6 --
     * a perfectly ordinary IR cutoff here -- renders as "0.000000" and distinct grids would
     * silently share one dataset. Widening the precision until the text parses back to the
     * original value keeps the common cases readable ("0.005", not "0.0050000000000000001") while
     * still separating values that genuinely differ.
     */
    static std::string round_trip(const NT value)
    {
      char buffer[40];
      for (int precision = 6; precision < std::numeric_limits<NT>::max_digits10; ++precision) {
        std::snprintf(buffer, sizeof(buffer), "%.*g", precision, double(value));
        if (NT(std::strtod(buffer, nullptr)) == value) return std::string(buffer);
      }
      std::snprintf(buffer, sizeof(buffer), "%.*g", std::numeric_limits<NT>::max_digits10, double(value));
      return std::string(buffer);
    }

    /**
     * @brief asinh(c * (u - u0)), written out to pick the cancellation-free branch.
     *
     * Naively log(v + sqrt(v^2 + 1)) cancels catastrophically for v << 0 -- these grids reach
     * v ~ -40, where it loses several digits. Using (v + D)(D - v) = 1 to rewrite the negative
     * branch as -log(D - v) makes both sides exact to round-off. Static and templated on the
     * precision, as the constructor evaluates it in double.
     */
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

  template <typename Coordinates> auto make_idx_grid(const Coordinates &coordinates) -> std::vector<double>
  {
    using ctype = typename Coordinates::ctype;
    using cortype = device::array<ctype, Coordinates::dim>;
    std::vector<cortype> grid(coordinates.size());
    for (size_t i = 0; i < coordinates.size(); ++i) {
      const auto forwarded = coordinates.from_linear_index(i);
      for (size_t j = 0; j < Coordinates::dim; ++j) {
        grid[i][j] = forwarded[j];
      }
    }
    return grid;
  }

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

  // Definitions of useful combined coordinates
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
