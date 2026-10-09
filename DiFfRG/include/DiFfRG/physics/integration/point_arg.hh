#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/types.hh>

// external libraries
#include <autodiff/forward/real.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief A host array of per-point values: a column of a batch, a batch output, or the destination and
   * per-point arguments of map_points(). Does not own its data.
   *
   * `PointSpan<const T>` is read-only, `PointSpan<T>` writable; the latter converts to the former.
   */
  template <typename T> class PointSpan
  {
  public:
    using value_type = std::remove_const_t<T>;

    PointSpan() = default;
    PointSpan(T *data, const size_t size) : m_data(data), m_size(size) {}
    /// A view of all of @p v.
    PointSpan(std::vector<value_type> &v) : m_data(v.data()), m_size(v.size()) {}
    PointSpan(const std::vector<value_type> &v)
      requires std::is_const_v<T>
        : m_data(v.data()), m_size(v.size())
    {
    }
    /// The read-only view of a writable span. A template, so that it is not the copy constructor of PointSpan<T>.
    template <typename U>
      requires(std::is_const_v<T> && std::is_same_v<U, value_type>)
    PointSpan(const PointSpan<U> &other) : m_data(other.data()), m_size(other.size())
    {
    }

    T &operator[](const size_t i) const { return m_data[i]; }
    T *data() const { return m_data; }
    size_t size() const { return m_size; }
    T *begin() const { return m_data; }
    T *end() const { return m_data + m_size; }

  private:
    T *m_data = nullptr;
    size_t m_size = 0;
  };

  /**
   * @brief One kernel argument of map_points(): either a single value shared by all points, or one value per
   * point.
   *
   * Implicitly constructible from a value (shared), and from a PointSpan or std::vector (per point), so a
   * map_points() signature can take every argument as a `const PointArg<T> &` and the caller decides per call
   * which arguments vary. Values and arrays of another type convertible to T are converted, e.g. a double
   * column passed to an argument that is an AD number; a converted array is copied.
   */
  template <typename T> class PointArg
  {
  public:
    PointArg(const T &value) : value(value) {}
    template <typename U>
      requires(!std::is_same_v<std::remove_cvref_t<U>, T> && std::is_convertible_v<const U &, T>)
    PointArg(const U &value) : value(T(value))
    {
    }
    PointArg(const PointSpan<const T> &span) : values(span.data()), size(span.size()) {}
    PointArg(const PointSpan<T> &span) : values(span.data()), size(span.size()) {}
    PointArg(const std::vector<T> &v) : values(v.data()), size(v.size()) {}
    template <typename U>
      requires(!std::is_same_v<std::remove_const_t<U>, T> && std::is_convertible_v<const U &, T>)
    PointArg(const PointSpan<U> &span)
        : owned(std::make_shared<std::vector<T>>(span.begin(), span.end())), values(owned->data()), size(span.size())
    {
    }
    template <typename U>
      requires(!std::is_same_v<U, T> && std::is_convertible_v<const U &, T>)
    PointArg(const std::vector<U> &v) : PointArg(PointSpan<const U>(v))
    {
    }

    bool per_point() const { return values != nullptr; }
    const T &operator[](const size_t i) const { return values != nullptr ? values[i] : value; }

    /// Throws unless the argument is shared or holds exactly @p n values.
    void check_size(const size_t n) const
    {
      if (per_point() && size != n)
        throw std::runtime_error("map_points: a per-point argument holds " + std::to_string(size) + " values for " +
                                 std::to_string(n) + " points.");
    }

  private:
    std::shared_ptr<const std::vector<T>> owned;

  public:
    const T *values = nullptr;
    size_t size = 0;
    T value{};
  };

  namespace internal
  {
    /// The value type of a map_points() argument: T for a PointArg<T>, a PointSpan<T> or a std::vector<T>,
    /// otherwise the argument's own type.
    template <typename A> struct point_arg_value {
      using type = A;
    };
    template <typename T> struct point_arg_value<PointArg<T>> {
      using type = T;
    };
    template <typename T> struct point_arg_value<PointSpan<T>> {
      using type = std::remove_const_t<T>;
    };
    template <typename T> struct point_arg_value<std::vector<T>> {
      using type = T;
    };
    template <typename A> using point_arg_value_t = typename point_arg_value<std::remove_cvref_t<A>>::type;

    // get_type::single_precision, extended to the integrators' autodiff scalars
    template <typename T> struct _single_precision {
      using value = get_type::single_precision<T>;
    };
    template <size_t N> struct _single_precision<autodiff::Real<N, double>> {
      using value = autodiff::Real<N, float>;
    };

    /**
     * @brief The type a map_points() argument of type T is evaluated in by an integrator computing
     * in ctype: a single-precision integrator receives single-precision arguments, so its kernel does
     * not silently promote back to double.
     */
    template <typename T, typename ctype>
    using compute_arg_t = std::conditional_t<std::is_same_v<ctype, float>, typename _single_precision<T>::value, T>;

    /**
     * @brief Device-side view of a PointArg: a pointer into staged per-point values, or the broadcast
     * value. Broadcast values are returned by reference, so an interpolator passed as a shared
     * argument is not copied per thread.
     */
    template <typename T> struct DevicePointArg {
      const T *values;
      T value;

      KOKKOS_FORCEINLINE_FUNCTION const T &operator()(const size_t i) const
      {
        return values != nullptr ? values[i] : value;
      }
    };
  } // namespace internal
} // namespace DiFfRG
