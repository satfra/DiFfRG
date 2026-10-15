#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/types.hh>

// std
#include <cstddef>
#include <type_traits>

/**
 * @file multi_value.hh
 * @brief MultiValue, a fixed number of results that add and scale element-wise, so that one kernel can compute
 * several integrals at once.
 */

namespace DiFfRG
{
  /**
   * @brief \f$N\f$ values of type `T` that add and scale element-wise.
   *
   * A kernel returning `MultiValue<T, N>` computes \f$N\f$ integrals in one pass: an integrator instantiated with
   * `NT = MultiValue<T, N>` sums every component separately, and its map() writes component \f$j\f$ to the \f$j\f$-th
   * destination. This is meant for several projections of the same loop, which share everything but the
   * projection-specific part of the integrand.
   *
   * `T` is a real or complex number type; autodiff types are not supported.
   */
  template <typename T, size_t N> struct MultiValue {
    static_assert(N >= 1, "MultiValue needs at least one component");
    static_assert(!get_type::is_autodiff<T>, "MultiValue does not support autodiff types");

    using value_type = T;
    static constexpr size_t size = N;

    T v[N];

    /// All components zero.
    KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue() : v{} {}

    /// Every component set to @p s, so `MultiValue(0)` is the zero.
    KOKKOS_FORCEINLINE_FUNCTION explicit constexpr MultiValue(const T &s) : v{}
    {
      for (size_t i = 0; i < N; ++i)
        v[i] = s;
    }

    /// One value per component, each converted to `T`.
    template <typename... U>
      requires(sizeof...(U) == N && N > 1)
    KOKKOS_FORCEINLINE_FUNCTION explicit constexpr MultiValue(const U &...u) : v{static_cast<T>(u)...}
    {
    }

    /// Element-wise conversion, e.g. from single to double precision; implicit where `U` converts implicitly to `T`.
    template <typename U>
      requires(!std::is_same_v<U, T>)
    KOKKOS_FORCEINLINE_FUNCTION explicit(!std::is_convertible_v<U, T>) constexpr MultiValue(
        const MultiValue<U, N> &other)
        : v{}
    {
      for (size_t i = 0; i < N; ++i)
        v[i] = static_cast<T>(other.v[i]);
    }

    KOKKOS_FORCEINLINE_FUNCTION constexpr T &operator[](const size_t i) { return v[i]; }
    KOKKOS_FORCEINLINE_FUNCTION constexpr const T &operator[](const size_t i) const { return v[i]; }

    KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue &operator+=(const MultiValue &other)
    {
      for (size_t i = 0; i < N; ++i)
        v[i] += other.v[i];
      return *this;
    }
    KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue &operator-=(const MultiValue &other)
    {
      for (size_t i = 0; i < N; ++i)
        v[i] -= other.v[i];
      return *this;
    }
    template <typename S>
      requires requires(T t, S s) { t *= s; }
    KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue &operator*=(const S &s)
    {
      for (size_t i = 0; i < N; ++i)
        v[i] *= s;
      return *this;
    }
    template <typename S>
      requires requires(T t, S s) { t /= s; }
    KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue &operator/=(const S &s)
    {
      for (size_t i = 0; i < N; ++i)
        v[i] /= s;
      return *this;
    }
  };

  /// Whether `T` is a MultiValue.
  template <typename T> inline constexpr bool is_multi_value_v = false;
  template <typename T, size_t N> inline constexpr bool is_multi_value_v<MultiValue<T, N>> = true;

  namespace internal
  {
    template <typename T> struct multi_value_traits {
      static constexpr size_t size = 1;
      using element_type = T;
    };
    template <typename T, size_t N> struct multi_value_traits<MultiValue<T, N>> {
      static constexpr size_t size = N;
      using element_type = T;
    };
  } // namespace internal

  /// Number of components of a MultiValue; 1 for any other type.
  template <typename T> inline constexpr size_t multi_value_size_v = internal::multi_value_traits<T>::size;
  /// Component type of a MultiValue; the type itself for any other type.
  template <typename T> using multi_value_element_t = typename internal::multi_value_traits<T>::element_type;

  template <typename T, size_t N>
  KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue<T, N> operator+(MultiValue<T, N> a, const MultiValue<T, N> &b)
  {
    return a += b;
  }
  template <typename T, size_t N>
  KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue<T, N> operator-(MultiValue<T, N> a, const MultiValue<T, N> &b)
  {
    return a -= b;
  }
  template <typename T, size_t N> KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue<T, N> operator-(MultiValue<T, N> a)
  {
    for (size_t i = 0; i < N; ++i)
      a.v[i] = -a.v[i];
    return a;
  }
  template <typename T, size_t N, typename S>
    requires(!is_multi_value_v<S> && requires(T t, S s) { t *= s; })
  KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue<T, N> operator*(MultiValue<T, N> a, const S &s)
  {
    return a *= s;
  }
  template <typename T, size_t N, typename S>
    requires(!is_multi_value_v<S> && requires(T t, S s) { t *= s; })
  KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue<T, N> operator*(const S &s, MultiValue<T, N> a)
  {
    return a *= s;
  }
  template <typename T, size_t N, typename S>
    requires(!is_multi_value_v<S> && requires(T t, S s) { t /= s; })
  KOKKOS_FORCEINLINE_FUNCTION constexpr MultiValue<T, N> operator/(MultiValue<T, N> a, const S &s)
  {
    return a /= s;
  }

  namespace get_type::internal
  {
    template <typename T, size_t N> struct _ctype<MultiValue<T, N>> : _ctype<T> {
    };
    template <typename T, size_t N> struct _double_precision<MultiValue<T, N>> {
      using value = MultiValue<typename _double_precision<T>::value, N>;
    };
    template <typename T, size_t N> struct _single_precision<MultiValue<T, N>> {
      using value = MultiValue<typename _single_precision<T>::value, N>;
    };
  } // namespace get_type::internal
} // namespace DiFfRG

namespace Kokkos
{ // reduction identity must be defined in Kokkos namespace
  template <typename T, size_t N> struct reduction_identity<DiFfRG::MultiValue<T, N>> {
    KOKKOS_FORCEINLINE_FUNCTION static DiFfRG::MultiValue<T, N> sum() { return DiFfRG::MultiValue<T, N>(); }
  };
} // namespace Kokkos
