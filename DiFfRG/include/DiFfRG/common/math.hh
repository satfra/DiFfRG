#pragma once

// DiFfRG
#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/kokkos.hh>

// standard library
#include <cmath>

// external libraries
#include <autodiff/forward/real.hpp>
#include <type_traits>

namespace DiFfRG
{
  /**
   * @brief Type trait: true iff T is any autodiff::Real<N, U> specialization.
   * Allows generic handling of higher-order forward-mode AD types beyond
   * autodiff::real (= autodiff::Real<1, double>).
   */
  template <typename T> struct is_autodiff_real : std::false_type {};
  template <size_t N, typename U> struct is_autodiff_real<autodiff::Real<N, U>> : std::true_type {};
  template <typename T> inline constexpr bool is_autodiff_real_v = is_autodiff_real<T>::value;

  /**
   * @brief Finite-ness check for autodiff::real
   *
   * @param x Number to check
   * @return Whether x and its derivative are finite
   */
  template <size_t N, typename T> bool isfinite(const autodiff::Real<N, T> &x)
  {
    return std::isfinite(autodiff::val(x)) && std::isfinite(autodiff::derivative(x));
  }
  using std::isfinite;

  /**
   * @brief A compile-time evaluatable power function for whole number exponents
   *
   * @tparam n Exponent of type int
   * @tparam RF Type of argument
   * @param x Argument
   * @return x^n
   */
  template <int n, typename NumberType>
    requires requires(NumberType x) {
      x * x;
      NumberType(1.) / x;
    }
  constexpr KOKKOS_INLINE_FUNCTION NumberType powr(const NumberType x)
  {
    if constexpr (n == 0)
      return NumberType(1.);
    else if constexpr (n < 0)
      return NumberType(1.) / powr<-n, NumberType>(x);
    else if constexpr (n == 1)
      return x;
    else if constexpr (n % 2 == 0)
      return powr<n / 2>(x) * powr<n / 2>(x);
    else
      return powr<n / 2>(x) * powr<n / 2>(x) * x;
  }

  template <typename NumberType>
    requires std::is_integral_v<NumberType>
  constexpr KOKKOS_INLINE_FUNCTION NumberType factorial(const NumberType &x)
  {
    NumberType res = 1;
    for (NumberType i = 2; i <= x; ++i)
      res *= i;
    return res;
  }

  /**
   * @brief Volume of a d-dimensional sphere
   *
   * @tparam NT Type of the number
   * @param d Dimension of the sphere
   */
  template <typename NT> constexpr KOKKOS_INLINE_FUNCTION double V_d(NT d)
  {
    using std::pow;
    using std::tgamma;
    return pow(M_PI, d / 2.) / tgamma(d / 2. + 1.);
  }

  /**
   * @brief Volume of a d-dimensional sphere with extent
   *
   * @tparam NT1 Type of the number
   * @tparam NT2 Type of the extent
   * @param d Dimension of the sphere
   * @param extent Extent of the sphere
   */
  template <typename NT1, typename NT2> constexpr KOKKOS_INLINE_FUNCTION double V_d(NT1 d, NT2 extent)
  {
    using std::pow;
    using std::tgamma;
    return pow(M_PI, d / 2.) / tgamma(d / 2. + 1.) * pow(extent, d);
  }

  /**
   * @brief Surface of a d-dimensional sphere
   *
   * @tparam NT Type of the number
   * @param d Dimension of the sphere
   */
  template <typename NT> constexpr KOKKOS_INLINE_FUNCTION double S_d(NT d)
  {
    using std::pow;
    using std::tgamma;
    return 2. * pow(M_PI, d / 2.) / tgamma(d / 2.);
  }

  /**
   * @brief Surface of a d-dimensional sphere (precompiled)
   *
   * @tparam NT Type of the number
   * @param d Dimension of the sphere
   */
  template <typename NT> consteval NT S_d_prec(uint d)
  {
    if (d == 1)
      return 2;
    else if (d == 2)
      return static_cast<NT>(2) * static_cast<NT>(M_PI);
    else if (d == 3)
      return static_cast<NT>(4) * static_cast<NT>(M_PI);
    else if (d == 4)
      return static_cast<NT>(2) * powr<2>(static_cast<NT>(M_PI));
    else if (d == 5)
      return static_cast<NT>(8) * powr<2>(static_cast<NT>(M_PI)) / static_cast<NT>(3);
    else if (d == 6)
      return powr<3>(static_cast<NT>(M_PI));
    else if (d == 7)
      return static_cast<NT>(16) * powr<3>(static_cast<NT>(M_PI)) / static_cast<NT>(15);
    return std::numeric_limits<NT>::quiet_NaN();
  }

  /**
   * @brief A compile-time evaluatable theta function
   */
  template <typename NumberType>
    requires requires(NumberType x) { x >= 0; }
  constexpr KOKKOS_INLINE_FUNCTION auto heaviside_theta(const NumberType x)
  {
    if constexpr (std::is_same_v<NumberType, autodiff::real>)
      return x >= 0. ? 1. : 0.;
    else
      return x >= static_cast<NumberType>(0) ? static_cast<NumberType>(1) : static_cast<NumberType>(0);
  }

  /**
   * @brief A compile-time evaluatable sign function
   */
  template <typename NumberType>
    requires requires(NumberType x) { x >= 0; }
  constexpr KOKKOS_INLINE_FUNCTION auto sign(const NumberType x)
  {
    if constexpr (std::is_same_v<NumberType, autodiff::real>)
      return x >= 0. ? 1. : -1.;
    else
      return x >= static_cast<NumberType>(0) ? static_cast<NumberType>(1) : static_cast<NumberType>(-1);
  }

  /**
   * @brief Function to evaluate whether two floats are equal to numerical precision.
   * Tests for both relative and absolute equality.
   *
   * @param eps_ Precision with which to compare a and b
   * @return bool
   */
  template <typename T1, typename T2, typename T3>
    requires(std::is_floating_point<T1>::value || is_autodiff_real_v<T1> || is_complex<T1>::value) &&
            (std::is_floating_point<T2>::value || is_autodiff_real_v<T2> || is_complex<T2>::value) &&
            std::is_floating_point<T3>::value
  bool KOKKOS_INLINE_FUNCTION is_close(T1 a, T2 b, T3 eps_)
  {
    if constexpr (is_complex<T1>::value || is_complex<T2>::value) {
      return is_close(real(a), real(b), eps_) && is_close(imag(a), imag(b), eps_);
    } else if constexpr (is_autodiff_real_v<T1> || is_autodiff_real_v<T2>)
      return is_close((double)a, (double)b, (double)eps_);
    else {
      T1 diff = std::fabs(a - b);
      if (diff <= eps_) return true;
      if (diff <= std::fmax(std::fabs(a), std::fabs(b)) * eps_) return true;
    }
    return false;
  }

  /**
   * @brief Function to evaluate whether two floats are equal to numerical precision.
   * Tests for both relative and absolute equality.
   *
   * @return bool
   */
  template <typename T1, typename T2>
    requires(std::is_floating_point<T1>::value || is_autodiff_real_v<T1> || is_complex<T1>::value) &&
            (std::is_floating_point<T2>::value || is_autodiff_real_v<T2> || is_complex<T2>::value)
  bool KOKKOS_INLINE_FUNCTION is_close(T1 a, T2 b)
  {
    if constexpr (is_complex<T1>::value || is_complex<T2>::value) {
      return is_close(real(a), real(b)) && is_close(imag(a), imag(b));
    } else if constexpr (is_autodiff_real_v<T1> || is_autodiff_real_v<T2>) {
      constexpr auto eps_ = std::numeric_limits<double>::epsilon() * 10.;
      return is_close((double)a, (double)b, eps_);
    } else {
      constexpr auto eps_ = std::max(std::numeric_limits<T1>::epsilon(), std::numeric_limits<T2>::epsilon());
      return is_close(a, b, eps_);
    }
    return false;
  }

  /**
   * @brief A dot product which takes the dot product between a1 and a2, assuming each has n entries which can be
   * accessed via the [] operator.
   */
  template <uint n, typename NT, typename A1, typename A2>
    requires requires(A1 a1, A2 a2) { a1[0] * a2[0]; }
  NT dot(const A1 &a1, const A2 &a2)
  {
    NT ret = a1[0] * a2[0];
    for (uint i = 1; i < n; ++i)
      ret += a1[i] * a2[i];
    return ret;
  }

  namespace compute
  {
    using ::Kokkos::abs;
    using ::Kokkos::atan;
    using ::Kokkos::cos;
    using ::Kokkos::cosh;
    using ::Kokkos::exp;
    using ::Kokkos::imag;
    using ::Kokkos::log;
    using ::Kokkos::pow;
    using ::Kokkos::real;
    using ::Kokkos::sin;
    using ::Kokkos::sinh;
    using ::Kokkos::sqrt;
    using ::Kokkos::tan;
    using ::Kokkos::tanh;

    using ::Kokkos::fmax;
    using ::Kokkos::fmin;
    using ::Kokkos::max;
    using ::Kokkos::min;

    using ::Kokkos::abs;
    using ::Kokkos::fabs;

    using ::Kokkos::atan2;
    using ::Kokkos::fma;

    template <typename T1, typename T2, typename T3>
      requires(!std::is_arithmetic_v<T1> || !std::is_arithmetic_v<T2> || !std::is_arithmetic_v<T3>)
    constexpr KOKKOS_FORCEINLINE_FUNCTION auto fma(const T1 &a, const T2 &b, const T3 &c)
    {
      return a * b + c;
    }

    template <size_t N, typename T>
      requires std::is_arithmetic_v<T>
    constexpr KOKKOS_FORCEINLINE_FUNCTION T conj(const autodiff::Real<N, T> x)
    {
      return x;
    }

    template <size_t N, typename T>
      requires std::is_arithmetic_v<T>
    constexpr KOKKOS_FORCEINLINE_FUNCTION T conj(const cxReal<N, T> x)
    {
      cxReal<N, T> res;
      autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = Kokkos::conj(x[i]); });
      return res;
    }

    template <typename T>
      requires std::is_arithmetic_v<T>
    constexpr KOKKOS_FORCEINLINE_FUNCTION T conj(const T x)
    {
      return x;
    }

    template <typename T>
      requires is_complex<T>::value
    constexpr KOKKOS_FORCEINLINE_FUNCTION T conj(const T x)
    {
      return Kokkos::conj(x);
    }

    using DiFfRG::powr;

    template <typename NT> constexpr auto cot(const NT x) { return NT(1) / tan(x); }
    template <typename NT> constexpr auto coth(const NT x) { return NT(1) / tanh(x); }
  } // namespace compute
} // namespace DiFfRG