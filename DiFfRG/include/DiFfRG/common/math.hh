#pragma once

// DiFfRG
#include <DiFfRG/common/complex_math.hh>
#include <DiFfRG/common/kokkos.hh>

// standard library
#include <cmath>

// external libraries
#include <autodiff/forward/real.hpp>
#include <type_traits>

/**
 * @file math.hh
 * @brief Small math helpers usable in host and device code: integer powers, sphere volumes and areas, step and sign
 * functions, floating-point comparison, and the `compute::` namespace of math functions for kernels.
 */

namespace DiFfRG
{
  /**
   * @brief True if `T` is an `autodiff::Real<N, U>` (forward-mode derivative type of any order \f$N\f$).
   *
   * `autodiff::real` is `autodiff::Real<1, double>`.
   */
  template <typename T> struct is_autodiff_real : std::false_type {
  };
  template <size_t N, typename U> struct is_autodiff_real<autodiff::Real<N, U>> : std::true_type {
  };
  template <typename T> inline constexpr bool is_autodiff_real_v = is_autodiff_real<T>::value;

  /**
   * @brief Whether the value and the first derivative of `x` are finite.
   *
   * Higher derivatives (\f$N \geq 2\f$) are not checked.
   */
  template <size_t N, typename T> bool isfinite(const autodiff::Real<N, T> &x)
  {
    return std::isfinite(autodiff::val(x)) && std::isfinite(autodiff::derivative(x));
  }
  using std::isfinite;

  /**
   * @brief \f$x^n\f$ for an integer exponent \f$n\f$ known at compile time, by repeated squaring.
   *
   * Negative \f$n\f$ give \f$1/x^{|n|}\f$, and \f$x^0 = 1\f$.
   *
   * @tparam n the exponent.
   * @tparam NumberType type of the argument, e.g. `double`, complex or autodiff types.
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

  /// \f$x!\f$ for an integer \f$x\f$; \f$0! = 1\f$.
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
   * @brief Volume of the unit ball in \f$d\f$ dimensions, \f$V_d = \pi^{d/2} / \Gamma(d/2 + 1)\f$.
   *
   * E.g. \f$V_2 = \pi\f$, \f$V_3 = 4\pi/3\f$. Computed in double precision.
   *
   * @param d the dimension; need not be an integer.
   */
  template <typename NT> constexpr KOKKOS_INLINE_FUNCTION double V_d(NT d)
  {
    using std::pow;
    using std::tgamma;
    return pow(M_PI, d / 2.) / tgamma(d / 2. + 1.);
  }

  /**
   * @brief Volume \f$V_d\, r^d\f$ of the ball of radius \f$r\f$ = `extent` in \f$d\f$ dimensions.
   *
   * @param d the dimension; need not be an integer.
   * @param extent the radius \f$r\f$.
   */
  template <typename NT1, typename NT2> constexpr KOKKOS_INLINE_FUNCTION double V_d(NT1 d, NT2 extent)
  {
    using std::pow;
    using std::tgamma;
    return pow(M_PI, d / 2.) / tgamma(d / 2. + 1.) * pow(extent, d);
  }

  /**
   * @brief Area of the unit sphere in \f$\mathbb{R}^d\f$, \f$S_d = 2\pi^{d/2} / \Gamma(d/2)\f$.
   *
   * E.g. \f$S_2 = 2\pi\f$ (circumference of the unit circle), \f$S_3 = 4\pi\f$. This is the angular factor in
   * \f$\int d^dq\, f(|q|) = S_d \int_0^\infty dq\, q^{d-1} f(q)\f$. Computed in double precision.
   *
   * @param d the dimension of the embedding space; need not be an integer.
   */
  template <typename NT> constexpr KOKKOS_INLINE_FUNCTION double S_d(NT d)
  {
    using std::pow;
    using std::tgamma;
    return 2. * pow(M_PI, d / 2.) / tgamma(d / 2.);
  }

  /**
   * @brief S_d at compile time for integer \f$d\f$, in the number type `NT`.
   *
   * Uses \f$S_1 = 2\f$, \f$S_2 = 2\pi\f$ and \f$S_{d+2} = 2\pi S_d / d\f$. \f$d = 0\f$ is a compile error.
   * \f$\pi\f$ enters as the double-precision constant, so wider `NT` gain nothing beyond double precision.
   *
   * @tparam NT the result type.
   * @param d the dimension, \f$d \geq 1\f$.
   */
  template <typename NT> consteval NT S_d_prec(uint d)
  {
    // A throw is not a constant expression, so d = 0 fails to compile.
    if (d == 0) throw "S_d_prec: the dimension must be at least 1";
    NT result = d % 2 == 1 ? static_cast<NT>(2) : static_cast<NT>(2) * static_cast<NT>(M_PI);
    for (uint n = 2 - d % 2; n + 2 <= d; n += 2)
      result *= static_cast<NT>(2) * static_cast<NT>(M_PI) / static_cast<NT>(n);
    return result;
  }

  /// Step function \f$\Theta(x)\f$, with \f$\Theta(0) = 1\f$. Returns a plain number (no derivative) for
  /// `autodiff::real` input.
  template <typename NumberType>
    requires requires(NumberType x) { x >= 0; }
  constexpr KOKKOS_INLINE_FUNCTION auto heaviside_theta(const NumberType x)
  {
    if constexpr (std::is_same_v<NumberType, autodiff::real>)
      return x >= 0. ? 1. : 0.;
    else
      return x >= static_cast<NumberType>(0) ? static_cast<NumberType>(1) : static_cast<NumberType>(0);
  }

  /// Sign of \f$x\f$: \f$+1\f$ for \f$x \geq 0\f$ (including 0), \f$-1\f$ otherwise.
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
   * @brief Whether \f$a\f$ and \f$b\f$ agree to within \f$\epsilon\f$, absolutely or relatively.
   *
   * True if \f$|a - b| \leq \epsilon\f$ or \f$|a - b| \leq \epsilon \max(|a|, |b|)\f$. Complex numbers are compared
   * part by part; autodiff numbers by their value only.
   *
   * @param a, b the numbers to compare.
   * @param eps_ the tolerance \f$\epsilon\f$, used both as absolute and as relative tolerance.
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
   * @brief is_close() with \f$\epsilon\f$ the machine epsilon of the less precise of the two types.
   *
   * For autodiff numbers \f$\epsilon\f$ is 10 times the double-precision machine epsilon.
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
   * @brief \f$\sum_{i=0}^{n-1} a_1[i]\, a_2[i]\f$ for any two objects indexable by `[]`.
   *
   * @tparam n number of entries, \f$n \geq 1\f$.
   * @tparam NT result type.
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

  /**
   * @brief Math functions to use inside kernels: `using namespace DiFfRG::compute;` (or `compute::exp(x)`) picks
   * the overload that works on host and device for double, float, complex and autodiff arguments.
   *
   * Contains the Kokkos math functions (`exp`, `log`, `sqrt`, `pow`, trigonometric and hyperbolic functions, `abs`,
   * `min`, `max`, `fma`, `real`, `imag`, ...), `powr`, `conj` (identity for real types) and `cot`, `coth`.
   */
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
    constexpr KOKKOS_FORCEINLINE_FUNCTION autodiff::Real<N, T> conj(const autodiff::Real<N, T> x)
    {
      return x;
    }

    template <size_t N, typename T>
      requires std::is_arithmetic_v<T>
    constexpr KOKKOS_FORCEINLINE_FUNCTION cxReal<N, T> conj(const cxReal<N, T> x)
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

    /// \f$\cot x = 1/\tan x\f$.
    template <typename NT> constexpr auto cot(const NT x) { return NT(1) / tan(x); }
    /// \f$\coth x = 1/\tanh x\f$.
    template <typename NT> constexpr auto coth(const NT x) { return NT(1) / tanh(x); }
  } // namespace compute
} // namespace DiFfRG