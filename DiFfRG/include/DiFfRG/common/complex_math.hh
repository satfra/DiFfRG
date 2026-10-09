#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>

// external libraries
#include <autodiff/common/numbertraits.hpp>

namespace DiFfRG
{
  using ::Kokkos::complex;

  using ::Kokkos::imag;
  using ::Kokkos::real;
} // namespace DiFfRG

// Specialize isArithmetic for complex to make it compatible with real
namespace autodiff::detail
{
  using ::Kokkos::complex;
  template <typename T> struct ArithmeticTraits<::Kokkos::complex<T>> : ArithmeticTraits<T> {
  };
} // namespace autodiff::detail

namespace autodiff::detail
{
  // operators for multiplication of float and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const T1 x, const complex<T2> y)
  {
    return complex<decltype(T1(1.) * T2(1.))>(x * y.real(), x * y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) * T2(1.))>(y * x.real(), y * x.imag());
  }
  // operators for addition of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x + y.real(), y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x.real() + y, x.imag());
  }
  // operators for subtraction of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) - T2(1.))>(x - y.real(), -y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) - T2(1.))>(x.real() - y, x.imag());
  }
  // operators for division of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) / T2(1.))>(x * y.real(), -x * y.imag()) / (powr<2>(y.real()) + powr<2>(y.imag()));
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const complex<T1> x, const T2 &y)
  {
    return complex<decltype(T1(1.) / T2(1.))>(x.real() / y, x.imag() / y);
  }
} // namespace autodiff::detail

// external libraries
#include <autodiff/forward/real.hpp>

namespace DiFfRG
{
  template <size_t N, typename T> using cxReal = autodiff::Real<N, complex<T>>;
  using cxreal = autodiff::Real<1, complex<double>>;

  template <typename T> struct is_complex : public std::false_type {
  };
  template <typename T> struct is_complex<complex<T>> : public std::true_type {
  };
  template <size_t N, typename T> struct is_complex<cxReal<N, T>> : public std::true_type {
  };

  template <size_t N, typename T> KOKKOS_FORCEINLINE_FUNCTION auto real(const autodiff::Real<N, T> &a) { return a; }
  template <size_t N, typename T> constexpr KOKKOS_FORCEINLINE_FUNCTION auto imag(const autodiff::Real<N, T> &)
  {
    return 0.;
  }

  template <size_t N, typename T> KOKKOS_FORCEINLINE_FUNCTION auto real(const cxReal<N, T> &x)
  {
    autodiff::Real<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = real(x[i]); });
    return res;
  }
  template <size_t N, typename T> KOKKOS_FORCEINLINE_FUNCTION auto imag(const cxReal<N, T> &x)
  {
    autodiff::Real<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = imag(x[i]); });
    return res;
  }

  // operators for multiplication of real and complex
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const autodiff::Real<N, T> &x, const complex<double> &y)
  {
    cxReal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x[i] * y; });
    return res;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const complex<double> &x, const autodiff::Real<N, T> &y)
  {
    cxReal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x * y[i]; });
    return res;
  }
  // operators for addition of real and complex
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const autodiff::Real<N, T> &x, const complex<double> &y)
  {
    cxReal<N, T> res;
    res[0] = x[0] + y;
    autodiff::detail::For<1, N + 1>([&](auto i) constexpr { res[i] = x[i]; });
    return res;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const complex<double> &x, const autodiff::Real<N, T> &y)
  {
    cxReal<N, T> res;
    res[0] = x + y[0];
    autodiff::detail::For<1, N + 1>([&](auto i) constexpr { res[i] = y[i]; });
    return res;
  }
  // operators for subtraction of real and complex
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const autodiff::Real<N, T> &x, const complex<double> &y)
  {
    cxReal<N, T> res;
    res[0] = x[0] - y;
    autodiff::detail::For<1, N + 1>([&](auto i) constexpr { res[i] = x[i]; });
    return res;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const complex<double> &x, const autodiff::Real<N, T> &y)
  {
    cxReal<N, T> res;
    res[0] = x - y[0];
    autodiff::detail::For<1, N + 1>([&](auto i) constexpr { res[i] = -y[i]; });
    return res;
  }
  // operators for division of real and complex
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const autodiff::Real<N, T> &x, const complex<double> &y)
  {
    cxReal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x[i] / y; });
    return res;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const complex<double> &x, const autodiff::Real<N, T> &y)
  {
    cxReal<N, T> ux(x);
    cxReal<N, T> uy(y);
    return ux /= uy;
  }

  // operators for multiplication of real and cxReal
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const autodiff::Real<N, T> &x, const cxReal<N, T> &y)
  {
    cxReal<N, T> res(x);
    return res *= y;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const cxReal<N, T> &x, const autodiff::Real<N, T> &y)
  {
    cxReal<N, T> res(y);
    return res *= x;
  }
  // operators for addition of real and cxReal
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const autodiff::Real<N, T> &x, const cxReal<N, T> &y)
  {
    cxReal<N, T> res(x);
    return res += y;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const cxReal<N, T> &x, const autodiff::Real<N, T> &y)
  {
    cxReal<N, T> res(y);
    return res += x;
  }
  // operators for subtraction of real and cxReal
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const autodiff::Real<N, T> &x, const cxReal<N, T> &y)
  {
    cxReal<N, T> res(x);
    return res -= y;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const cxReal<N, T> &x, const autodiff::Real<N, T> &y)
  {
    cxReal<N, T> ux(x);
    cxReal<N, T> uy(y);
    return ux -= uy;
  }
  // operators for division of real and cxReal
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const autodiff::Real<N, T> &x, const cxReal<N, T> &y)
  {
    cxReal<N, T> ux(x);
    return ux /= y;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const cxReal<N, T> &x, const autodiff::Real<N, T> &y)
  {
    cxReal<N, T> uy(y);
    return x / uy;
  }

  // operators for multiplication of complex and cxReal
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const complex<double> &x, const cxReal<N, T> &y)
  {
    cxReal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = y[i] * x; });
    return res;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const cxReal<N, T> &x, const complex<double> &y)
  {
    cxReal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x[i] * y; });
    return res;
  }
  // operators for addition of complex and cxReal
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const complex<double> &x, const cxReal<N, T> &y)
  {
    cxReal<N, T> res(y);
    res[0] += x;
    return res;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const cxReal<N, T> &x, const complex<double> &y)
  {
    cxReal<N, T> res(x);
    res[0] += y;
    return res;
  }
  // operators for subtraction of complex and cxReal
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const complex<double> &x, const cxReal<N, T> &y)
  {
    cxReal<N, T> ux(x);
    return ux -= y;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const cxReal<N, T> &x, const complex<double> &y)
  {
    cxReal<N, T> res(x);
    res[0] -= y;
    return res;
  }
  // operators for division of complex and cxReal
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const cxReal<N, T> &x, const complex<double> &y)
  {
    cxReal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x[i] / y; });
    return res;
  }
  template <size_t N, typename T>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const complex<double> &x, const cxReal<N, T> &y)
  {
    cxReal<N, T> ux(x);
    cxReal<N, T> uy(y);
    return ux /= uy;
  }

  // operators for division of double and cxReal
  template <size_t N, typename T> KOKKOS_FORCEINLINE_FUNCTION auto operator/(const double x, const cxReal<N, T> &y)
  {
    cxReal<N, T> res;
    res[0] = x;
    autodiff::detail::For<N + 1>([&](auto i) constexpr {
      res[i] -= autodiff::detail::Sum<0, i>([&](auto j) constexpr {
        constexpr auto c = autodiff::detail::BinomialCoefficient<i.index, j.index>;
        return c * res[j] * y[i - j];
      });
      res[i] /= y[0];
    });
    return res;
  }

  // ------------------------------------------------------------------
  // operators for complex<T1> and T2
  // ------------------------------------------------------------------

  // operators for multiplication of float and complex
  /*
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const T1 x, const complex<T2> y)
  {
    return complex<decltype(T1(1.) * T2(1.))>(x * y.real(), x * y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator*(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) * T2(1.))>(y * x.real(), y * x.imag());
  }
  // operators for addition of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x + y.real(), y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator+(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x.real() + y, x.imag());
  }
  // operators for subtraction of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x - y.real(), -y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator-(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x.real() - y, x.imag());
  }
  // operators for division of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x * y.real(), -x * y.imag()) / (powr<2>(y.real()) + powr<2>(y.imag()));
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  KOKKOS_FORCEINLINE_FUNCTION auto operator/(const complex<T1> x, const T2 &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x.real() / y, x.imag() / y);
  }
*/
} // namespace DiFfRG