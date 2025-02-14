#pragma once

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>

// external libraries
#include <autodiff/common/numbertraits.hpp>

#ifdef USE_CUDA

#include <cuda/std/complex>

namespace DiFfRG
{
  using ::cuda::std::atan2;
  using ::cuda::std::complex;
  using ::cuda::std::cos;
  using ::cuda::std::cosh;
  using ::cuda::std::imag;
  using ::cuda::std::log;
  using ::cuda::std::pow;
  using ::cuda::std::real;
  using ::cuda::std::sin;
  using ::cuda::std::sinh;
  using ::cuda::std::sqrt;
  using ::cuda::std::tan;
  using ::cuda::std::tanh;
  template <typename NT> constexpr auto cot(const NT x) { return NT(1) / tan(x); }

} // namespace DiFfRG

#else

#include <complex>
namespace DiFfRG
{
  using ::std::atan2;
  using ::std::complex;
  using ::std::cos;
  using ::std::cosh;
  using ::std::imag;
  using ::std::log;
  using ::std::pow;
  using ::std::real;
  using ::std::sin;
  using ::std::sinh;
  using ::std::sqrt;
  using ::std::tan;
  using ::std::tanh;
  template <typename NT> constexpr auto cot(const NT x) { return NT(1) / tan(x); }
} // namespace DiFfRG

#endif

#ifdef USE_CUDA

// Specialize isArithmetic for complex to make it compatible with real
namespace autodiff::detail
{
  using ::cuda::std::complex;
  template <typename T> struct ArithmeticTraits<::cuda::std::__4::complex<T>> : ArithmeticTraits<T> {
  };
} // namespace autodiff::detail

#else

// Specialize isArithmetic for complex to make it compatible with real
namespace autodiff::detail
{
  using ::std::complex;
  template <typename T> struct ArithmeticTraits<::std::complex<T>> : ArithmeticTraits<T> {
  };
} // namespace autodiff::detail

#endif

namespace autodiff::detail
{
  // operators for multiplication of float and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  __host__ __device__ auto operator*(const T1 x, const complex<T2> y)
  {
    return complex<decltype(T1(1.) * T2(1.))>(x * y.real(), x * y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  __host__ __device__ auto operator*(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) * T2(1.))>(y * x.real(), y * x.imag());
  }
  // operators for addition of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  __host__ __device__ auto operator+(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x + y.real(), y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  __host__ __device__ auto operator+(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x.real() + y, x.imag());
  }
  // operators for subtraction of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  __host__ __device__ auto operator-(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) - T2(1.))>(x - y.real(), -y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  __host__ __device__ auto operator-(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) - T2(1.))>(x.real() - y, x.imag());
  }
  // operators for division of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  __host__ __device__ auto operator/(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) / T2(1.))>(x * y.real(), -x * y.imag()) / (powr<2>(y.real()) + powr<2>(y.imag()));
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  __host__ __device__ auto operator/(const complex<T1> x, const T2 &y)
  {
    return complex<decltype(T1(1.) / T2(1.))>(x.real() / y, x.imag() / y);
  }
} // namespace autodiff::detail

// external libraries
#include <autodiff/forward/real.hpp>

namespace DiFfRG
{
  template <size_t N, typename T> using cxreal = autodiff::Real<N, complex<T>>;
  using cxReal = autodiff::Real<1, complex<double>>;

  template <typename T> struct is_complex : public std::false_type {
  };
  template <typename T> struct is_complex<complex<T>> : public std::true_type {
  };
  template <size_t N, typename T> struct is_complex<cxreal<N, T>> : public std::true_type {
  };

  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto real(const autodiff::Real<N, T> &a) { return a; }
  template <size_t N, typename T> constexpr AUTODIFF_DEVICE_FUNC auto imag(const autodiff::Real<N, T> &) { return 0.; }

  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto real(const cxreal<N, T> &x)
  {
    autodiff::Real<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = real(x[i]); });
    return res;
  }
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto imag(const cxreal<N, T> &x)
  {
    autodiff::Real<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = imag(x[i]); });
    return res;
  }

  // operators for multiplication of real and complex
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator*(const autodiff::Real<N, T> &x, const complex<double> &y)
  {
    cxreal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x[i] * y; });
    return res;
  }
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator*(const complex<double> x, const autodiff::Real<N, T> y)
  {
    cxreal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x * y[i]; });
    return res;
  }
  // operators for addition of real and complex
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator+(const autodiff::Real<N, T> &x, const complex<double> &y)
  {
    cxreal<N, T> res;
    res[0] = x[0] + y;
    autodiff::detail::For<1, N + 1>([&](auto i) constexpr { res[i] = x[i]; });
    return res;
  }
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator+(const complex<double> x, const autodiff::Real<N, T> y)
  {
    cxreal<N, T> res;
    res[0] = x + y[0];
    autodiff::detail::For<1, N + 1>([&](auto i) constexpr { res[i] = y[i]; });
    return res;
  }
  // operators for subtraction of real and complex
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator-(const autodiff::Real<N, T> &x, const complex<double> &y)
  {
    cxreal<N, T> res;
    res[0] = x[0] - y;
    autodiff::detail::For<1, N + 1>([&](auto i) constexpr { res[i] = x[i]; });
    return res;
  }
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator-(const complex<double> x, const autodiff::Real<N, T> y)
  {
    cxreal<N, T> res;
    res[0] = x - y[0];
    autodiff::detail::For<1, N + 1>([&](auto i) constexpr { res[i] = -y[i]; });
    return res;
  }
  // operators for division of real and complex
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator/(const autodiff::Real<N, T> &x, const complex<double> &y)
  {
    cxreal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x[i] / y; });
    return res;
  }
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator/(const complex<double> x, const autodiff::Real<N, T> y)
  {
    cxreal<N, T> ux(x);
    cxreal<N, T> uy(y);
    return ux /= uy;
  }

  // operators for multiplication of real and cxreal
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator*(const autodiff::Real<N, T> &x, const cxreal<N, T> &y)
  {
    cxreal<N, T> res(x);
    return res *= y;
  }
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator*(const cxreal<N, T> &x, const autodiff::Real<N, T> &y)
  {
    cxreal<N, T> res(y);
    return res *= x;
  }
  // operators for addition of real and cxreal
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator+(const autodiff::Real<N, T> &x, const cxreal<N, T> &y)
  {
    cxreal<N, T> res(x);
    return res += y;
  }
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator+(const cxreal<N, T> &x, const autodiff::Real<N, T> &y)
  {
    cxreal<N, T> res(y);
    return res += x;
  }
  // operators for subtraction of real and cxreal
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator-(const autodiff::Real<N, T> &x, const cxreal<N, T> &y)
  {
    cxreal<N, T> res(x);
    return res -= y;
  }
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator-(const cxreal<N, T> &x, const autodiff::Real<N, T> &y)
  {
    cxreal<N, T> ux(x);
    cxreal<N, T> uy(y);
    return ux -= uy;
  }
  // operators for division of real and cxreal
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator/(const autodiff::Real<N, T> &x, const cxreal<N, T> &y)
  {
    cxreal<N, T> ux(x);
    return ux /= y;
  }
  template <size_t N, typename T>
  AUTODIFF_DEVICE_FUNC auto operator/(const cxreal<N, T> &x, const autodiff::Real<N, T> &y)
  {
    cxreal<N, T> uy(y);
    return x / uy;
  }

  // operators for multiplication of complex and cxreal
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator*(const complex<double> &x, const cxreal<N, T> &y)
  {
    cxreal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = y[i] * x; });
    return res;
  }
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator*(const cxreal<N, T> &x, const complex<double> &y)
  {
    cxreal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x[i] * y; });
    return res;
  }
  // operators for addition of complex and cxreal
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator+(const complex<double> &x, const cxreal<N, T> &y)
  {
    cxreal<N, T> res(y);
    res[0] += x;
    return res;
  }
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator+(const cxreal<N, T> &x, const complex<double> &y)
  {
    cxreal<N, T> res(x);
    res[0] += y;
    return res;
  }
  // operators for subtraction of complex and cxreal
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator-(const complex<double> &x, const cxreal<N, T> &y)
  {
    cxreal<N, T> ux(x);
    return ux -= y;
  }
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator-(const cxreal<N, T> &x, const complex<double> &y)
  {
    cxreal<N, T> res(x);
    res[0] -= y;
    return res;
  }
  // operators for division of complex and cxreal
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator/(const cxreal<N, T> &x, const complex<double> &y)
  {
    cxreal<N, T> res;
    autodiff::detail::For<0, N + 1>([&](auto i) constexpr { res[i] = x[i] / y; });
    return res;
  }
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator/(const complex<double> &x, const cxreal<N, T> &y)
  {
    cxreal<N, T> ux(x);
    cxreal<N, T> uy(y);
    return ux /= uy;
  }

  // operators for division of double and cxreal
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator/(const double x, const cxreal<N, T> &y)
  {
    cxreal<N, T> res;
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
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  AUTODIFF_DEVICE_FUNC auto operator*(const T1 x, const complex<T2> y)
  {
    return complex<decltype(T1(1.) * T2(1.))>(x * y.real(), x * y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  AUTODIFF_DEVICE_FUNC auto operator*(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) * T2(1.))>(y * x.real(), y * x.imag());
  }
  // operators for addition of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  AUTODIFF_DEVICE_FUNC auto operator+(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x + y.real(), y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  AUTODIFF_DEVICE_FUNC auto operator+(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x.real() + y, x.imag());
  }
  // operators for subtraction of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  AUTODIFF_DEVICE_FUNC auto operator-(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x - y.real(), -y.imag());
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  AUTODIFF_DEVICE_FUNC auto operator-(const complex<T1> &x, const T2 &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x.real() - y, x.imag());
  }
  // operators for division of real and complex
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  AUTODIFF_DEVICE_FUNC auto operator/(const T1 &x, const complex<T2> &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x * y.real(), -x * y.imag()) / (powr<2>(y.real()) + powr<2>(y.imag()));
  }
  template <typename T1, typename T2>
    requires(!std::is_same_v<T1, T2>) && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2>
  AUTODIFF_DEVICE_FUNC auto operator/(const complex<T1> x, const T2 &y)
  {
    return complex<decltype(T1(1.) + T2(1.))>(x.real() / y, x.imag() / y);
  }
} // namespace DiFfRG