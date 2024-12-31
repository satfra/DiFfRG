#pragma once

// external libraries
#include <autodiff/forward/real.hpp>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>

#ifdef USE_CUDA

#include <cuda/std/complex>

// Specialize isArithmetic for complex to make it compatible with real
namespace autodiff::detail
{
  template <typename T> struct ArithmeticTraits<::cuda::std::__4::complex<T>> : ArithmeticTraits<T> {
  };
} // namespace autodiff::detail
namespace DiFfRG
{
  using ::cuda::std::atan2;
  using ::cuda::std::complex;
  using ::cuda::std::cos;
  using ::cuda::std::imag;
  using ::cuda::std::log;
  using ::cuda::std::pow;
  using ::cuda::std::real;
  using ::cuda::std::sin;
  using ::cuda::std::sqrt;

  template <size_t N, typename T> using cxreal = autodiff::Real<N, complex<T>>;
  using cxReal = autodiff::Real<1, complex<double>>;
} // namespace DiFfRG

#else

#include <complex>

// Specialize isArithmetic for complex to make it compatible with real
namespace autodiff::detail
{
  template <typename T> struct ArithmeticTraits<::std::complex<T>> : ArithmeticTraits<T> {
  };
} // namespace autodiff::detail
namespace DiFfRG
{
  using ::std::atan2;
  using ::std::complex;
  using ::std::cos;
  using ::std::imag;
  using ::std::log;
  using ::std::pow;
  using ::std::real;
  using ::std::sin;
  using ::std::sqrt;

  template <size_t N, typename T> using cxreal = autodiff::Real<N, complex<T>>;
  using cxReal = autodiff::Real<1, complex<double>>;
} // namespace DiFfRG

#endif

namespace DiFfRG
{
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

  // operators for multiplication of float and complex
  template <typename T> AUTODIFF_DEVICE_FUNC auto operator*(const float &x, const complex<T> &y)
  {
    return y * static_cast<double>(x);
  }
  template <typename T> AUTODIFF_DEVICE_FUNC auto operator*(const complex<T> &x, const float &y)
  {
    return x * static_cast<double>(y);
  }
  // operators for addition of real and complex
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator+(const float &x, const complex<T> &y)
  {
    return y + static_cast<double>(x);
  }
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator+(const complex<T> &x, const float &y)
  {
    return x + static_cast<double>(y);
  }
  // operators for subtraction of real and complex
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator-(const float &x, const complex<T> &y)
  {
    return static_cast<double>(x) - y;
  }
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator-(const complex<T> &x, const float &y)
  {
    return x - static_cast<double>(y);
  }
  // operators for division of real and complex
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator/(const float &x, const complex<T> &y)
  {
    return static_cast<double>(x) / y;
  }
  template <size_t N, typename T> AUTODIFF_DEVICE_FUNC auto operator/(const complex<T> x, const float &y)
  {
    return x / static_cast<double>(y);
  }
} // namespace DiFfRG