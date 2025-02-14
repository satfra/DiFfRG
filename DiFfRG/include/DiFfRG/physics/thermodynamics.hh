#pragma once

// standard library
#include <cmath>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  using std::cosh, std::sinh, std::tanh, std::exp, std::expm1;

  // ----------------------------------------------------------------------------------------------------
  // For convenience, all hyperbolic functions
  // ----------------------------------------------------------------------------------------------------
  template <typename T> auto __forceinline__ __device__ __host__ Cosh(const T x) { return cosh(x); }
  template <typename T> auto __forceinline__ __device__ __host__ Sinh(const T x) { return sinh(x); }
  template <typename T> auto __forceinline__ __device__ __host__ Tanh(const T x) { return tanh(x); }
  template <typename T> auto __forceinline__ __device__ __host__ Coth(const T x) { return 1. / tanh(x); }
  template <typename T> auto __forceinline__ __device__ __host__ Sech(const T x) { return 1. / cosh(x); }
  template <typename T> auto __forceinline__ __device__ __host__ Csch(const T x) { return 1. / sinh(x); }

  // ----------------------------------------------------------------------------------------------------
  // Finite temperature hyperbolic functions
  // ----------------------------------------------------------------------------------------------------

  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto __forceinline__ __device__ __host__ CothFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Coth(x / (2 * T)));
    return is_close(T / x, T2(0)) ? (R)(real(x) < 0 ? -1 : 1) : Coth(x / (2 * T));
  }
  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto __forceinline__ __device__ __host__ TanhFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Tanh(x / (2 * T)));
    return is_close(T / x, T2(0)) ? (R)(real(x) < 0 ? -1 : 1) : Tanh(x / (2 * T));
  }
  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto __forceinline__ __device__ __host__ SechFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Cosh(x / (2 * T)));
    const auto res = Cosh(x / (2 * T));
    return !isfinite(res) ? (R)0 : (R)(1) / res;
  }
  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto __forceinline__ __device__ __host__ CschFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Sinh(x / (2 * T)));
    const auto res = Sinh(x / (2 * T));
    return !isfinite(res) ? (R)0 : (R)(1) / res;
  }

  // ----------------------------------------------------------------------------------------------------
  // Thermodynamic hyperbolic functions
  // ----------------------------------------------------------------------------------------------------
  // coth(e/2T)
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ cothS(const T1 e, const T2 T)
  {
    return 1. / tanh(e / (T * 2.));
  }
  // d/de cothS
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ dcothS(const T1 e, const T2 T)
  {
    return -1. / powr<2>(sinh(e / (T * 2.))) / (2. * T);
  }
  // d^2/de^2 cothS
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ ddcothS(const T1 e, const T2 T)
  {
    return -dcothS(e, T) * cothS(e, T) / T;
  }
  // d^3/de^3 cothS
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ dddcothS(const T1 e, const T2 T)
  {
    return -(ddcothS(e, T) * cothS(e, T) + powr<2>(dcothS(e, T)));
  }
  // tanh(e/2T)
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ tanhS(const T1 e, const T2 T)
  {
    return tanh(e / (T * 2.));
  }
  // d/de tanhS
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ dtanhS(const T1 e, const T2 T)
  {
    return 1. / powr<2>(cosh(e / (T * 2.))) / (2. * T);
  }
  // d^2/de^2 tanhS
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ ddtanhS(const T1 e, const T2 T)
  {
    return -dtanhS(e, T) * tanhS(e, T) / T;
  }
  // sech(e/2T)
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ sechS(const T1 e, const T2 T)
  {
    return 1. / cosh(e / (T * 2.));
  }
  // csch(e/2T)
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ cschS(const T1 e, const T2 T)
  {
    return 1. / sinh(e / (T * 2.));
  }

  // ----------------------------------------------------------------------------------------------------
  // Distribution Functions nB, nF and their Derivatives
  // ----------------------------------------------------------------------------------------------------
  // Bosonic distribution
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ nB(const T1 e, const T2 T)
  {
    return 1. / expm1(e / T);
  }
  // Derivative d/de of the bosonic distribution
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ dnB(const T1 e, const T2 T)
  {
    return -exp(e / T) / powr<2>(expm1(e / T)) / T;
  }
  // Derivative d²/de² of the bosonic distribution
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ ddnB(const T1 e, const T2 T)
  {
    return exp(e / T) * (1. + exp(e / T)) / powr<3>(expm1(e / T)) / powr<2>(T);
  }
  // Fermionic distribution
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ nF(const T1 e, const T2 T)
  {
    return 1. / (exp(e / T) + 1.);
  }
  // Derivative d/de of the fermionic distribution
  template <typename T1, typename T2> auto __forceinline__ __device__ __host__ dnF(const T1 e, const T2 T)
  {
    return -exp(e / T) / powr<2>(exp(e / T) + 1.) / T;
  }
} // namespace DiFfRG