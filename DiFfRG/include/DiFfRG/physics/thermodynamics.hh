#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

// standard library
#include <cmath>

namespace DiFfRG
{

  // ----------------------------------------------------------------------------------------------------
  // For convenience, all hyperbolic functions
  // ----------------------------------------------------------------------------------------------------
  namespace internal
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    template <typename T> auto KOKKOS_FORCEINLINE_FUNCTION Cosh(const T x) { return cosh(x); }
    template <typename T> auto KOKKOS_FORCEINLINE_FUNCTION Sinh(const T x) { return sinh(x); }
    template <typename T> auto KOKKOS_FORCEINLINE_FUNCTION Tanh(const T x) { return tanh(x); }
    template <typename T> auto KOKKOS_FORCEINLINE_FUNCTION Coth(const T x) { return 1. / tanh(x); }
    template <typename T> auto KOKKOS_FORCEINLINE_FUNCTION Sech(const T x) { return 1. / cosh(x); }
    template <typename T> auto KOKKOS_FORCEINLINE_FUNCTION Csch(const T x) { return 1. / sinh(x); }
  } // namespace internal

  using internal::Cosh, internal::Sinh, internal::Tanh, internal::Coth, internal::Sech, internal::Csch;

  // ----------------------------------------------------------------------------------------------------
  // Finite temperature hyperbolic functions
  // ----------------------------------------------------------------------------------------------------

  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto KOKKOS_FORCEINLINE_FUNCTION CothFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Coth(x / (2 * T)));
    return is_close(T / x, T2(0)) ? (R)(real(x) < 0 ? -1 : 1) : Coth(x / (2 * T));
  }
  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto KOKKOS_FORCEINLINE_FUNCTION TanhFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Tanh(x / (2 * T)));
    return is_close(T / x, T2(0)) ? (R)(real(x) < 0 ? -1 : 1) : Tanh(x / (2 * T));
  }
  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto KOKKOS_FORCEINLINE_FUNCTION SechFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Cosh(x / (2 * T)));
    const auto res = Cosh(x / (2 * T));
    return !isfinite(res) ? (R)0 : (R)(1) / res;
  }
  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto KOKKOS_FORCEINLINE_FUNCTION CschFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Sinh(x / (2 * T)));
    const auto res = Sinh(x / (2 * T));
    return !isfinite(res) ? (R)0 : (R)(1) / res;
  }

  // ----------------------------------------------------------------------------------------------------
  // Thermodynamic hyperbolic functions
  // ----------------------------------------------------------------------------------------------------
  // coth(e/2T)
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION cothS(const T1 e, const T2 T)
  {
    return Coth(e / (T * 2.));
  }
  // d/de cothS
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dcothS(const T1 e, const T2 T)
  {
    return -1. / powr<2>(Sinh(e / (T * 2.))) / (2. * T);
  }
  // d^2/de^2 cothS
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION ddcothS(const T1 e, const T2 T)
  {
    return -dcothS(e, T) * cothS(e, T) / T;
  }
  // d^3/de^3 cothS
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dddcothS(const T1 e, const T2 T)
  {
    return -(ddcothS(e, T) * cothS(e, T) + powr<2>(dcothS(e, T)));
  }
  // tanh(e/2T)
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION tanhS(const T1 e, const T2 T)
  {
    return Tanh(e / (T * 2.));
  }
  // d/de tanhS
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dtanhS(const T1 e, const T2 T)
  {
    return 1. / powr<2>(Cosh(e / (T * 2.))) / (2. * T);
  }
  // d^2/de^2 tanhS
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION ddtanhS(const T1 e, const T2 T)
  {
    return -dtanhS(e, T) * tanhS(e, T) / T;
  }
  // sech(e/2T)
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION sechS(const T1 e, const T2 T)
  {
    return 1. / Cosh(e / (T * 2.));
  }
  // csch(e/2T)
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION cschS(const T1 e, const T2 T)
  {
    return 1. / Sinh(e / (T * 2.));
  }

  // ----------------------------------------------------------------------------------------------------
  // Distribution Functions nB, nF and their Derivatives
  // ----------------------------------------------------------------------------------------------------
  // Bosonic distribution
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION nB(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return 1. / expm1(e / T);
  }
  // Derivative d/de of the bosonic distribution
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dnB(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return -exp(e / T) / powr<2>(expm1(e / T)) / T;
  }
  // Derivative d²/de² of the bosonic distribution
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION ddnB(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return exp(e / T) * (1. + exp(e / T)) / powr<3>(expm1(e / T)) / powr<2>(T);
  }
  // Fermionic distribution
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION nF(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return 1. / (exp(e / T) + 1.);
  }
  // Derivative d/de of the fermionic distribution
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dnF(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return -exp(e / T) / powr<2>(exp(e / T) + 1.) / T;
  }
} // namespace DiFfRG