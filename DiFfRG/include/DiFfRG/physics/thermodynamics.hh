#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

// standard library
#include <cmath>

/**
 * @file thermodynamics.hh
 * @brief Thermal distribution functions and the hyperbolic functions that appear in Matsubara sums.
 *
 * All functions take an energy (or general argument) and the temperature \f$T\f$, and are usable in host and device
 * code. None of them takes a chemical potential: for a fermion with chemical potential \f$\mu\f$, pass
 * \f$e \mp \mu\f$ as the energy argument. The functions with suffix `S` and the distributions are related by
 * \f[
 *   \coth\frac{e}{2T} = 1 + 2 n_B(e)\,, \qquad \tanh\frac{e}{2T} = 1 - 2 n_F(e)\,.
 * \f]
 */

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

  // Cosh, Sinh, Tanh, Coth, Sech, Csch: the hyperbolic functions for any argument type (real, complex, autodiff).
  using internal::Cosh, internal::Sinh, internal::Tanh, internal::Coth, internal::Sech, internal::Csch;

  // ----------------------------------------------------------------------------------------------------
  // Finite temperature hyperbolic functions
  // ----------------------------------------------------------------------------------------------------

  /**
   * @brief \f$\coth(x/2T)\f$, safe for \f$T \to 0\f$: when \f$T/x\f$ vanishes numerically, returns the limit
   * \f$\mathrm{sign}(\mathrm{Re}\,x)\f$.
   */
  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto KOKKOS_FORCEINLINE_FUNCTION CothFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Coth(x / (2 * T)));
    return is_close(T / x, T2(0)) ? (R)(real(x) < 0 ? -1 : 1) : Coth(x / (2 * T));
  }
  /**
   * @brief \f$\tanh(x/2T)\f$, safe for \f$T \to 0\f$: when \f$T/x\f$ vanishes numerically, returns the limit
   * \f$\mathrm{sign}(\mathrm{Re}\,x)\f$.
   */
  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto KOKKOS_FORCEINLINE_FUNCTION TanhFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Tanh(x / (2 * T)));
    return is_close(T / x, T2(0)) ? (R)(real(x) < 0 ? -1 : 1) : Tanh(x / (2 * T));
  }
  /// @brief \f$1/\cosh(x/2T)\f$; returns 0 where \f$\cosh\f$ overflows (e.g. \f$T \to 0\f$).
  template <typename T1, typename T2>
    requires(std::is_arithmetic_v<T2>)
  auto KOKKOS_FORCEINLINE_FUNCTION SechFiniteT(const T1 x, const T2 T)
  {
    using R = decltype(Cosh(x / (2 * T)));
    const auto res = Cosh(x / (2 * T));
    return !isfinite(res) ? (R)0 : (R)(1) / res;
  }
  /// @brief \f$1/\sinh(x/2T)\f$; returns 0 where \f$\sinh\f$ overflows (e.g. \f$T \to 0\f$).
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
  /// @brief \f$\coth(e/2T) = 1 + 2 n_B(e)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION cothS(const T1 e, const T2 T)
  {
    return Coth(e / (T * 2.));
  }
  /// @brief \f$\partial_e \coth(e/2T)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dcothS(const T1 e, const T2 T)
  {
    return -1. / powr<2>(Sinh(e / (T * 2.))) / (2. * T);
  }
  /// @brief \f$\partial_e^2 \coth(e/2T)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION ddcothS(const T1 e, const T2 T)
  {
    return -dcothS(e, T) * cothS(e, T) / T;
  }
  /// @brief \f$\partial_e^3 \coth(e/2T)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dddcothS(const T1 e, const T2 T)
  {
    return -(ddcothS(e, T) * cothS(e, T) + powr<2>(dcothS(e, T))) / T;
  }
  /// @brief \f$\tanh(e/2T) = 1 - 2 n_F(e)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION tanhS(const T1 e, const T2 T)
  {
    return Tanh(e / (T * 2.));
  }
  /// @brief \f$\partial_e \tanh(e/2T)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dtanhS(const T1 e, const T2 T)
  {
    return 1. / powr<2>(Cosh(e / (T * 2.))) / (2. * T);
  }
  /// @brief \f$\partial_e^2 \tanh(e/2T)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION ddtanhS(const T1 e, const T2 T)
  {
    return -dtanhS(e, T) * tanhS(e, T) / T;
  }
  /// @brief \f$1/\cosh(e/2T)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION sechS(const T1 e, const T2 T)
  {
    return 1. / Cosh(e / (T * 2.));
  }
  /// @brief \f$1/\sinh(e/2T)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION cschS(const T1 e, const T2 T)
  {
    return 1. / Sinh(e / (T * 2.));
  }

  // ----------------------------------------------------------------------------------------------------
  // Distribution Functions nB, nF and their Derivatives
  // ----------------------------------------------------------------------------------------------------
  /// @brief Bose distribution \f$n_B(e) = 1/(e^{e/T} - 1)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION nB(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return 1. / expm1(e / T);
  }
  /// @brief \f$\partial_e n_B(e)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dnB(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return -exp(e / T) / powr<2>(expm1(e / T)) / T;
  }
  /// @brief \f$\partial_e^2 n_B(e)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION ddnB(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return exp(e / T) * (1. + exp(e / T)) / powr<3>(expm1(e / T)) / powr<2>(T);
  }
  /// @brief Fermi distribution \f$n_F(e) = 1/(e^{e/T} + 1)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION nF(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return 1. / (exp(e / T) + 1.);
  }
  /// @brief \f$\partial_e n_F(e)\f$.
  template <typename T1, typename T2> auto KOKKOS_FORCEINLINE_FUNCTION dnF(const T1 e, const T2 T)
  {
    using Kokkos::cosh, Kokkos::sinh, Kokkos::tanh, Kokkos::exp, Kokkos::expm1;
    return -exp(e / T) / powr<2>(exp(e / T) + 1.) / T;
  }
} // namespace DiFfRG