#pragma once

// standard library
#include <cmath>
#include <type_traits>

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  /**
   * @brief Implements the Litim regulator, i.e.
   * \f[
   *  R_B(k^2,q^2) = (k^2 - q^2) \Theta(k^2 - q^2)
   * \f]
   *
   * Provides the following functions:
   * - RB(k2, q2) = \f$ p^2 r_B(k^2,p^2) \f$
   * - RBdot(k2, q2) = \f$ \partial_t R_B(k^2,p^2) \f$
   * - RF(k2, q2) = \f$ p r_F(k^2,p^2) \f$
   * - RFdot(k2, q2) = \f$ p \partial_ t R_F(k^2,p^2) \f$
   */
  template <class Dummy = void> struct LitimRegulator {
    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RB(const NT1 k2, const NT2 q2)
    {
      return (k2 - q2) * (k2 > q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RBdot(const NT1 k2, const NT2 q2)
    {
      using T = std::decay_t<decltype(k2 * q2)>;
      return T(2) * k2 * (k2 > q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RFdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      using T = std::decay_t<decltype(k2 * q2)>;
      return T(0.5) * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }

    // Give an explicit compile error if a call to dq2RB is made
    template <typename NT1, typename NT2> static void dq2RB(const NT1, const NT2) = delete;
  };

  struct BosonicRegulatorOpts {
    static constexpr int b = 2;
  };
  /**
   * @brief Implements one of the standard exponential regulators, i.e.
   * \f[
   *   R_B(k^2,q^2) = q^2 \frac{(q^2/k^2)^{b-1}}{\exp((q^2/k^2)^b) - 1}
   * \f]
   *
   * Provides the following functions:
   * - RB(k2, q2) = \f$ p^2 r_B(k^2,p^2) \f$
   * - dq2RB(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_B(k^2,p^2) \f$
   * - RBdot(k2, q2) = \f$ \partial_t R_B(k^2,p^2) \f$
   * - RF(k2, q2) = \f$ p r_F(k^2,p^2) \f$
   * - dq2RF(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_F(k^2,p^2) \f$
   * - RFdot(k2, q2) = \f$ p \partial_ t R_F(k^2,p^2) \f$
   *
   * @tparam b The exponent in the regulator.
   */
  template <class OPTS = BosonicRegulatorOpts> struct BosonicRegulator {
    // INTEGER, as in ExponentialRegulator: b is the exponent of powr<b>, a non-type int parameter.
    static constexpr int b = OPTS::b;

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RB(const NT1 k2, const NT2 q2)
    {
      using Kokkos::expm1;
      return q2 * powr<b - 1>(q2 / k2) / expm1(powr<b>(q2 / k2));
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto dq2RB(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      using Kokkos::expm1;
      const auto xb = powr<b>(q2 / k2);
      const auto mexp = exp(xb);
      const auto mexpm1 = expm1(xb);
      using T = std::decay_t<decltype(xb)>;
      return T(b) * mexp * powr<b - 1>(q2 / k2) * (xb - mexpm1) / powr<2>(mexpm1);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RBdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      using Kokkos::expm1;
      const auto xb = powr<b>(q2 / k2);
      const auto mexp = exp(xb);
      const auto mexpm1 = expm1(xb);
      using T = std::decay_t<decltype(xb)>;
      return T(2) * k2 * xb * (mexpm1 * T(1 - b) + T(b) * mexp * xb) / powr<2>(mexpm1);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto dq2RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      const auto q = sqrt(q2);
      using T = std::decay_t<decltype(q)>;
      return (T(-1) + q * (T(1) + dq2RB(k2, q2)) / (q + RF(k2, q2))) / (T(2) * q);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RFdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      using T = std::decay_t<decltype(k2 * q2)>;
      return T(0.5) * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }
  };

  struct ExponentialRegulatorOpts {
    static constexpr int b = 2;
  };
  /**
   * @brief Implements one of the standard exponential regulators, i.e.
   * \f[
   *   R_B(k^2,q^2) = k^2 \exp(-(q^2/k^2)^b)
   * \f]
   *
   * Provides the following functions:
   * - RB(k2, q2) = \f$ p^2 r_B(k^2,p^2) \f$
   * - dq2RB(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_B(k^2,p^2) \f$
   * - RBdot(k2, q2) = \f$ \partial_t R_B(k^2,p^2) \f$
   * - RF(k2, q2) = \f$ p r_F(k^2,p^2) \f$
   * - dq2RF(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_F(k^2,p^2) \f$
   * - RFdot(k2, q2) = \f$ p \partial_ t R_F(k^2,p^2) \f$
   *
   * @tparam b The exponent in the regulator.
   */
  template <class OPTS = ExponentialRegulatorOpts> struct ExponentialRegulator {
    // INTEGER: every use below goes through powr<b>, whose exponent is a non-type int parameter.
    // As a double this template never instantiated -- and nothing in the tree instantiated it, so
    // it never had to.
    static constexpr int b = OPTS::b;

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RB(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      const auto xb = powr<b>(q2 / k2);
      return k2 * exp(-xb);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto dq2RB(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      const auto xbm1 = powr<b - 1>(q2 / k2);
      const auto xb = xbm1 * (q2 / k2);
      using T = std::decay_t<decltype(xb)>;
      return T(-b) * xbm1 * exp(-xb);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RBdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      const auto xb = powr<b>(q2 / k2);
      using T = std::decay_t<decltype(xb)>;
      return T(2) * exp(-xb) * k2 * (T(1) + T(b) * xb);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto dq2RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      const auto q = sqrt(q2);
      using T = std::decay_t<decltype(q)>;
      return (T(-1) + q * (T(1) + dq2RB(k2, q2)) / (q + RF(k2, q2))) / (T(2) * q);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RFdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      using T = std::decay_t<decltype(k2 * q2)>;
      return T(0.5) * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }
  };

  struct SmoothedLitimRegulatorOpts {
    static constexpr double alpha = 2e-3;
  };
  /**
   * @brief Implements one of the standard exponential regulators, i.e.
   * \f[
   *   R_B(k^2,q^2) = k^2 \exp(-(q^2/k^2)^b)
   * \f]
   *
   * Provides the following functions:
   * - RB(k2, q2) = \f$ p^2 r_B(k^2,p^2) \f$
   * - dq2RB(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_B(k^2,p^2) \f$
   * - RBdot(k2, q2) = \f$ \partial_t R_B(k^2,p^2) \f$
   * - RF(k2, q2) = \f$ p r_F(k^2,p^2) \f$
   * - dq2RF(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_F(k^2,p^2) \f$
   * - RFdot(k2, q2) = \f$ p \partial_ t R_F(k^2,p^2) \f$
   *
   * @tparam b The exponent in the regulator.
   */
  template <class OPTS = SmoothedLitimRegulatorOpts> struct SmoothedLitimRegulator {
    static constexpr double alpha = OPTS::alpha;

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RB(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      using T = std::decay_t<decltype(k2 * q2)>;
      return (k2 - q2) / (T(1) + exp((q2 / k2 - T(1)) / T(alpha)));
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RBdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      const auto x = q2 / k2;
      using T = std::decay_t<decltype(x)>;
      const auto mexp = exp((x - T(1)) / T(alpha));
      // Past the cutoff the result underflows; return 0 before mexp^2 overflows to inf/inf. 1e50 is
      // beyond float range, so single precision cuts off at 1e30 instead.
      constexpr double cutoff = std::is_same_v<T, float> ? 1e30 : 1e50;
      return mexp > T(cutoff)
                 ? T(0)
                 : T(2) * k2 * (T(1) + mexp * (x - powr<2>(x) + T(alpha)) / T(alpha)) / powr<2>(T(1) + mexp);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RFdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      using T = std::decay_t<decltype(k2 * q2)>;
      return T(0.5) * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }
  };

  struct RationalExpRegulatorOpts {
    static constexpr int order = 8;
    constexpr static double c = 2;
    constexpr static double b0 = 0.;
  };
  /**
   * @brief Implements a regulator given by \f[R_B(x) = k^2 e^{-f(x)}\,,\f] where \f$f(x)\f$ is a rational function
   * chosen such that the propagator gets a pole of order order at x = 0 if the mass becomes negative (convexity
   * restoration).
   *
   * Provides the following functions:
   * - RB(k2, q2) = \f$ k^2 e^{-f(x)} \f$
   * - RBdot(k2, q2) = \f$ \partial_t R_B(k^2,p^2) \f$
   * - dq2RB(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_B(k^2,p^2) \f$
   * - RF(k2, q2) = \f$ \sqrt{R_B(k^2,p^2) + p^2} - p \f$
   * - RFdot(k2, q2) = \f$ p \partial_ t R_F(k^2,p^2) \f$
   * - dq2RF(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_F(k^2,p^2) \f$
   *
   * @tparam order order of the pole.
   */
  template <class OPTS = RationalExpRegulatorOpts> struct RationalExpRegulator {
    static constexpr int order = OPTS::order;
    static_assert(order > 0, "RationalExpRegulator : Regulator order must be positive!");
    // These are magic numbers. c controls the tail, which is of shape exp(-c q^2/k^2), whereas b0 controls how strong
    // the litim-like part gets cut off into the tail, see also the ConvexRegulators.nb Mathematica notebook
    constexpr static double c = OPTS::c;
    constexpr static double b0 = OPTS::b0;

    // Every literal and coefficient is converted to x's type, so a float x stays float throughout.
    static KOKKOS_INLINE_FUNCTION auto get_f(const auto x)
    {
      using T = std::decay_t<decltype(x)>;
      const T b0 = T(OPTS::b0), c = T(OPTS::c);
      if constexpr (order == 1) return x;
      if constexpr (order == 2)
        return x * (-T(2.) + c * (T(2.) + x + T(2.) * b0 * x)) * powr<-1>(-T(2.) + x + T(2.) * c * (T(1.) + b0 * x));
      if constexpr (order == 3)
        return x * (-T(3.) * (T(2.) + x + T(2.) * b0 * x) + c * (T(6.) + (T(3.) + T(6.) * b0) * x + (T(2.) + T(3.) * b0) * powr<2>(x))) *
               powr<-1>(T(3.) * b0 * (-T(2.) + x) * x + T(6.) * c * (T(1.) + b0 * x) + T(2.) * (-T(3.) + powr<2>(x)));
      if constexpr (order == 4)
        return T(0.08333333333333333) * x *
               (T(12.) + T(6.) * x + T(4.) * (T(1.) + T(3.) * b0) * powr<2>(x) +
                T(3.) * (c + T(2.) * b0 * c) * powr<-1>(-T(1.) + c) * powr<3>(x)) *
               powr<-1>(T(1.) + b0 * powr<2>(x) + T(0.25) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<3>(x));
      if constexpr (order == 5)
        return T(0.016666666666666666) * x *
               (T(60.) + T(30.) * x + T(20.) * (T(1.) + T(3.) * b0) * powr<2>(x) + T(15.) * (T(1.) + T(2.) * b0) * powr<3>(x) +
                T(4.) * (T(3.) + T(5.) * b0) * c * powr<-1>(-T(1.) + c) * powr<4>(x)) *
               powr<-1>(T(1.) + b0 * powr<2>(x) + T(0.06666666666666667) * (T(3.) + T(5.) * b0) * powr<-1>(-T(1.) + c) * powr<4>(x));
      if constexpr (order == 6)
        return T(0.016666666666666666) * x *
               (T(60.) + T(30.) * x + T(20.) * powr<2>(x) + T(15.) * (T(1.) + T(4.) * b0) * powr<3>(x) +
                T(6.) * (T(2.) + T(5.) * b0) * powr<4>(x) + T(10.) * (c + T(2.) * b0 * c) * powr<-1>(-T(1.) + c) * powr<5>(x)) *
               powr<-1>(T(1.) + b0 * powr<3>(x) + T(0.16666666666666666) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<5>(x));
      if constexpr (order == 7)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + (T(0.25) + b0) * powr<4>(x) +
                T(0.1) * (T(2.) + T(5.) * b0) * powr<5>(x) + T(0.16666666666666666) * (T(1.) + T(2.) * b0) * powr<6>(x) +
                T(0.03571428571428571) * (T(4.) + T(7.) * b0) * c * powr<-1>(-T(1.) + c) * powr<7>(x)) *
               powr<-1>(T(1.) + b0 * powr<3>(x) + T(0.03571428571428571) * (T(4.) + T(7.) * b0) * powr<-1>(-T(1.) + c) * powr<6>(x));
      if constexpr (order == 8)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + (T(0.2) + b0) * powr<5>(x) +
                T(0.16666666666666666) * (T(1.) + T(3.) * b0) * powr<6>(x) + T(0.047619047619047616) * (T(3.) + T(7.) * b0) * powr<7>(x) +
                T(0.125) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<8>(x)) *
               powr<-1>(T(1.) + b0 * powr<4>(x) + T(0.125) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<7>(x));
      if constexpr (order == 9)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + (T(0.2) + b0) * powr<5>(x) +
                T(0.16666666666666666) * (T(1.) + T(3.) * b0) * powr<6>(x) + T(0.047619047619047616) * (T(3.) + T(7.) * b0) * powr<7>(x) +
                T(0.125) * (T(1.) + T(2.) * b0) * powr<8>(x) +
                T(0.022222222222222223) * (T(5.) + T(9.) * b0) * c * powr<-1>(-T(1.) + c) * powr<9>(x)) *
               powr<-1>(T(1.) + b0 * powr<4>(x) + T(0.022222222222222223) * (T(5.) + T(9.) * b0) * powr<-1>(-T(1.) + c) * powr<8>(x));
      if constexpr (order == 10)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                (T(0.16666666666666666) + b0) * powr<6>(x) + T(0.07142857142857142) * (T(2.) + T(7.) * b0) * powr<7>(x) +
                T(0.041666666666666664) * (T(3.) + T(8.) * b0) * powr<8>(x) +
                T(0.027777777777777776) * (T(4.) + T(9.) * b0) * powr<9>(x) +
                T(0.1) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<10>(x)) *
               powr<-1>(T(1.) + b0 * powr<5>(x) + T(0.1) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<9>(x));
      if constexpr (order == 11)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                (T(0.16666666666666666) + b0) * powr<6>(x) + T(0.07142857142857142) * (T(2.) + T(7.) * b0) * powr<7>(x) +
                T(0.041666666666666664) * (T(3.) + T(8.) * b0) * powr<8>(x) +
                T(0.027777777777777776) * (T(4.) + T(9.) * b0) * powr<9>(x) + T(0.1) * (T(1.) + T(2.) * b0) * powr<10>(x) +
                T(0.015151515151515152) * (T(6.) + T(11.) * b0) * c * powr<-1>(-T(1.) + c) * powr<11>(x)) *
               powr<-1>(T(1.) + b0 * powr<5>(x) +
                        T(0.015151515151515152) * (T(6.) + T(11.) * b0) * powr<-1>(-T(1.) + c) * powr<10>(x));
      if constexpr (order == 12)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                T(0.16666666666666666) * powr<6>(x) + (T(0.14285714285714285) + b0) * powr<7>(x) +
                T(0.125) * (T(1.) + T(4.) * b0) * powr<8>(x) + T(0.1111111111111111) * (T(1.) + T(3.) * b0) * powr<9>(x) +
                T(0.05) * (T(2.) + T(5.) * b0) * powr<10>(x) + T(0.01818181818181818) * (T(5.) + T(11.) * b0) * powr<11>(x) +
                T(0.08333333333333333) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<12>(x)) *
               powr<-1>(T(1.) + b0 * powr<6>(x) + T(0.08333333333333333) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<11>(x));
      if constexpr (order == 13)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                T(0.16666666666666666) * powr<6>(x) + (T(0.14285714285714285) + b0) * powr<7>(x) +
                T(0.125) * (T(1.) + T(4.) * b0) * powr<8>(x) + T(0.1111111111111111) * (T(1.) + T(3.) * b0) * powr<9>(x) +
                T(0.05) * (T(2.) + T(5.) * b0) * powr<10>(x) + T(0.01818181818181818) * (T(5.) + T(11.) * b0) * powr<11>(x) +
                T(0.08333333333333333) * (T(1.) + T(2.) * b0) * powr<12>(x) +
                T(0.01098901098901099) * (T(7.) + T(13.) * b0) * c * powr<-1>(-T(1.) + c) * powr<13>(x)) *
               powr<-1>(T(1.) + b0 * powr<6>(x) + T(0.01098901098901099) * (T(7.) + T(13.) * b0) * powr<-1>(-T(1.) + c) * powr<12>(x));
      if constexpr (order == 14)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                T(0.16666666666666666) * powr<6>(x) + T(0.14285714285714285) * powr<7>(x) + (T(0.125) + b0) * powr<8>(x) +
                T(0.05555555555555555) * (T(2.) + T(9.) * b0) * powr<9>(x) +
                T(0.03333333333333333) * (T(3.) + T(10.) * b0) * powr<10>(x) +
                T(0.022727272727272728) * (T(4.) + T(11.) * b0) * powr<11>(x) +
                T(0.016666666666666666) * (T(5.) + T(12.) * b0) * powr<12>(x) +
                T(0.01282051282051282) * (T(6.) + T(13.) * b0) * powr<13>(x) +
                T(0.07142857142857142) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<14>(x)) *
               powr<-1>(T(1.) + b0 * powr<7>(x) + T(0.07142857142857142) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<13>(x));
      if constexpr (order == 15)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                T(0.16666666666666666) * powr<6>(x) + T(0.14285714285714285) * powr<7>(x) + (T(0.125) + b0) * powr<8>(x) +
                T(0.05555555555555555) * (T(2.) + T(9.) * b0) * powr<9>(x) +
                T(0.03333333333333333) * (T(3.) + T(10.) * b0) * powr<10>(x) +
                T(0.022727272727272728) * (T(4.) + T(11.) * b0) * powr<11>(x) +
                T(0.016666666666666666) * (T(5.) + T(12.) * b0) * powr<12>(x) +
                T(0.01282051282051282) * (T(6.) + T(13.) * b0) * powr<13>(x) +
                T(0.07142857142857142) * (T(1.) + T(2.) * b0) * powr<14>(x) +
                T(0.008333333333333333) * (T(8.) + T(15.) * b0) * c * powr<-1>(-T(1.) + c) * powr<15>(x)) *
               powr<-1>(T(1.) + b0 * powr<7>(x) +
                        T(0.008333333333333333) * (T(8.) + T(15.) * b0) * powr<-1>(-T(1.) + c) * powr<14>(x));
      if constexpr (order == 16)
        return (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                T(0.16666666666666666) * powr<6>(x) + T(0.14285714285714285) * powr<7>(x) + T(0.125) * powr<8>(x) +
                (T(0.1111111111111111) + b0) * powr<9>(x) + T(0.1) * (T(1.) + T(5.) * b0) * powr<10>(x) +
                T(0.030303030303030304) * (T(3.) + T(11.) * b0) * powr<11>(x) +
                T(0.08333333333333333) * (T(1.) + T(3.) * b0) * powr<12>(x) +
                T(0.015384615384615385) * (T(5.) + T(13.) * b0) * powr<13>(x) +
                T(0.023809523809523808) * (T(3.) + T(7.) * b0) * powr<14>(x) +
                (T(0.06666666666666667) + T(0.14285714285714285) * b0) * powr<15>(x) +
                T(0.0625) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<16>(x)) *
               powr<-1>(T(1.) + b0 * powr<8>(x) + T(0.0625) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<15>(x));
      static_assert(order <= 16, "Rational regulator of order > 16 not implemented");
    }
    static KOKKOS_INLINE_FUNCTION auto get_df(const auto x)
    {
      using T = std::decay_t<decltype(x)>;
      const T b0 = T(OPTS::b0), c = T(OPTS::c);
      if constexpr (order == 1) return T(1.);
      if constexpr (order == 2)
        return (T(4.) + c * (-T(8.) - T(4.) * (T(1.) + T(2.) * b0) * x + (T(1.) + T(2.) * b0) * powr<2>(x)) +
                T(2.) * powr<2>(c) * (T(2.) + (T(2.) + T(4.) * b0) * x + b0 * (T(1.) + T(2.) * b0) * powr<2>(x))) *
               powr<-2>(-T(2.) + x + T(2.) * c * (T(1.) + b0 * x));
      if constexpr (order == 3)
        return (T(1.) + x + T(2.) * b0 * x +
                T(0.3333333333333333) * (-T(1.) + T(3.) * c + b0 * (-T(3.) + T(6.) * c) + T(3.) * (-T(1.) + c) * powr<2>(b0)) *
                    powr<-1>(-T(1.) + c) * powr<2>(x) +
                T(0.3333333333333333) * b0 * (T(2.) + T(3.) * b0) * c * powr<-1>(-T(1.) + c) * powr<3>(x) +
                T(0.027777777777777776) * c * powr<2>(T(2.) + T(3.) * b0) * powr<-2>(-T(1.) + c) * powr<4>(x)) *
               powr<-2>(T(1.) + b0 * x + T(0.16666666666666666) * (T(2.) + T(3.) * b0) * powr<-1>(-T(1.) + c) * powr<2>(x));
      if constexpr (order == 4)
        return (T(1.) + x + (T(1.) + T(2.) * b0) * powr<2>(x) +
                T(0.5) * (T(1.) + T(2.) * b0) * (-T(1.) + T(2.) * c) * powr<-1>(-T(1.) + c) * powr<3>(x) +
                T(0.041666666666666664) * (-T(3.) + T(2.) * b0 * (-T(7.) + T(4.) * c) + T(24.) * (-T(1.) + c) * powr<2>(b0)) *
                    powr<-1>(-T(1.) + c) * powr<4>(x) +
                T(0.5) * b0 * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<5>(x) +
                T(0.0625) * c * powr<2>(T(1.) + T(2.) * b0) * powr<-2>(-T(1.) + c) * powr<6>(x)) *
               powr<-2>(T(1.) + b0 * powr<2>(x) + T(0.25) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<3>(x));
      if constexpr (order == 5)
        return ((T(1.) + b0 * powr<2>(x) + T(0.06666666666666667) * (T(3.) + T(5.) * b0) * powr<-1>(-T(1.) + c) * powr<4>(x)) *
                    (T(1.) + x + (T(1.) + T(3.) * b0) * powr<2>(x) + (T(1.) + T(2.) * b0) * powr<3>(x) +
                     T(0.3333333333333333) * (T(3.) + T(5.) * b0) * c * powr<-1>(-T(1.) + c) * powr<4>(x)) -
                T(0.016666666666666666) * x *
                    (T(2.) * b0 * x + T(0.26666666666666666) * (T(3.) + T(5.) * b0) * powr<-1>(-T(1.) + c) * powr<3>(x)) *
                    (T(60.) + T(30.) * x + T(20.) * (T(1.) + T(3.) * b0) * powr<2>(x) + T(15.) * (T(1.) + T(2.) * b0) * powr<3>(x) +
                     T(4.) * (T(3.) + T(5.) * b0) * c * powr<-1>(-T(1.) + c) * powr<4>(x))) *
               powr<-2>(T(1.) + b0 * powr<2>(x) + T(0.06666666666666667) * (T(3.) + T(5.) * b0) * powr<-1>(-T(1.) + c) * powr<4>(x));
      if constexpr (order == 6)
        return ((T(1.) + b0 * powr<3>(x) + T(0.16666666666666666) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<5>(x)) *
                    (T(1.) + x + powr<2>(x) + (T(1.) + T(4.) * b0) * powr<3>(x) + (T(1.) + T(2.5) * b0) * powr<4>(x) +
                     (c + T(2.) * b0 * c) * powr<-1>(-T(1.) + c) * powr<5>(x)) -
                T(0.016666666666666666) * x *
                    (T(3.) * b0 * powr<2>(x) + T(0.8333333333333334) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<4>(x)) *
                    (T(60.) + T(30.) * x + T(20.) * powr<2>(x) + T(15.) * (T(1.) + T(4.) * b0) * powr<3>(x) +
                     T(6.) * (T(2.) + T(5.) * b0) * powr<4>(x) + T(10.) * (c + T(2.) * b0 * c) * powr<-1>(-T(1.) + c) * powr<5>(x))) *
               powr<-2>(T(1.) + b0 * powr<3>(x) + T(0.16666666666666666) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<5>(x));
      if constexpr (order == 7)
        return ((T(1.) + b0 * powr<3>(x) + T(0.03571428571428571) * (T(4.) + T(7.) * b0) * powr<-1>(-T(1.) + c) * powr<6>(x)) *
                    (T(1.) + x + powr<2>(x) + (T(1.) + T(4.) * b0) * powr<3>(x) + (T(1.) + T(2.5) * b0) * powr<4>(x) +
                     (T(1.) + T(2.) * b0) * powr<5>(x) + T(0.25) * (T(4.) + T(7.) * b0) * c * powr<-1>(-T(1.) + c) * powr<6>(x)) -
                T(1.) * (T(3.) * b0 * powr<2>(x) + T(0.21428571428571427) * (T(4.) + T(7.) * b0) * powr<-1>(-T(1.) + c) * powr<5>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + (T(0.25) + b0) * powr<4>(x) +
                     T(0.1) * (T(2.) + T(5.) * b0) * powr<5>(x) + T(0.16666666666666666) * (T(1.) + T(2.) * b0) * powr<6>(x) +
                     T(0.03571428571428571) * (T(4.) + T(7.) * b0) * c * powr<-1>(-T(1.) + c) * powr<7>(x))) *
               powr<-2>(T(1.) + b0 * powr<3>(x) + T(0.03571428571428571) * (T(4.) + T(7.) * b0) * powr<-1>(-T(1.) + c) * powr<6>(x));
      if constexpr (order == 8)
        return ((T(1.) + b0 * powr<4>(x) + T(0.125) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<7>(x)) *
                    (T(1.) + x + powr<2>(x) + powr<3>(x) + (T(1.) + T(5.) * b0) * powr<4>(x) + (T(1.) + T(3.) * b0) * powr<5>(x) +
                     (T(1.) + T(2.3333333333333335) * b0) * powr<6>(x) + (c + T(2.) * b0 * c) * powr<-1>(-T(1.) + c) * powr<7>(x)) -
                T(1.) * (T(4.) * b0 * powr<3>(x) + T(0.875) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<6>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) +
                     (T(0.2) + b0) * powr<5>(x) + T(0.16666666666666666) * (T(1.) + T(3.) * b0) * powr<6>(x) +
                     T(0.047619047619047616) * (T(3.) + T(7.) * b0) * powr<7>(x) +
                     T(0.125) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<8>(x))) *
               powr<-2>(T(1.) + b0 * powr<4>(x) + T(0.125) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<7>(x));
      if constexpr (order == 9)
        return ((T(1.) + b0 * powr<4>(x) + T(0.022222222222222223) * (T(5.) + T(9.) * b0) * powr<-1>(-T(1.) + c) * powr<8>(x)) *
                    (T(1.) + x + powr<2>(x) + powr<3>(x) + (T(1.) + T(5.) * b0) * powr<4>(x) + (T(1.) + T(3.) * b0) * powr<5>(x) +
                     (T(1.) + T(2.3333333333333335) * b0) * powr<6>(x) + (T(1.) + T(2.) * b0) * powr<7>(x) +
                     T(0.2) * (T(5.) + T(9.) * b0) * c * powr<-1>(-T(1.) + c) * powr<8>(x)) -
                T(1.) * (T(4.) * b0 * powr<3>(x) + T(0.17777777777777778) * (T(5.) + T(9.) * b0) * powr<-1>(-T(1.) + c) * powr<7>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) +
                     (T(0.2) + b0) * powr<5>(x) + T(0.16666666666666666) * (T(1.) + T(3.) * b0) * powr<6>(x) +
                     T(0.047619047619047616) * (T(3.) + T(7.) * b0) * powr<7>(x) + T(0.125) * (T(1.) + T(2.) * b0) * powr<8>(x) +
                     T(0.022222222222222223) * (T(5.) + T(9.) * b0) * c * powr<-1>(-T(1.) + c) * powr<9>(x))) *
               powr<-2>(T(1.) + b0 * powr<4>(x) + T(0.022222222222222223) * (T(5.) + T(9.) * b0) * powr<-1>(-T(1.) + c) * powr<8>(x));
      if constexpr (order == 10)
        return ((T(1.) + b0 * powr<5>(x) + T(0.1) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<9>(x)) *
                    (T(1.) + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + (T(1.) + T(6.) * b0) * powr<5>(x) +
                     (T(1.) + T(3.5) * b0) * powr<6>(x) + (T(1.) + T(2.6666666666666665) * b0) * powr<7>(x) +
                     (T(1.) + T(2.25) * b0) * powr<8>(x) + (c + T(2.) * b0 * c) * powr<-1>(-T(1.) + c) * powr<9>(x)) -
                T(1.) * (T(5.) * b0 * powr<4>(x) + T(0.9) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<8>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                     (T(0.16666666666666666) + b0) * powr<6>(x) + T(0.07142857142857142) * (T(2.) + T(7.) * b0) * powr<7>(x) +
                     T(0.041666666666666664) * (T(3.) + T(8.) * b0) * powr<8>(x) +
                     T(0.027777777777777776) * (T(4.) + T(9.) * b0) * powr<9>(x) +
                     T(0.1) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<10>(x))) *
               powr<-2>(T(1.) + b0 * powr<5>(x) + T(0.1) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<9>(x));
      if constexpr (order == 11)
        return ((T(1.) + b0 * powr<5>(x) + T(0.015151515151515152) * (T(6.) + T(11.) * b0) * powr<-1>(-T(1.) + c) * powr<10>(x)) *
                    (T(1.) + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + (T(1.) + T(6.) * b0) * powr<5>(x) +
                     (T(1.) + T(3.5) * b0) * powr<6>(x) + (T(1.) + T(2.6666666666666665) * b0) * powr<7>(x) +
                     (T(1.) + T(2.25) * b0) * powr<8>(x) + (T(1.) + T(2.) * b0) * powr<9>(x) +
                     T(0.16666666666666666) * (T(6.) + T(11.) * b0) * c * powr<-1>(-T(1.) + c) * powr<10>(x)) -
                T(1.) * (T(5.) * b0 * powr<4>(x) + T(0.15151515151515152) * (T(6.) + T(11.) * b0) * powr<-1>(-T(1.) + c) * powr<9>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                     (T(0.16666666666666666) + b0) * powr<6>(x) + T(0.07142857142857142) * (T(2.) + T(7.) * b0) * powr<7>(x) +
                     T(0.041666666666666664) * (T(3.) + T(8.) * b0) * powr<8>(x) +
                     T(0.027777777777777776) * (T(4.) + T(9.) * b0) * powr<9>(x) + T(0.1) * (T(1.) + T(2.) * b0) * powr<10>(x) +
                     T(0.015151515151515152) * (T(6.) + T(11.) * b0) * c * powr<-1>(-T(1.) + c) * powr<11>(x))) *
               powr<-2>(T(1.) + b0 * powr<5>(x) +
                        T(0.015151515151515152) * (T(6.) + T(11.) * b0) * powr<-1>(-T(1.) + c) * powr<10>(x));
      if constexpr (order == 12)
        return ((T(1.) + b0 * powr<6>(x) + T(0.08333333333333333) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<11>(x)) *
                    (T(1.) + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + (T(1.) + T(7.) * b0) * powr<6>(x) +
                     (T(1.) + T(4.) * b0) * powr<7>(x) + (T(1.) + T(3.) * b0) * powr<8>(x) + (T(1.) + T(2.5) * b0) * powr<9>(x) +
                     (T(1.) + T(2.2) * b0) * powr<10>(x) + (c + T(2.) * b0 * c) * powr<-1>(-T(1.) + c) * powr<11>(x)) -
                T(1.) * (T(6.) * b0 * powr<5>(x) + T(0.9166666666666666) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<10>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                     T(0.16666666666666666) * powr<6>(x) + (T(0.14285714285714285) + b0) * powr<7>(x) +
                     T(0.125) * (T(1.) + T(4.) * b0) * powr<8>(x) + T(0.1111111111111111) * (T(1.) + T(3.) * b0) * powr<9>(x) +
                     T(0.05) * (T(2.) + T(5.) * b0) * powr<10>(x) + T(0.01818181818181818) * (T(5.) + T(11.) * b0) * powr<11>(x) +
                     T(0.08333333333333333) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<12>(x))) *
               powr<-2>(T(1.) + b0 * powr<6>(x) + T(0.08333333333333333) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<11>(x));
      if constexpr (order == 13)
        return ((T(1.) + b0 * powr<6>(x) + T(0.01098901098901099) * (T(7.) + T(13.) * b0) * powr<-1>(-T(1.) + c) * powr<12>(x)) *
                    (T(1.) + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + (T(1.) + T(7.) * b0) * powr<6>(x) +
                     (T(1.) + T(4.) * b0) * powr<7>(x) + (T(1.) + T(3.) * b0) * powr<8>(x) + (T(1.) + T(2.5) * b0) * powr<9>(x) +
                     (T(1.) + T(2.2) * b0) * powr<10>(x) + (T(1.) + T(2.) * b0) * powr<11>(x) +
                     T(0.14285714285714285) * (T(7.) + T(13.) * b0) * c * powr<-1>(-T(1.) + c) * powr<12>(x)) -
                T(1.) * (T(6.) * b0 * powr<5>(x) + T(0.13186813186813187) * (T(7.) + T(13.) * b0) * powr<-1>(-T(1.) + c) * powr<11>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                     T(0.16666666666666666) * powr<6>(x) + (T(0.14285714285714285) + b0) * powr<7>(x) +
                     T(0.125) * (T(1.) + T(4.) * b0) * powr<8>(x) + T(0.1111111111111111) * (T(1.) + T(3.) * b0) * powr<9>(x) +
                     T(0.05) * (T(2.) + T(5.) * b0) * powr<10>(x) + T(0.01818181818181818) * (T(5.) + T(11.) * b0) * powr<11>(x) +
                     T(0.08333333333333333) * (T(1.) + T(2.) * b0) * powr<12>(x) +
                     T(0.01098901098901099) * (T(7.) + T(13.) * b0) * c * powr<-1>(-T(1.) + c) * powr<13>(x))) *
               powr<-2>(T(1.) + b0 * powr<6>(x) + T(0.01098901098901099) * (T(7.) + T(13.) * b0) * powr<-1>(-T(1.) + c) * powr<12>(x));
      if constexpr (order == 14)
        return ((T(1.) + b0 * powr<7>(x) + T(0.07142857142857142) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<13>(x)) *
                    (T(1.) + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + powr<6>(x) +
                     (T(1.) + T(8.) * b0) * powr<7>(x) + (T(1.) + T(4.5) * b0) * powr<8>(x) +
                     (T(1.) + T(3.3333333333333335) * b0) * powr<9>(x) + (T(1.) + T(2.75) * b0) * powr<10>(x) +
                     (T(1.) + T(2.4) * b0) * powr<11>(x) + (T(1.) + T(2.1666666666666665) * b0) * powr<12>(x) +
                     (c + T(2.) * b0 * c) * powr<-1>(-T(1.) + c) * powr<13>(x)) -
                T(1.) * (T(7.) * b0 * powr<6>(x) + T(0.9285714285714286) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<12>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                     T(0.16666666666666666) * powr<6>(x) + T(0.14285714285714285) * powr<7>(x) + (T(0.125) + b0) * powr<8>(x) +
                     T(0.05555555555555555) * (T(2.) + T(9.) * b0) * powr<9>(x) +
                     T(0.03333333333333333) * (T(3.) + T(10.) * b0) * powr<10>(x) +
                     T(0.022727272727272728) * (T(4.) + T(11.) * b0) * powr<11>(x) +
                     T(0.016666666666666666) * (T(5.) + T(12.) * b0) * powr<12>(x) +
                     T(0.01282051282051282) * (T(6.) + T(13.) * b0) * powr<13>(x) +
                     T(0.07142857142857142) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<14>(x))) *
               powr<-2>(T(1.) + b0 * powr<7>(x) + T(0.07142857142857142) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<13>(x));
      if constexpr (order == 15)
        return ((T(1.) + b0 * powr<7>(x) + T(0.008333333333333333) * (T(8.) + T(15.) * b0) * powr<-1>(-T(1.) + c) * powr<14>(x)) *
                    (T(1.) + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + powr<6>(x) +
                     (T(1.) + T(8.) * b0) * powr<7>(x) + (T(1.) + T(4.5) * b0) * powr<8>(x) +
                     (T(1.) + T(3.3333333333333335) * b0) * powr<9>(x) + (T(1.) + T(2.75) * b0) * powr<10>(x) +
                     (T(1.) + T(2.4) * b0) * powr<11>(x) + (T(1.) + T(2.1666666666666665) * b0) * powr<12>(x) +
                     (T(1.) + T(2.) * b0) * powr<13>(x) + T(0.125) * (T(8.) + T(15.) * b0) * c * powr<-1>(-T(1.) + c) * powr<14>(x)) -
                T(1.) * (T(7.) * b0 * powr<6>(x) + T(0.11666666666666667) * (T(8.) + T(15.) * b0) * powr<-1>(-T(1.) + c) * powr<13>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                     T(0.16666666666666666) * powr<6>(x) + T(0.14285714285714285) * powr<7>(x) + (T(0.125) + b0) * powr<8>(x) +
                     T(0.05555555555555555) * (T(2.) + T(9.) * b0) * powr<9>(x) +
                     T(0.03333333333333333) * (T(3.) + T(10.) * b0) * powr<10>(x) +
                     T(0.022727272727272728) * (T(4.) + T(11.) * b0) * powr<11>(x) +
                     T(0.016666666666666666) * (T(5.) + T(12.) * b0) * powr<12>(x) +
                     T(0.01282051282051282) * (T(6.) + T(13.) * b0) * powr<13>(x) +
                     T(0.07142857142857142) * (T(1.) + T(2.) * b0) * powr<14>(x) +
                     T(0.008333333333333333) * (T(8.) + T(15.) * b0) * c * powr<-1>(-T(1.) + c) * powr<15>(x))) *
               powr<-2>(T(1.) + b0 * powr<7>(x) +
                        T(0.008333333333333333) * (T(8.) + T(15.) * b0) * powr<-1>(-T(1.) + c) * powr<14>(x));
      if constexpr (order == 16)
        return ((T(1.) + b0 * powr<8>(x) + T(0.0625) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<15>(x)) *
                    (T(1.) + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + powr<6>(x) + powr<7>(x) +
                     (T(1.) + T(9.) * b0) * powr<8>(x) + (T(1.) + T(5.) * b0) * powr<9>(x) +
                     (T(1.) + T(3.6666666666666665) * b0) * powr<10>(x) + (T(1.) + T(3.) * b0) * powr<11>(x) +
                     (T(1.) + T(2.6) * b0) * powr<12>(x) + (T(1.) + T(2.3333333333333335) * b0) * powr<13>(x) +
                     (T(1.) + T(2.142857142857143) * b0) * powr<14>(x) +
                     (c + T(2.) * b0 * c) * powr<-1>(-T(1.) + c) * powr<15>(x)) -
                T(1.) * (T(8.) * b0 * powr<7>(x) + T(0.9375) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<14>(x)) *
                    (x + T(0.5) * powr<2>(x) + T(0.3333333333333333) * powr<3>(x) + T(0.25) * powr<4>(x) + T(0.2) * powr<5>(x) +
                     T(0.16666666666666666) * powr<6>(x) + T(0.14285714285714285) * powr<7>(x) + T(0.125) * powr<8>(x) +
                     (T(0.1111111111111111) + b0) * powr<9>(x) + T(0.1) * (T(1.) + T(5.) * b0) * powr<10>(x) +
                     T(0.030303030303030304) * (T(3.) + T(11.) * b0) * powr<11>(x) +
                     T(0.08333333333333333) * (T(1.) + T(3.) * b0) * powr<12>(x) +
                     T(0.015384615384615385) * (T(5.) + T(13.) * b0) * powr<13>(x) +
                     T(0.023809523809523808) * (T(3.) + T(7.) * b0) * powr<14>(x) +
                     (T(0.06666666666666667) + T(0.14285714285714285) * b0) * powr<15>(x) +
                     T(0.0625) * (T(1.) + T(2.) * b0) * c * powr<-1>(-T(1.) + c) * powr<16>(x))) *
               powr<-2>(T(1.) + b0 * powr<8>(x) + T(0.0625) * (T(1.) + T(2.) * b0) * powr<-1>(-T(1.) + c) * powr<15>(x));
      static_assert(order <= 16, "Rational regulator of order > 16 not implemented");
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RB(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      const auto x = q2 / k2;
      const auto f = get_f(x);
      return k2 * exp(-f);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RBdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      const auto x = q2 / k2;
      const auto f = get_f(x);
      const auto df = get_df(x);
      using T = std::decay_t<decltype(x)>;
      return T(2) * k2 * exp(-f) * (T(1) + df * x);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto dq2RB(const NT1 k2, const NT2 q2)
    {
      using Kokkos::exp;
      const auto x = q2 / k2;
      const auto f = get_f(x);
      const auto df = get_df(x);
      return -exp(-f) * df;
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RFdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      using T = std::decay_t<decltype(k2 * q2)>;
      return T(0.5) * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto dq2RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      const auto q = sqrt(q2);
      using T = std::decay_t<decltype(q)>;
      return (T(-1) + q * (T(1) + dq2RB(k2, q2)) / (q + RF(k2, q2))) / (T(2) * q);
    }
  };

  struct PolynomialExpRegulatorOpts {
    static constexpr int order = 8;
  };
  /**
   * @brief Implements a regulator given by \f[R_B(x) = k^2 e^{-f(x)}\,,\f] where \f$f(x)\f$ is a polynomial chosen such
   * that the propagator gets a pole of order order at x = 0 if the mass becomes negative (convexity restoration).
   *
   * Provides the following functions:
   * - RB(k2, q2) = \f$ k^2 e^{-f(x)} \f$
   * - RBdot(k2, q2) = \f$ \partial_t R_B(k^2,p^2) \f$
   * - dq2RB(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_B(k^2,p^2) \f$
   * - RF(k2, q2) = \f$ \sqrt{R_B(k^2,p^2) + p^2} - p \f$
   * - RFdot(k2, q2) = \f$ p \partial_ t R_F(k^2,p^2) \f$
   * - dq2RF(k2, q2) = \f$ \frac{\partial}{\partial q^2} R_F(k^2,p^2) \f$
   *
   * @tparam order order of the pole.
   */
  template <class OPTS = PolynomialExpRegulatorOpts> struct PolynomialExpRegulator {
    static constexpr int order = OPTS::order;
    static_assert(order > 0, "PolynomialExpRegulator: order must be > 0 !");

    template <int c0, int... c>
    static KOKKOS_INLINE_FUNCTION auto get_f(std::integer_sequence<int, c0, c...>, const auto x)
    {
      using T = decltype(x);
      if constexpr (sizeof...(c) == 0)
        return powr<c0 + 1>(x) / T(c0 + 1);
      else
        return powr<c0 + 1>(x) / T(c0 + 1) + get_f(std::integer_sequence<int, c...>{}, x);
    }
    template <int c0, int... c>
    static KOKKOS_INLINE_FUNCTION auto get_df(std::integer_sequence<int, c0, c...>, const auto x)
    {
      if constexpr (sizeof...(c) == 0)
        return powr<c0>(x);
      else
        return powr<c0>(x) + get_df(std::integer_sequence<int, c...>{}, x);
    }
    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto f = get_f(std::make_integer_sequence<int, order>{}, x);
      return k2 * exp(-f);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RBdot(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto f = get_f(std::make_integer_sequence<int, order>{}, x);
      const auto df = get_df(std::make_integer_sequence<int, order>{}, x);
      return 2 * k2 * exp(-f) * (1 + df * x);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto dq2RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto f = get_f(std::make_integer_sequence<int, order>{}, x);
      const auto df = get_df(std::make_integer_sequence<int, order>{}, x);
      return -exp(-f) * df;
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto RFdot(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      using T = std::decay_t<decltype(k2 * q2)>;
      return T(0.5) * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }

    template <typename NT1, typename NT2> static KOKKOS_INLINE_FUNCTION auto dq2RF(const NT1 k2, const NT2 q2)
    {
      using Kokkos::sqrt;
      const auto q = sqrt(q2);
      return (-1 + q * (1 + dq2RB(k2, q2)) / (q + RF(k2, q2))) / (2 * q);
    }
  };
} // namespace DiFfRG
