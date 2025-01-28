#pragma once

// standard library
#include <cmath>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
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
    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RB(const NT1 k2, const NT2 q2)
    {
      return (k2 - q2) * (k2 > q2);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RBdot(const NT1 k2, const NT2 q2)
    {
      return 2. * k2 * (k2 > q2);
    }

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RFdot(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return 0.5 * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }

    // Give an explicit compile error if a call to dq2RB is made
    template <typename NT1, typename NT2> static void dq2RB(const NT1, const NT2) = delete;
  };

  struct BosonicRegulatorOpts {
    static constexpr double b = 2.;
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
    static constexpr double b = OPTS::b;

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RB(const NT1 k2, const NT2 q2)
    {
      using std::expm1;
      return q2 * powr<b - 1>(q2 / k2) / expm1(powr<b>(q2 / k2));
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto dq2RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      using std::expm1;
      const auto xb = powr<b>(q2 / k2);
      const auto mexp = exp(xb);
      const auto mexpm1 = expm1(xb);
      return b * mexp * powr<b - 1>(q2 / k2) * (xb - mexpm1) / powr<2>(mexpm1);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RBdot(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      using std::expm1;
      const auto xb = powr<b>(q2 / k2);
      const auto mexp = exp(xb);
      const auto mexpm1 = expm1(xb);
      return 2. * k2 * xb * (mexpm1 * (1. - b) + b * mexp * xb) / powr<2>(mexpm1);
    }

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto dq2RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      const auto q = sqrt(q2);
      return (-1. + q * (1. + dq2RB(k2, q2)) / (q + RF(k2, q2))) / (2. * q);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RFdot(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return 0.5 * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }
  };

  struct ExponentialRegulatorOpts {
    static constexpr double b = 2.;
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
    static constexpr double b = OPTS::b;

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto xb = powr<b>(q2 / k2);
      return k2 * exp(-xb);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto dq2RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto xbm1 = powr<b - 1>(q2 / k2);
      const auto xb = xbm1 * (q2 / k2);
      return -b * xbm1 * exp(-xb);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RBdot(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto xb = powr<b>(q2 / k2);
      return 2. * exp(-xb) * k2 * (1. + b * xb);
    }

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto dq2RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      const auto q = sqrt(q2);
      return (-1. + q * (1. + dq2RB(k2, q2)) / (q + RF(k2, q2))) / (2. * q);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RFdot(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return 0.5 * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
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

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      return (k2 - q2) / (1. + exp((q2 / k2 - 1.) / alpha));
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RBdot(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto mexp = exp((x - 1.) / alpha);
      return mexp > 1e50 ? 0. : 2. * k2 * (1. + mexp * (x - powr<2>(x) + alpha) / alpha) / powr<2>(1. + mexp);
    }

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RFdot(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return 0.5 * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
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

    static __forceinline__ __device__ __host__ auto get_f(const auto x)
    {
      constexpr double b0 = OPTS::b0;
      if constexpr (order == 1) return x;
      if constexpr (order == 2)
        return x * (-2. + c * (2. + x + 2. * b0 * x)) * powr<-1>(-2. + x + 2. * c * (1. + b0 * x));
      if constexpr (order == 3)
        return x * (-3. * (2. + x + 2. * b0 * x) + c * (6. + (3. + 6. * b0) * x + (2. + 3. * b0) * powr<2>(x))) *
               powr<-1>(3. * b0 * (-2. + x) * x + 6. * c * (1. + b0 * x) + 2. * (-3. + powr<2>(x)));
      if constexpr (order == 4)
        return 0.08333333333333333 * x *
               (12. + 6. * x + 4. * (1. + 3. * b0) * powr<2>(x) +
                3. * (c + 2. * b0 * c) * powr<-1>(-1. + c) * powr<3>(x)) *
               powr<-1>(1. + b0 * powr<2>(x) + 0.25 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<3>(x));
      if constexpr (order == 5)
        return 0.016666666666666666 * x *
               (60. + 30. * x + 20. * (1. + 3. * b0) * powr<2>(x) + 15. * (1. + 2. * b0) * powr<3>(x) +
                4. * (3. + 5. * b0) * c * powr<-1>(-1. + c) * powr<4>(x)) *
               powr<-1>(1. + b0 * powr<2>(x) + 0.06666666666666667 * (3. + 5. * b0) * powr<-1>(-1. + c) * powr<4>(x));
      if constexpr (order == 6)
        return 0.016666666666666666 * x *
               (60. + 30. * x + 20. * powr<2>(x) + 15. * (1. + 4. * b0) * powr<3>(x) +
                6. * (2. + 5. * b0) * powr<4>(x) + 10. * (c + 2. * b0 * c) * powr<-1>(-1. + c) * powr<5>(x)) *
               powr<-1>(1. + b0 * powr<3>(x) + 0.16666666666666666 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<5>(x));
      if constexpr (order == 7)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + (0.25 + b0) * powr<4>(x) +
                0.1 * (2. + 5. * b0) * powr<5>(x) + 0.16666666666666666 * (1. + 2. * b0) * powr<6>(x) +
                0.03571428571428571 * (4. + 7. * b0) * c * powr<-1>(-1. + c) * powr<7>(x)) *
               powr<-1>(1. + b0 * powr<3>(x) + 0.03571428571428571 * (4. + 7. * b0) * powr<-1>(-1. + c) * powr<6>(x));
      if constexpr (order == 8)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + (0.2 + b0) * powr<5>(x) +
                0.16666666666666666 * (1. + 3. * b0) * powr<6>(x) + 0.047619047619047616 * (3. + 7. * b0) * powr<7>(x) +
                0.125 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<8>(x)) *
               powr<-1>(1. + b0 * powr<4>(x) + 0.125 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<7>(x));
      if constexpr (order == 9)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + (0.2 + b0) * powr<5>(x) +
                0.16666666666666666 * (1. + 3. * b0) * powr<6>(x) + 0.047619047619047616 * (3. + 7. * b0) * powr<7>(x) +
                0.125 * (1. + 2. * b0) * powr<8>(x) +
                0.022222222222222223 * (5. + 9. * b0) * c * powr<-1>(-1. + c) * powr<9>(x)) *
               powr<-1>(1. + b0 * powr<4>(x) + 0.022222222222222223 * (5. + 9. * b0) * powr<-1>(-1. + c) * powr<8>(x));
      if constexpr (order == 10)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                (0.16666666666666666 + b0) * powr<6>(x) + 0.07142857142857142 * (2. + 7. * b0) * powr<7>(x) +
                0.041666666666666664 * (3. + 8. * b0) * powr<8>(x) +
                0.027777777777777776 * (4. + 9. * b0) * powr<9>(x) +
                0.1 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<10>(x)) *
               powr<-1>(1. + b0 * powr<5>(x) + 0.1 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<9>(x));
      if constexpr (order == 11)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                (0.16666666666666666 + b0) * powr<6>(x) + 0.07142857142857142 * (2. + 7. * b0) * powr<7>(x) +
                0.041666666666666664 * (3. + 8. * b0) * powr<8>(x) +
                0.027777777777777776 * (4. + 9. * b0) * powr<9>(x) + 0.1 * (1. + 2. * b0) * powr<10>(x) +
                0.015151515151515152 * (6. + 11. * b0) * c * powr<-1>(-1. + c) * powr<11>(x)) *
               powr<-1>(1. + b0 * powr<5>(x) +
                        0.015151515151515152 * (6. + 11. * b0) * powr<-1>(-1. + c) * powr<10>(x));
      if constexpr (order == 12)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                0.16666666666666666 * powr<6>(x) + (0.14285714285714285 + b0) * powr<7>(x) +
                0.125 * (1. + 4. * b0) * powr<8>(x) + 0.1111111111111111 * (1. + 3. * b0) * powr<9>(x) +
                0.05 * (2. + 5. * b0) * powr<10>(x) + 0.01818181818181818 * (5. + 11. * b0) * powr<11>(x) +
                0.08333333333333333 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<12>(x)) *
               powr<-1>(1. + b0 * powr<6>(x) + 0.08333333333333333 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<11>(x));
      if constexpr (order == 13)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                0.16666666666666666 * powr<6>(x) + (0.14285714285714285 + b0) * powr<7>(x) +
                0.125 * (1. + 4. * b0) * powr<8>(x) + 0.1111111111111111 * (1. + 3. * b0) * powr<9>(x) +
                0.05 * (2. + 5. * b0) * powr<10>(x) + 0.01818181818181818 * (5. + 11. * b0) * powr<11>(x) +
                0.08333333333333333 * (1. + 2. * b0) * powr<12>(x) +
                0.01098901098901099 * (7. + 13. * b0) * c * powr<-1>(-1. + c) * powr<13>(x)) *
               powr<-1>(1. + b0 * powr<6>(x) + 0.01098901098901099 * (7. + 13. * b0) * powr<-1>(-1. + c) * powr<12>(x));
      if constexpr (order == 14)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                0.16666666666666666 * powr<6>(x) + 0.14285714285714285 * powr<7>(x) + (0.125 + b0) * powr<8>(x) +
                0.05555555555555555 * (2. + 9. * b0) * powr<9>(x) +
                0.03333333333333333 * (3. + 10. * b0) * powr<10>(x) +
                0.022727272727272728 * (4. + 11. * b0) * powr<11>(x) +
                0.016666666666666666 * (5. + 12. * b0) * powr<12>(x) +
                0.01282051282051282 * (6. + 13. * b0) * powr<13>(x) +
                0.07142857142857142 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<14>(x)) *
               powr<-1>(1. + b0 * powr<7>(x) + 0.07142857142857142 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<13>(x));
      if constexpr (order == 15)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                0.16666666666666666 * powr<6>(x) + 0.14285714285714285 * powr<7>(x) + (0.125 + b0) * powr<8>(x) +
                0.05555555555555555 * (2. + 9. * b0) * powr<9>(x) +
                0.03333333333333333 * (3. + 10. * b0) * powr<10>(x) +
                0.022727272727272728 * (4. + 11. * b0) * powr<11>(x) +
                0.016666666666666666 * (5. + 12. * b0) * powr<12>(x) +
                0.01282051282051282 * (6. + 13. * b0) * powr<13>(x) +
                0.07142857142857142 * (1. + 2. * b0) * powr<14>(x) +
                0.008333333333333333 * (8. + 15. * b0) * c * powr<-1>(-1. + c) * powr<15>(x)) *
               powr<-1>(1. + b0 * powr<7>(x) +
                        0.008333333333333333 * (8. + 15. * b0) * powr<-1>(-1. + c) * powr<14>(x));
      if constexpr (order == 16)
        return (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                0.16666666666666666 * powr<6>(x) + 0.14285714285714285 * powr<7>(x) + 0.125 * powr<8>(x) +
                (0.1111111111111111 + b0) * powr<9>(x) + 0.1 * (1. + 5. * b0) * powr<10>(x) +
                0.030303030303030304 * (3. + 11. * b0) * powr<11>(x) +
                0.08333333333333333 * (1. + 3. * b0) * powr<12>(x) +
                0.015384615384615385 * (5. + 13. * b0) * powr<13>(x) +
                0.023809523809523808 * (3. + 7. * b0) * powr<14>(x) +
                (0.06666666666666667 + 0.14285714285714285 * b0) * powr<15>(x) +
                0.0625 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<16>(x)) *
               powr<-1>(1. + b0 * powr<8>(x) + 0.0625 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<15>(x));
      static_assert(order <= 16, "Rational regulator of order > 16 not implemented");
    }
    static __forceinline__ __device__ __host__ auto get_df(const auto x)
    {
      constexpr double b0 = OPTS::b0;
      if constexpr (order == 1) return 1.;
      if constexpr (order == 2)
        return (4. + c * (-8. - 4. * (1. + 2. * b0) * x + (1. + 2. * b0) * powr<2>(x)) +
                2. * powr<2>(c) * (2. + (2. + 4. * b0) * x + b0 * (1. + 2. * b0) * powr<2>(x))) *
               powr<-2>(-2. + x + 2. * c * (1. + b0 * x));
      if constexpr (order == 3)
        return (1. + x + 2. * b0 * x +
                0.3333333333333333 * (-1. + 3. * c + b0 * (-3. + 6. * c) + 3. * (-1. + c) * powr<2>(b0)) *
                    powr<-1>(-1. + c) * powr<2>(x) +
                0.3333333333333333 * b0 * (2. + 3. * b0) * c * powr<-1>(-1. + c) * powr<3>(x) +
                0.027777777777777776 * c * powr<2>(2. + 3. * b0) * powr<-2>(-1. + c) * powr<4>(x)) *
               powr<-2>(1. + b0 * x + 0.16666666666666666 * (2. + 3. * b0) * powr<-1>(-1. + c) * powr<2>(x));
      if constexpr (order == 4)
        return (1. + x + (1. + 2. * b0) * powr<2>(x) +
                0.5 * (1. + 2. * b0) * (-1. + 2. * c) * powr<-1>(-1. + c) * powr<3>(x) +
                0.041666666666666664 * (-3. + 2. * b0 * (-7. + 4. * c) + 24. * (-1. + c) * powr<2>(b0)) *
                    powr<-1>(-1. + c) * powr<4>(x) +
                0.5 * b0 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<5>(x) +
                0.0625 * c * powr<2>(1. + 2. * b0) * powr<-2>(-1. + c) * powr<6>(x)) *
               powr<-2>(1. + b0 * powr<2>(x) + 0.25 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<3>(x));
      if constexpr (order == 5)
        return ((1. + b0 * powr<2>(x) + 0.06666666666666667 * (3. + 5. * b0) * powr<-1>(-1. + c) * powr<4>(x)) *
                    (1. + x + (1. + 3. * b0) * powr<2>(x) + (1. + 2. * b0) * powr<3>(x) +
                     0.3333333333333333 * (3. + 5. * b0) * c * powr<-1>(-1. + c) * powr<4>(x)) -
                0.016666666666666666 * x *
                    (2. * b0 * x + 0.26666666666666666 * (3. + 5. * b0) * powr<-1>(-1. + c) * powr<3>(x)) *
                    (60. + 30. * x + 20. * (1. + 3. * b0) * powr<2>(x) + 15. * (1. + 2. * b0) * powr<3>(x) +
                     4. * (3. + 5. * b0) * c * powr<-1>(-1. + c) * powr<4>(x))) *
               powr<-2>(1. + b0 * powr<2>(x) + 0.06666666666666667 * (3. + 5. * b0) * powr<-1>(-1. + c) * powr<4>(x));
      if constexpr (order == 6)
        return ((1. + b0 * powr<3>(x) + 0.16666666666666666 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<5>(x)) *
                    (1. + x + powr<2>(x) + (1. + 4. * b0) * powr<3>(x) + (1. + 2.5 * b0) * powr<4>(x) +
                     (c + 2. * b0 * c) * powr<-1>(-1. + c) * powr<5>(x)) -
                0.016666666666666666 * x *
                    (3. * b0 * powr<2>(x) + 0.8333333333333334 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<4>(x)) *
                    (60. + 30. * x + 20. * powr<2>(x) + 15. * (1. + 4. * b0) * powr<3>(x) +
                     6. * (2. + 5. * b0) * powr<4>(x) + 10. * (c + 2. * b0 * c) * powr<-1>(-1. + c) * powr<5>(x))) *
               powr<-2>(1. + b0 * powr<3>(x) + 0.16666666666666666 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<5>(x));
      if constexpr (order == 7)
        return ((1. + b0 * powr<3>(x) + 0.03571428571428571 * (4. + 7. * b0) * powr<-1>(-1. + c) * powr<6>(x)) *
                    (1. + x + powr<2>(x) + (1. + 4. * b0) * powr<3>(x) + (1. + 2.5 * b0) * powr<4>(x) +
                     (1. + 2. * b0) * powr<5>(x) + 0.25 * (4. + 7. * b0) * c * powr<-1>(-1. + c) * powr<6>(x)) -
                1. * (3. * b0 * powr<2>(x) + 0.21428571428571427 * (4. + 7. * b0) * powr<-1>(-1. + c) * powr<5>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + (0.25 + b0) * powr<4>(x) +
                     0.1 * (2. + 5. * b0) * powr<5>(x) + 0.16666666666666666 * (1. + 2. * b0) * powr<6>(x) +
                     0.03571428571428571 * (4. + 7. * b0) * c * powr<-1>(-1. + c) * powr<7>(x))) *
               powr<-2>(1. + b0 * powr<3>(x) + 0.03571428571428571 * (4. + 7. * b0) * powr<-1>(-1. + c) * powr<6>(x));
      if constexpr (order == 8)
        return ((1. + b0 * powr<4>(x) + 0.125 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<7>(x)) *
                    (1. + x + powr<2>(x) + powr<3>(x) + (1. + 5. * b0) * powr<4>(x) + (1. + 3. * b0) * powr<5>(x) +
                     (1. + 2.3333333333333335 * b0) * powr<6>(x) + (c + 2. * b0 * c) * powr<-1>(-1. + c) * powr<7>(x)) -
                1. * (4. * b0 * powr<3>(x) + 0.875 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<6>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) +
                     (0.2 + b0) * powr<5>(x) + 0.16666666666666666 * (1. + 3. * b0) * powr<6>(x) +
                     0.047619047619047616 * (3. + 7. * b0) * powr<7>(x) +
                     0.125 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<8>(x))) *
               powr<-2>(1. + b0 * powr<4>(x) + 0.125 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<7>(x));
      if constexpr (order == 9)
        return ((1. + b0 * powr<4>(x) + 0.022222222222222223 * (5. + 9. * b0) * powr<-1>(-1. + c) * powr<8>(x)) *
                    (1. + x + powr<2>(x) + powr<3>(x) + (1. + 5. * b0) * powr<4>(x) + (1. + 3. * b0) * powr<5>(x) +
                     (1. + 2.3333333333333335 * b0) * powr<6>(x) + (1. + 2. * b0) * powr<7>(x) +
                     0.2 * (5. + 9. * b0) * c * powr<-1>(-1. + c) * powr<8>(x)) -
                1. * (4. * b0 * powr<3>(x) + 0.17777777777777778 * (5. + 9. * b0) * powr<-1>(-1. + c) * powr<7>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) +
                     (0.2 + b0) * powr<5>(x) + 0.16666666666666666 * (1. + 3. * b0) * powr<6>(x) +
                     0.047619047619047616 * (3. + 7. * b0) * powr<7>(x) + 0.125 * (1. + 2. * b0) * powr<8>(x) +
                     0.022222222222222223 * (5. + 9. * b0) * c * powr<-1>(-1. + c) * powr<9>(x))) *
               powr<-2>(1. + b0 * powr<4>(x) + 0.022222222222222223 * (5. + 9. * b0) * powr<-1>(-1. + c) * powr<8>(x));
      if constexpr (order == 10)
        return ((1. + b0 * powr<5>(x) + 0.1 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<9>(x)) *
                    (1. + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + (1. + 6. * b0) * powr<5>(x) +
                     (1. + 3.5 * b0) * powr<6>(x) + (1. + 2.6666666666666665 * b0) * powr<7>(x) +
                     (1. + 2.25 * b0) * powr<8>(x) + (c + 2. * b0 * c) * powr<-1>(-1. + c) * powr<9>(x)) -
                1. * (5. * b0 * powr<4>(x) + 0.9 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<8>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                     (0.16666666666666666 + b0) * powr<6>(x) + 0.07142857142857142 * (2. + 7. * b0) * powr<7>(x) +
                     0.041666666666666664 * (3. + 8. * b0) * powr<8>(x) +
                     0.027777777777777776 * (4. + 9. * b0) * powr<9>(x) +
                     0.1 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<10>(x))) *
               powr<-2>(1. + b0 * powr<5>(x) + 0.1 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<9>(x));
      if constexpr (order == 11)
        return ((1. + b0 * powr<5>(x) + 0.015151515151515152 * (6. + 11. * b0) * powr<-1>(-1. + c) * powr<10>(x)) *
                    (1. + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + (1. + 6. * b0) * powr<5>(x) +
                     (1. + 3.5 * b0) * powr<6>(x) + (1. + 2.6666666666666665 * b0) * powr<7>(x) +
                     (1. + 2.25 * b0) * powr<8>(x) + (1. + 2. * b0) * powr<9>(x) +
                     0.16666666666666666 * (6. + 11. * b0) * c * powr<-1>(-1. + c) * powr<10>(x)) -
                1. * (5. * b0 * powr<4>(x) + 0.15151515151515152 * (6. + 11. * b0) * powr<-1>(-1. + c) * powr<9>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                     (0.16666666666666666 + b0) * powr<6>(x) + 0.07142857142857142 * (2. + 7. * b0) * powr<7>(x) +
                     0.041666666666666664 * (3. + 8. * b0) * powr<8>(x) +
                     0.027777777777777776 * (4. + 9. * b0) * powr<9>(x) + 0.1 * (1. + 2. * b0) * powr<10>(x) +
                     0.015151515151515152 * (6. + 11. * b0) * c * powr<-1>(-1. + c) * powr<11>(x))) *
               powr<-2>(1. + b0 * powr<5>(x) +
                        0.015151515151515152 * (6. + 11. * b0) * powr<-1>(-1. + c) * powr<10>(x));
      if constexpr (order == 12)
        return ((1. + b0 * powr<6>(x) + 0.08333333333333333 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<11>(x)) *
                    (1. + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + (1. + 7. * b0) * powr<6>(x) +
                     (1. + 4. * b0) * powr<7>(x) + (1. + 3. * b0) * powr<8>(x) + (1. + 2.5 * b0) * powr<9>(x) +
                     (1. + 2.2 * b0) * powr<10>(x) + (c + 2. * b0 * c) * powr<-1>(-1. + c) * powr<11>(x)) -
                1. * (6. * b0 * powr<5>(x) + 0.9166666666666666 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<10>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                     0.16666666666666666 * powr<6>(x) + (0.14285714285714285 + b0) * powr<7>(x) +
                     0.125 * (1. + 4. * b0) * powr<8>(x) + 0.1111111111111111 * (1. + 3. * b0) * powr<9>(x) +
                     0.05 * (2. + 5. * b0) * powr<10>(x) + 0.01818181818181818 * (5. + 11. * b0) * powr<11>(x) +
                     0.08333333333333333 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<12>(x))) *
               powr<-2>(1. + b0 * powr<6>(x) + 0.08333333333333333 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<11>(x));
      if constexpr (order == 13)
        return ((1. + b0 * powr<6>(x) + 0.01098901098901099 * (7. + 13. * b0) * powr<-1>(-1. + c) * powr<12>(x)) *
                    (1. + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + (1. + 7. * b0) * powr<6>(x) +
                     (1. + 4. * b0) * powr<7>(x) + (1. + 3. * b0) * powr<8>(x) + (1. + 2.5 * b0) * powr<9>(x) +
                     (1. + 2.2 * b0) * powr<10>(x) + (1. + 2. * b0) * powr<11>(x) +
                     0.14285714285714285 * (7. + 13. * b0) * c * powr<-1>(-1. + c) * powr<12>(x)) -
                1. * (6. * b0 * powr<5>(x) + 0.13186813186813187 * (7. + 13. * b0) * powr<-1>(-1. + c) * powr<11>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                     0.16666666666666666 * powr<6>(x) + (0.14285714285714285 + b0) * powr<7>(x) +
                     0.125 * (1. + 4. * b0) * powr<8>(x) + 0.1111111111111111 * (1. + 3. * b0) * powr<9>(x) +
                     0.05 * (2. + 5. * b0) * powr<10>(x) + 0.01818181818181818 * (5. + 11. * b0) * powr<11>(x) +
                     0.08333333333333333 * (1. + 2. * b0) * powr<12>(x) +
                     0.01098901098901099 * (7. + 13. * b0) * c * powr<-1>(-1. + c) * powr<13>(x))) *
               powr<-2>(1. + b0 * powr<6>(x) + 0.01098901098901099 * (7. + 13. * b0) * powr<-1>(-1. + c) * powr<12>(x));
      if constexpr (order == 14)
        return ((1. + b0 * powr<7>(x) + 0.07142857142857142 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<13>(x)) *
                    (1. + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + powr<6>(x) +
                     (1. + 8. * b0) * powr<7>(x) + (1. + 4.5 * b0) * powr<8>(x) +
                     (1. + 3.3333333333333335 * b0) * powr<9>(x) + (1. + 2.75 * b0) * powr<10>(x) +
                     (1. + 2.4 * b0) * powr<11>(x) + (1. + 2.1666666666666665 * b0) * powr<12>(x) +
                     (c + 2. * b0 * c) * powr<-1>(-1. + c) * powr<13>(x)) -
                1. * (7. * b0 * powr<6>(x) + 0.9285714285714286 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<12>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                     0.16666666666666666 * powr<6>(x) + 0.14285714285714285 * powr<7>(x) + (0.125 + b0) * powr<8>(x) +
                     0.05555555555555555 * (2. + 9. * b0) * powr<9>(x) +
                     0.03333333333333333 * (3. + 10. * b0) * powr<10>(x) +
                     0.022727272727272728 * (4. + 11. * b0) * powr<11>(x) +
                     0.016666666666666666 * (5. + 12. * b0) * powr<12>(x) +
                     0.01282051282051282 * (6. + 13. * b0) * powr<13>(x) +
                     0.07142857142857142 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<14>(x))) *
               powr<-2>(1. + b0 * powr<7>(x) + 0.07142857142857142 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<13>(x));
      if constexpr (order == 15)
        return ((1. + b0 * powr<7>(x) + 0.008333333333333333 * (8. + 15. * b0) * powr<-1>(-1. + c) * powr<14>(x)) *
                    (1. + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + powr<6>(x) +
                     (1. + 8. * b0) * powr<7>(x) + (1. + 4.5 * b0) * powr<8>(x) +
                     (1. + 3.3333333333333335 * b0) * powr<9>(x) + (1. + 2.75 * b0) * powr<10>(x) +
                     (1. + 2.4 * b0) * powr<11>(x) + (1. + 2.1666666666666665 * b0) * powr<12>(x) +
                     (1. + 2. * b0) * powr<13>(x) + 0.125 * (8. + 15. * b0) * c * powr<-1>(-1. + c) * powr<14>(x)) -
                1. * (7. * b0 * powr<6>(x) + 0.11666666666666667 * (8. + 15. * b0) * powr<-1>(-1. + c) * powr<13>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                     0.16666666666666666 * powr<6>(x) + 0.14285714285714285 * powr<7>(x) + (0.125 + b0) * powr<8>(x) +
                     0.05555555555555555 * (2. + 9. * b0) * powr<9>(x) +
                     0.03333333333333333 * (3. + 10. * b0) * powr<10>(x) +
                     0.022727272727272728 * (4. + 11. * b0) * powr<11>(x) +
                     0.016666666666666666 * (5. + 12. * b0) * powr<12>(x) +
                     0.01282051282051282 * (6. + 13. * b0) * powr<13>(x) +
                     0.07142857142857142 * (1. + 2. * b0) * powr<14>(x) +
                     0.008333333333333333 * (8. + 15. * b0) * c * powr<-1>(-1. + c) * powr<15>(x))) *
               powr<-2>(1. + b0 * powr<7>(x) +
                        0.008333333333333333 * (8. + 15. * b0) * powr<-1>(-1. + c) * powr<14>(x));
      if constexpr (order == 16)
        return ((1. + b0 * powr<8>(x) + 0.0625 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<15>(x)) *
                    (1. + x + powr<2>(x) + powr<3>(x) + powr<4>(x) + powr<5>(x) + powr<6>(x) + powr<7>(x) +
                     (1. + 9. * b0) * powr<8>(x) + (1. + 5. * b0) * powr<9>(x) +
                     (1. + 3.6666666666666665 * b0) * powr<10>(x) + (1. + 3. * b0) * powr<11>(x) +
                     (1. + 2.6 * b0) * powr<12>(x) + (1. + 2.3333333333333335 * b0) * powr<13>(x) +
                     (1. + 2.142857142857143 * b0) * powr<14>(x) +
                     (c + 2. * b0 * c) * powr<-1>(-1. + c) * powr<15>(x)) -
                1. * (8. * b0 * powr<7>(x) + 0.9375 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<14>(x)) *
                    (x + 0.5 * powr<2>(x) + 0.3333333333333333 * powr<3>(x) + 0.25 * powr<4>(x) + 0.2 * powr<5>(x) +
                     0.16666666666666666 * powr<6>(x) + 0.14285714285714285 * powr<7>(x) + 0.125 * powr<8>(x) +
                     (0.1111111111111111 + b0) * powr<9>(x) + 0.1 * (1. + 5. * b0) * powr<10>(x) +
                     0.030303030303030304 * (3. + 11. * b0) * powr<11>(x) +
                     0.08333333333333333 * (1. + 3. * b0) * powr<12>(x) +
                     0.015384615384615385 * (5. + 13. * b0) * powr<13>(x) +
                     0.023809523809523808 * (3. + 7. * b0) * powr<14>(x) +
                     (0.06666666666666667 + 0.14285714285714285 * b0) * powr<15>(x) +
                     0.0625 * (1. + 2. * b0) * c * powr<-1>(-1. + c) * powr<16>(x))) *
               powr<-2>(1. + b0 * powr<8>(x) + 0.0625 * (1. + 2. * b0) * powr<-1>(-1. + c) * powr<15>(x));
      static_assert(order <= 16, "Rational regulator of order > 16 not implemented");
    }

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto f = get_f(x);
      return k2 * exp(-f);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RBdot(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto f = get_f(x);
      const auto df = get_df(x);
      return 2. * k2 * exp(-f) * (1. + df * x);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto dq2RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto f = get_f(x);
      const auto df = get_df(x);
      return -exp(-f) * df;
    }

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RFdot(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return 0.5 * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto dq2RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      const auto q = sqrt(q2);
      return (-1. + q * (1. + dq2RB(k2, q2)) / (q + RF(k2, q2))) / (2. * q);
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
    static __forceinline__ __device__ __host__ auto get_f(std::integer_sequence<int, c0, c...>, const auto x)
    {
      using T = decltype(x);
      if constexpr (sizeof...(c) == 0)
        return powr<c0 + 1>(x) / T(c0 + 1);
      else
        return powr<c0 + 1>(x) / T(c0 + 1) + get_f(std::integer_sequence<int, c...>{}, x);
    }
    template <int c0, int... c>
    static __forceinline__ __device__ __host__ auto get_df(std::integer_sequence<int, c0, c...>, const auto x)
    {
      if constexpr (sizeof...(c) == 0)
        return powr<c0>(x);
      else
        return powr<c0>(x) + get_df(std::integer_sequence<int, c...>{}, x);
    }
    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto f = get_f(std::make_integer_sequence<int, order>{}, x);
      return k2 * exp(-f);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RBdot(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto f = get_f(std::make_integer_sequence<int, order>{}, x);
      const auto df = get_df(std::make_integer_sequence<int, order>{}, x);
      return 2 * k2 * exp(-f) * (1 + df * x);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto dq2RB(const NT1 k2, const NT2 q2)
    {
      using std::exp;
      const auto x = q2 / k2;
      const auto f = get_f(std::make_integer_sequence<int, order>{}, x);
      const auto df = get_df(std::make_integer_sequence<int, order>{}, x);
      return -exp(-f) * df;
    }

    template <typename NT1, typename NT2> static __forceinline__ __device__ __host__ auto RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      return sqrt(RB(k2, q2) + q2) - sqrt(q2);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto RFdot(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      using T = decltype(k2 * q2);
      return T(0.5) * RBdot(k2, q2) / sqrt(RB(k2, q2) + q2);
    }

    template <typename NT1, typename NT2>
    static __forceinline__ __device__ __host__ auto dq2RF(const NT1 k2, const NT2 q2)
    {
      using std::sqrt;
      const auto q = sqrt(q2);
      return (-1 + q * (1 + dq2RB(k2, q2)) / (q + RF(k2, q2))) / (2 * q);
    }
  };
} // namespace DiFfRG
