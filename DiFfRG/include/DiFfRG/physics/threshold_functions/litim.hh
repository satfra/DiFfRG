#pragma once

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/thermodynamics.hh>

namespace DiFfRG
{
  namespace fRG
  {
    /**
     * @brief Threshold functions at finite temperature for the Litim regulator on spatial momenta.
     *
     * With a spatial Litim regulator, \f$R(\vec q^{\,2}) = (k^2 - \vec q^{\,2})\,\Theta(k^2 - \vec q^{\,2})\f$, the
     * spatial loop integral is trivial and the Matsubara sum is done analytically, so the loop integrals of the local
     * potential approximation become closed-form functions. Conventions for all functions here:
     * - mass arguments (`mb2`, `mf2`, ...) are the dimensionless \f$m^2/k^2\f$ and enter through
     *   \f$\epsilon = \sqrt{1 + m^2/k^2}\f$;
     * - `k`, `T` and `mu` (\f$\mu\f$, quark chemical potential) are dimensionful;
     * - `d` is the spacetime dimension (spatial dimension \f$d - 1\f$);
     * - \f$\omega_n = 2\pi n T\f$ and \f$\nu_n = (2n+1)\pi T\f$ are bosonic and fermionic Matsubara frequencies,
     *   \f$\sum_n\f$ runs over all \f$n \in \mathbb{Z}\f$.
     *
     * The functions are host-only. Orders that are not implemented throw `std::runtime_error` at run time. For the
     * explicit expressions, see the appendix of https://arxiv.org/abs/1909.02991.
     */
    namespace TFLitimSpatial
    {
      static constexpr auto Pi = M_PI;
      static constexpr auto Pi2 = Pi * Pi;
      using Kokkos::sqrt, Kokkos::pow;

      /**
       * @brief Bosonic Matsubara sum with `nb` propagators.
       *
       * \f[
       *   B_{n}(m^2) = k^{2n-1}\, T \sum_{j} \frac{1}{\big(\omega_j^2 + k^2 \epsilon^2\big)^{n}}\,,
       *   \qquad B_1 = \frac{\coth(k\epsilon/2T)}{2\epsilon}\,,
       *   \qquad B_{n+1} = -\frac{1}{n} \frac{\partial B_n}{\partial (m^2/k^2)}\,.
       * \f]
       * Implemented for \f$1 \le\f$ `nb` \f$\le 4\f$.
       *
       * @param mb2 dimensionless boson mass \f$m^2/k^2\f$
       * @param k RG scale
       * @param T temperature
       */
      template <int nb, typename NT> NT B(const NT &mb2, const double &k, const double &T)
      {
        if constexpr (nb == 1)
          return (CothFiniteT(k * powr<1>(sqrt(1 + mb2)), T) * powr<-1>(sqrt(1 + mb2))) / 2.;
        else if constexpr (nb == 2)
          return (powr<-1>(T) * powr<2>(CschFiniteT((k * powr<1>(sqrt(1 + mb2))), T)) * powr<-3>(sqrt(1 + mb2)) *
                  (k * powr<1>(sqrt(1 + mb2)) + T * Sinh(k * powr<-1>(T) * powr<1>(sqrt(1 + mb2))))) /
                 8.;
        else if constexpr (nb == 3)
          return (powr<-3>(1 + mb2) * powr<-2>(T) * powr<2>(CschFiniteT((k * powr<1>(sqrt(1 + mb2))), T)) *
                  (3 * k * (1 + mb2) * T + CothFiniteT((k * powr<1>(sqrt(1 + mb2))), T) *
                                               ((1 + mb2) * powr<2>(k) - 3 * powr<2>(T) +
                                                3 * Cosh(k * powr<-1>(T) * powr<1>(sqrt(1 + mb2))) * powr<2>(T)) *
                                               powr<1>(sqrt(1 + mb2)))) /
                 32.;
        else if constexpr (nb == 4)
          return (powr<-3>(T) * powr<4>(CschFiniteT((k * powr<1>(sqrt(1 + mb2))), T)) * powr<-7>(sqrt(1 + mb2)) *
                  (2 * k * (2 * (1 + mb2) * powr<2>(k) - 15 * powr<2>(T)) * powr<1>(sqrt(1 + mb2)) +
                   2 * k * Cosh(k * powr<-1>(T) * powr<1>(sqrt(1 + mb2))) * ((1 + mb2) * powr<2>(k) + 15 * powr<2>(T)) *
                       powr<1>(sqrt(1 + mb2)) +
                   6 * T *
                       (2 * (1 + mb2) * powr<2>(k) - 5 * powr<2>(T) +
                        5 * Cosh(k * powr<-1>(T) * powr<1>(sqrt(1 + mb2))) * powr<2>(T)) *
                       Sinh(k * powr<-1>(T) * powr<1>(sqrt(1 + mb2))))) /
                 768.;
        else
          throw std::runtime_error("Threshold Function B is not implemented for given indices");
      }

      /**
       * @brief Fermionic Matsubara sum with `nf` propagators at chemical potential \f$\mu\f$.
       *
       * \f[
       *   F_{n}(m^2) = k^{2n-1}\, T \sum_{j} \frac{1}{\big((\nu_j + i\mu)^2 + k^2 \epsilon^2\big)^{n}}\,,
       *   \qquad F_1 = \frac{\tanh\frac{k\epsilon - \mu}{2T} + \tanh\frac{k\epsilon + \mu}{2T}}{4\epsilon}\,,
       *   \qquad F_{n+1} = -\frac{1}{n} \frac{\partial F_n}{\partial (m^2/k^2)}\,.
       * \f]
       * Implemented for \f$1 \le\f$ `nf` \f$\le 3\f$.
       *
       * @param mf2 dimensionless fermion mass \f$m^2/k^2\f$
       * @param k RG scale
       * @param T temperature
       * @param mu chemical potential
       */
      template <int nf, typename NT> NT F(const NT &mf2, const double &k, const double &T, const double &mu)
      {
        if constexpr (nf == 1)
          return (powr<-1>(sqrt(1 + mf2)) * (TanhFiniteT(-mu + k * powr<1>(sqrt(1 + mf2)), T) +
                                             TanhFiniteT(mu + k * powr<1>(sqrt(1 + mf2)), T))) /
                 4.;
        else if constexpr (nf == 2)
          return -0.0625 * (k * powr<-1>(1 + mf2) * powr<-1>(T) *
                            (powr<2>(SechFiniteT(((-mu + k * powr<1>(sqrt(1 + mf2)))), T)) +
                             powr<2>(SechFiniteT(((mu + k * powr<1>(sqrt(1 + mf2)))), T)))) +
                 (powr<-3>(sqrt(1 + mf2)) * (TanhFiniteT(((-mu + k * powr<1>(sqrt(1 + mf2)))), T) +
                                             TanhFiniteT(((mu + k * powr<1>(sqrt(1 + mf2)))), T))) /
                     8.;
        else if constexpr (nf == 3)
          return -0.0078125 *
                 (powr<-3>(1 + mf2) * powr<-2>(T) *
                  (-3 * powr<2>(T) * powr<3>(SechFiniteT(((-mu + k * powr<1>(sqrt(1 + mf2)))), T)) *
                       powr<1>(sqrt(1 + mf2)) * Sinh((3 * (-mu + k * powr<1>(sqrt(1 + mf2)))) / T * 0.5) +
                   powr<2>(SechFiniteT(((-mu + k * powr<1>(sqrt(1 + mf2)))), T)) *
                       (6 * k * (1. + mf2) * T + (2 * (1 + mf2) * powr<2>(k) - 3 * powr<2>(T)) *
                                                     powr<1>(sqrt(1 + mf2)) *
                                                     TanhFiniteT(((-mu + k * powr<1>(sqrt(1 + mf2)))), T)) +
                   2. * powr<2>(SechFiniteT(((mu + k * powr<1>(sqrt(1 + mf2)))), T)) *
                       (3. * k * (1. + mf2) * T +
                        ((1 + mf2) * powr<2>(k) - 3 * powr<2>(T) -
                         3. * Cosh((mu + k * powr<1>(sqrt(1. + mf2))) / T) * powr<2>(T)) *
                            powr<1>(sqrt(1. + mf2)) * TanhFiniteT(((mu + k * powr<1>(sqrt(1. + mf2)))), T))));
        else
          throw std::runtime_error("Threshold Function F is not implemented for given indices");
      }

      /**
       * @brief Bosonic Matsubara sum with two different masses.
       *
       * \f[
       *   BB_{n_1 n_2} = k^{2(n_1+n_2)-1}\, T \sum_{j}
       *   \frac{1}{\big(\omega_j^2 + k^2 \epsilon_1^2\big)^{n_1} \big(\omega_j^2 + k^2 \epsilon_2^2\big)^{n_2}}\,.
       * \f]
       * Equal masses fall back to \f$B_{n_1+n_2}\f$. Otherwise only \f$n_1 = n_2 = 2\f$ is implemented. The result type
       * `NT` must be given explicitly.
       *
       * @param mb12, mb22 dimensionless boson masses \f$m_1^2/k^2\f$, \f$m_2^2/k^2\f$
       * @param k RG scale
       * @param T temperature
       */
      template <int nb1, int nb2, typename NT, typename NT1, typename NT2>
      NT BB(const NT1 &mb12, const NT2 &mb22, const double &k, const double &T)
      {
        if (is_close(double(mb12), double(mb22))) return B<nb1 + nb2, NT>(mb12, k, T);
        if constexpr (nb1 == 2 && nb2 == 2)
          return (powr<-1>(T) * (powr<-3>(mb12 - mb22) * powr<2>(CschFiniteT((k * powr<1>(sqrt(1 + mb12))), T)) *
                                     powr<-3>(sqrt(1 + mb12)) *
                                     (k * (mb12 - mb22) * powr<1>(sqrt(1 + mb12)) +
                                      (4 + 5 * mb12 - mb22) * T * Sinh(k * powr<-1>(T) * powr<1>(sqrt(1 + mb12)))) +
                                 powr<-3>(-mb12 + mb22) * powr<2>(CschFiniteT((k * powr<1>(sqrt(1 + mb22))), T)) *
                                     powr<-3>(sqrt(1 + mb22)) *
                                     (k * (-mb12 + mb22) * powr<1>(sqrt(1 + mb22)) -
                                      (-4 + mb12 - 5 * mb22) * T * Sinh(k * powr<-1>(T) * powr<1>(sqrt(1 + mb22)))))) /
                 8.;
        else
          throw std::runtime_error("Threshold Function BB is not implemented for given indices");
      }

      /**
       * @brief Mixed Matsubara sum with `nf` fermion and `nb` boson propagators, at external fermion momentum
       * \f$p = (\pi T, \vec 0)\f$ (the lowest fermionic frequency).
       *
       * Implemented for (`nf`, `nb`) = (1, 2) and (2, 1). The result type `NT` must be given explicitly.
       *
       * @param mf2 dimensionless fermion mass \f$m_f^2/k^2\f$
       * @param mb2 dimensionless boson mass \f$m_b^2/k^2\f$
       * @param k RG scale
       * @param T temperature
       * @param mu chemical potential
       */
      template <int nf, int nb, typename NT, typename NTf, typename NTb>
      NT FBFermiPiT(const NTf &mf2, const NTb &mb2, const double &k, const double &T, const double &mu)
      {
        if constexpr (nf == 1 && nb == 2)
          return (powr<-1>(1 + mb2) * powr<-1>(T) * powr<2>(CschFiniteT((k * powr<1>(sqrt(1 + mb2))), T)) *
                  (-(powr<9>(k) * powr<3>(mb2 - mf2)) -
                   powr<7>(k) * (2 * mb2 * (2 + mf2) - mf2 * (4 + 3 * mf2) + powr<2>(mb2)) *
                       (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                   powr<5>(k) * ((4 + mb2 + 3 * mf2) * powr<4>(mu) +
                                 2 * (4 + 5 * mb2 - mf2) * powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                                 (4 + mb2 + 3 * mf2) * powr<4>(Pi) * powr<4>(T)) +
                   powr<3>(k) * (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) *
                       powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T))) *
                  powr<-1>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 + 3 * mb2 - mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (-mb2 + mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mb2)) -
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mb2))) *
                  powr<-1>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 + 3 * mb2 - mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (mb2 - mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mb2)) +
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mb2)))) /
                     8. +
                 (CothFiniteT((k * powr<1>(sqrt(1 + mb2))), T) * powr<2>(k) *
                  (-((2 + 3 * mb2 - mf2) * powr<14>(k) * powr<6>(mb2 - mf2)) -
                   powr<12>(k) * powr<4>(mb2 - mf2) *
                       (8 - 8 * mf2 + 6 * mb2 * (4 + mf2) + 9 * powr<2>(mb2) - 7 * powr<2>(mf2)) *
                       (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                   powr<10>(k) * powr<2>(mb2 - mf2) *
                       ((32 + 80 * mf2 + 3 * (2 + mf2) * powr<2>(mb2) + powr<3>(mb2) + 70 * powr<2>(mf2) +
                         mb2 * (16 + 20 * mf2 + 7 * powr<2>(mf2)) + 21 * powr<3>(mf2)) *
                            powr<4>(mu) +
                        2 *
                            (160 + 144 * mf2 + (206 + 71 * mf2) * powr<2>(mb2) + 45 * powr<3>(mb2) + 14 * powr<2>(mf2) +
                             mb2 * (336 + 260 * mf2 + 59 * powr<2>(mf2)) - 15 * powr<3>(mf2)) *
                            powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                        (32 + 80 * mf2 + 3 * (2 + mf2) * powr<2>(mb2) + powr<3>(mb2) + 70 * powr<2>(mf2) +
                         mb2 * (16 + 20 * mf2 + 7 * powr<2>(mf2)) + 21 * powr<3>(mf2)) *
                            powr<4>(Pi) * powr<4>(T)) +
                   powr<8>(k) * (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) *
                       ((128 + 256 * mf2 + 20 * (8 + mf2) * powr<3>(mb2) + 35 * powr<4>(mb2) + 288 * powr<2>(mf2) +
                         6 * powr<2>(mb2) * (48 + 16 * mf2 + 3 * powr<2>(mf2)) + 160 * powr<3>(mf2) +
                         4 * mb2 * (64 + 48 * mf2 + 24 * powr<2>(mf2) + 5 * powr<3>(mf2)) + 35 * powr<4>(mf2)) *
                            powr<4>(mu) +
                        2 *
                            (128 + 256 * mf2 + (416 - 76 * mf2) * powr<3>(mb2) + 123 * powr<4>(mb2) +
                             powr<2>(mb2) * (416 - 416 * mf2 - 94 * powr<2>(mf2)) + 416 * powr<2>(mf2) +
                             160 * powr<3>(mf2) + 4 * mb2 * (64 - 16 * mf2 + 88 * powr<2>(mf2) + 45 * powr<3>(mf2)) -
                             5 * powr<4>(mf2)) *
                            powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                        (128 + 256 * mf2 + 20 * (8 + mf2) * powr<3>(mb2) + 35 * powr<4>(mb2) + 288 * powr<2>(mf2) +
                         6 * powr<2>(mb2) * (48 + 16 * mf2 + 3 * powr<2>(mf2)) + 160 * powr<3>(mf2) +
                         4 * mb2 * (64 + 48 * mf2 + 24 * powr<2>(mf2) + 5 * powr<3>(mf2)) + 35 * powr<4>(mf2)) *
                            powr<4>(Pi) * powr<4>(T)) +
                   powr<6>(k) *
                       ((224 + 304 * mf2 + 3 * (78 + 23 * mf2) * powr<2>(mb2) + 55 * powr<3>(mb2) + 170 * powr<2>(mf2) +
                         mb2 * (368 + 268 * mf2 + 65 * powr<2>(mf2)) + 35 * powr<3>(mf2)) *
                            powr<8>(mu) +
                        4 *
                            (96 + 112 * mf2 + (122 - 19 * mf2) * powr<2>(mb2) + 47 * powr<3>(mb2) + 58 * powr<2>(mf2) +
                             mb2 * (176 + 108 * mf2 + 73 * powr<2>(mf2)) - 5 * powr<3>(mf2)) *
                            powr<6>(mu) * powr<2>(Pi) * powr<2>(T) +
                        2 *
                            (160 + 144 * mf2 + 3 * (42 + 101 * mf2) * powr<2>(mb2) - 59 * powr<3>(mb2) +
                             mb2 * (336 + 420 * mf2 - 93 * powr<2>(mf2)) - 66 * powr<2>(mf2) + 9 * powr<3>(mf2)) *
                            powr<4>(mu) * powr<4>(Pi) * powr<4>(T) +
                        4 *
                            (96 + 112 * mf2 + (122 - 19 * mf2) * powr<2>(mb2) + 47 * powr<3>(mb2) + 58 * powr<2>(mf2) +
                             mb2 * (176 + 108 * mf2 + 73 * powr<2>(mf2)) - 5 * powr<3>(mf2)) *
                            powr<2>(mu) * powr<6>(Pi) * powr<6>(T) +
                        (224 + 304 * mf2 + 3 * (78 + 23 * mf2) * powr<2>(mb2) + 55 * powr<3>(mb2) + 170 * powr<2>(mf2) +
                         mb2 * (368 + 268 * mf2 + 65 * powr<2>(mf2)) + 35 * powr<3>(mf2)) *
                            powr<8>(Pi) * powr<8>(T)) +
                   powr<4>(k) * (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) *
                       ((104 + 88 * mf2 + 2 * mb2 * (60 + 23 * mf2) + 37 * powr<2>(mb2) + 21 * powr<2>(mf2)) *
                            powr<4>(mu) +
                        2 * (40 + 24 * mf2 + mb2 * (56 + 30 * mf2) + 13 * powr<2>(mb2) - 3 * powr<2>(mf2)) *
                            powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                        (104 + 88 * mf2 + 2 * mb2 * (60 + 23 * mf2) + 37 * powr<2>(mb2) + 21 * powr<2>(mf2)) *
                            powr<4>(Pi) * powr<4>(T)) *
                       powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                   powr<2>(k) *
                       ((18 + 11 * mb2 + 7 * mf2) * powr<4>(mu) -
                        2 * (22 + 17 * mb2 + 5 * mf2) * powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                        (18 + 11 * mb2 + 7 * mf2) * powr<4>(Pi) * powr<4>(T)) *
                       powr<4>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                   (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<6>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T))) *
                  powr<-2>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 + 3 * mb2 - mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (-mb2 + mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mb2)) -
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mb2))) *
                  powr<-2>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 + 3 * mb2 - mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (mb2 - mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mb2)) +
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mb2))) *
                  powr<-3>(sqrt(1 + mb2))) /
                     4. +
                 (powr<4>(k) *
                  powr<-2>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 - mb2 + 3 * mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (mb2 - mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mf2)) -
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mf2))) *
                  powr<-1>(sqrt(1 + mf2)) *
                  (powr<4>(k) * powr<2>(mb2 - mf2) + powr<4>(mu) - 6 * powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                   2 * (-2 + mb2 - 3 * mf2) * powr<2>(k) * (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                   powr<4>(Pi) * powr<4>(T) + 4 * (mb2 - mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mf2)) -
                   4 * k * mu * (powr<2>(mu) - 3 * powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mf2))) *
                  Tanh((powr<-1>(T) * (-mu + k * powr<1>(sqrt(1 + mf2)))) / 2.)) /
                     4. +
                 (powr<4>(k) *
                  powr<-2>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 - mb2 + 3 * mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (-mb2 + mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mf2)) +
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mf2))) *
                  powr<-1>(sqrt(1 + mf2)) *
                  (powr<4>(k) * powr<2>(mb2 - mf2) + powr<4>(mu) - 6 * powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                   2 * (-2 + mb2 - 3 * mf2) * powr<2>(k) * (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                   powr<4>(Pi) * powr<4>(T) + 4 * (-mb2 + mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mf2)) +
                   4 * k * mu * (powr<2>(mu) - 3 * powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mf2))) *
                  Tanh((powr<-1>(T) * (mu + k * powr<1>(sqrt(1 + mf2)))) / 2.)) /
                     4.;
        else if constexpr (nf == 2 && nb == 1)
          return (CothFiniteT((k * powr<1>(sqrt(1 + mb2))), T) * powr<4>(k) *
                  (powr<12>(k) * powr<6>(mb2 - mf2) -
                   2 * (-2 + mb2 - 3 * mf2) * powr<10>(k) * powr<4>(mb2 - mf2) *
                       (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) -
                   powr<8>(k) * powr<2>(mb2 - mf2) *
                       ((16 - 16 * mf2 + 2 * mb2 * (24 + 7 * mf2) + 17 * powr<2>(mb2) - 15 * powr<2>(mf2)) *
                            powr<4>(mu) +
                        2 * (80 + 48 * mf2 + 2 * mb2 * (56 + 11 * mf2) + 45 * powr<2>(mb2) + 13 * powr<2>(mf2)) *
                            powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                        (16 - 16 * mf2 + 2 * mb2 * (24 + 7 * mf2) + 17 * powr<2>(mb2) - 15 * powr<2>(mf2)) *
                            powr<4>(Pi) * powr<4>(T)) -
                   4 * powr<6>(k) * (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) *
                       ((16 + 8 * mf2 + (26 + 5 * mf2) * powr<2>(mb2) + 7 * powr<3>(mb2) - 6 * powr<2>(mf2) +
                         mb2 * (40 + 28 * mf2 + 9 * powr<2>(mf2)) - 5 * powr<3>(mf2)) *
                            powr<4>(mu) +
                        2 *
                            (16 + 8 * mf2 - 3 * (-14 + mf2) * powr<2>(mb2) + 15 * powr<3>(mb2) + 10 * powr<2>(mf2) +
                             mb2 * (40 - 4 * mf2 + powr<2>(mf2)) + 3 * powr<3>(mf2)) *
                            powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                        (16 + 8 * mf2 + (26 + 5 * mf2) * powr<2>(mb2) + 7 * powr<3>(mb2) - 6 * powr<2>(mf2) +
                         mb2 * (40 + 28 * mf2 + 9 * powr<2>(mf2)) - 5 * powr<3>(mf2)) *
                            powr<4>(Pi) * powr<4>(T)) +
                   powr<4>(k) *
                       (-((16 - 16 * mf2 + 2 * mb2 * (24 + 7 * mf2) + 17 * powr<2>(mb2) - 15 * powr<2>(mf2)) *
                          powr<8>(mu)) +
                        4 * (48 + 16 * mf2 + mb2 * (80 + 34 * mf2) + 23 * powr<2>(mb2) - 9 * powr<2>(mf2)) *
                            powr<6>(mu) * powr<2>(Pi) * powr<2>(T) +
                        2 * (208 + 48 * mf2 + mb2 * (368 + 22 * mf2) + 173 * powr<2>(mb2) + 13 * powr<2>(mf2)) *
                            powr<4>(mu) * powr<4>(Pi) * powr<4>(T) +
                        4 * (48 + 16 * mf2 + mb2 * (80 + 34 * mf2) + 23 * powr<2>(mb2) - 9 * powr<2>(mf2)) *
                            powr<2>(mu) * powr<6>(Pi) * powr<6>(T) -
                        (16 - 16 * mf2 + 2 * mb2 * (24 + 7 * mf2) + 17 * powr<2>(mb2) - 15 * powr<2>(mf2)) *
                            powr<8>(Pi) * powr<8>(T)) -
                   2 * powr<2>(k) * (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) *
                       ((-2 + mb2 - 3 * mf2) * powr<4>(mu) -
                        2 * (18 + 23 * mb2 - 5 * mf2) * powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                        (-2 + mb2 - 3 * mf2) * powr<4>(Pi) * powr<4>(T)) *
                       powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                   (powr<4>(mu) - 6 * powr<2>(mu) * powr<2>(Pi) * powr<2>(T) + powr<4>(Pi) * powr<4>(T)) *
                       powr<4>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T))) *
                  powr<-2>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 + 3 * mb2 - mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (-mb2 + mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mb2)) -
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mb2))) *
                  powr<-2>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 + 3 * mb2 - mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (mb2 - mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mb2)) +
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mb2))) *
                  powr<-1>(sqrt(1 + mb2))) /
                     2. +
                 (powr<3>(k) * powr<-1>(1 + mf2) * powr<-1>(T) *
                  powr<-1>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 - mb2 + 3 * mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (mb2 - mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mf2)) -
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mf2))) *
                  powr<2>(Sech((powr<-1>(T) * (-mu + k * powr<1>(sqrt(1 + mf2)))) / 2.)) *
                  ((-mb2 + mf2) * powr<2>(k) + powr<2>(mu) - powr<2>(Pi) * powr<2>(T) -
                   2 * k * mu * powr<1>(sqrt(1 + mf2)))) /
                     16. +
                 (powr<3>(k) * powr<-1>(1 + mf2) * powr<-1>(T) *
                  powr<-1>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 - mb2 + 3 * mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (-mb2 + mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mf2)) +
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mf2))) *
                  powr<2>(Sech((powr<-1>(T) * (mu + k * powr<1>(sqrt(1 + mf2)))) / 2.)) *
                  ((-mb2 + mf2) * powr<2>(k) + powr<2>(mu) - powr<2>(Pi) * powr<2>(T) +
                   2 * k * mu * powr<1>(sqrt(1 + mf2)))) /
                     16. +
                 (powr<2>(k) *
                  powr<-2>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 - mb2 + 3 * mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (mb2 - mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mf2)) -
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mf2))) *
                  powr<-3>(sqrt(1 + mf2)) *
                  ((-2 + mb2 - 3 * mf2) * powr<6>(k) * powr<2>(mb2 - mf2) +
                   powr<4>(k) * (-((8 + 40 * mf2 - 6 * mb2 * (4 + 5 * mf2) + 3 * powr<2>(mb2) + 35 * powr<2>(mf2)) *
                                   powr<2>(mu)) +
                                 (8 + 8 * mf2 + 2 * mb2 * (4 + mf2) + 3 * powr<2>(mb2) + 3 * powr<2>(mf2)) *
                                     powr<2>(Pi) * powr<2>(T)) +
                   powr<2>(k) * ((-22 + 3 * mb2 - 25 * mf2) * powr<4>(mu) -
                                 2 * (-2 + mb2 - 3 * mf2) * powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                                 (10 + 3 * mb2 + 7 * mf2) * powr<4>(Pi) * powr<4>(T)) +
                   (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                   8 * mu * powr<5>(k) * (-mb2 + mf2 - 3 * mb2 * mf2 + powr<2>(mb2) + 2 * powr<2>(mf2)) *
                       powr<1>(sqrt(1 + mf2)) -
                   8 * mu * powr<3>(k) *
                       ((-3 + 2 * mb2 - 5 * mf2) * powr<2>(mu) + (1 + mf2) * powr<2>(Pi) * powr<2>(T)) *
                       powr<1>(sqrt(1 + mf2)) +
                   8 * k * mu * (powr<4>(mu) - powr<4>(Pi) * powr<4>(T)) * powr<1>(sqrt(1 + mf2))) *
                  Tanh((powr<-1>(T) * (-mu + k * powr<1>(sqrt(1 + mf2)))) / 2.)) /
                     8. +
                 (powr<2>(k) *
                  powr<-2>(powr<4>(k) * powr<2>(mb2 - mf2) +
                           2 * powr<2>(k) *
                               ((2 - mb2 + 3 * mf2) * powr<2>(mu) + (2 + mb2 + mf2) * powr<2>(Pi) * powr<2>(T)) +
                           powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) +
                           4 * (-mb2 + mf2) * mu * powr<3>(k) * powr<1>(sqrt(1 + mf2)) +
                           4 * k * mu * (powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<1>(sqrt(1 + mf2))) *
                  powr<-3>(sqrt(1 + mf2)) *
                  ((-2 + mb2 - 3 * mf2) * powr<6>(k) * powr<2>(mb2 - mf2) +
                   powr<4>(k) * (-((8 + 40 * mf2 - 6 * mb2 * (4 + 5 * mf2) + 3 * powr<2>(mb2) + 35 * powr<2>(mf2)) *
                                   powr<2>(mu)) +
                                 (8 + 8 * mf2 + 2 * mb2 * (4 + mf2) + 3 * powr<2>(mb2) + 3 * powr<2>(mf2)) *
                                     powr<2>(Pi) * powr<2>(T)) +
                   powr<2>(k) * ((-22 + 3 * mb2 - 25 * mf2) * powr<4>(mu) -
                                 2 * (-2 + mb2 - 3 * mf2) * powr<2>(mu) * powr<2>(Pi) * powr<2>(T) +
                                 (10 + 3 * mb2 + 7 * mf2) * powr<4>(Pi) * powr<4>(T)) +
                   (-powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) * powr<2>(powr<2>(mu) + powr<2>(Pi) * powr<2>(T)) -
                   8 * mu * powr<5>(k) * (-mb2 + mf2 - 3 * mb2 * mf2 + powr<2>(mb2) + 2 * powr<2>(mf2)) *
                       powr<1>(sqrt(1 + mf2)) +
                   8 * mu * powr<3>(k) *
                       ((-3 + 2 * mb2 - 5 * mf2) * powr<2>(mu) + (1 + mf2) * powr<2>(Pi) * powr<2>(T)) *
                       powr<1>(sqrt(1 + mf2)) -
                   8 * k * mu * (powr<4>(mu) - powr<4>(Pi) * powr<4>(T)) * powr<1>(sqrt(1 + mf2))) *
                  TanhFiniteT(mu + k * powr<1>(sqrt(1 + mf2)), T)) /
                     8.;
        else
          throw std::runtime_error("Threshold Function FBFermiPiT is not implemented for given indices");
      }

      /**
       * @brief Bosonic threshold function with anomalous dimension.
       *
       * \f[
       *   l^{B}_{n} = \frac{2}{d-1} \Big(1 - \frac{\eta_b}{d+1}\Big) \max(n, 1)\, B_{n+1}(m^2)\,,
       * \f]
       * with \f$n\f$ = `lev` and \f$B\f$ from B().
       *
       * @param mb2 dimensionless boson mass \f$m^2/k^2\f$
       * @param etab boson anomalous dimension \f$\eta_b\f$
       * @param k RG scale
       * @param T temperature
       * @param d spacetime dimension
       */
      template <unsigned lev, typename NT1, typename NT2>
      auto lB(const NT1 &mb2, const NT2 &etab, const double &k, const double &T, const double &d)
      {
        const auto pref = 2. / (d - 1.) * (1. - etab / (d + 1.)) * double(lev + (lev == 0));
        return pref * B<lev + 1>(mb2, k, T);
      }

      /**
       * @brief lB() at \f$\eta_b = 0\f$ (local potential approximation).
       *
       * @param mb2 dimensionless boson mass \f$m^2/k^2\f$
       * @param k RG scale
       * @param T temperature
       * @param d spacetime dimension
       */
      template <unsigned lev, typename NT> auto lB(const NT &mb2, const double &k, const double &T, const double &d)
      {
        return lB<lev>(mb2, 0., k, T, d);
      }

      /**
       * @brief Fermionic threshold function with anomalous dimension.
       *
       * \f[
       *   l^{F}_{n} = \frac{2}{d-1} \Big(1 - \frac{\eta_f}{d}\Big) \max(n, 1)\, F_{n+1}(m^2)\,,
       * \f]
       * with \f$n\f$ = `lev` and \f$F\f$ from F().
       *
       * @param mf2 dimensionless fermion mass \f$m^2/k^2\f$
       * @param etaf fermion anomalous dimension \f$\eta_f\f$
       * @param k RG scale
       * @param T temperature
       * @param mu chemical potential
       * @param d spacetime dimension
       */
      template <unsigned lev, typename NT1, typename NT2>
      auto lF(const NT1 &mf2, const NT2 &etaf, const double &k, const double &T, const double &mu, const double &d)
      {
        const auto pref = 2. / (d - 1.) * (1. - etaf / d) * double(lev + (lev == 0));
        return pref * F<lev + 1>(mf2, k, T, mu);
      }

      /**
       * @brief lF() at \f$\eta_f = 0\f$ (local potential approximation).
       *
       * @param mf2 dimensionless fermion mass \f$m^2/k^2\f$
       * @param k RG scale
       * @param T temperature
       * @param mu chemical potential
       * @param d spacetime dimension
       */
      template <unsigned lev, typename NT>
      auto lF(const NT &mf2, const double &k, const double &T, const double &mu, const double &d)
      {
        return lF<lev>(mf2, 0., k, T, mu, d);
      }

      /**
       * @brief Mixed fermion-boson threshold function at external fermion momentum \f$p = (\pi T, \vec 0)\f$.
       *
       * \f[
       *   L^{FB}_{1,1} = \frac{2}{d-1} \Big[ \Big(1 - \frac{\eta_f}{d}\Big) FB_{2,1}
       *   + \Big(1 - \frac{\eta_b}{d+1}\Big) FB_{1,2} \Big]\,,
       * \f]
       * with \f$FB_{n_f n_b}\f$ from FBFermiPiT(). Only `levf` = `levb` = 1 is implemented. The result type `NT` must
       * be given explicitly.
       *
       * @param mf2 dimensionless fermion mass \f$m_f^2/k^2\f$
       * @param mb2 dimensionless boson mass \f$m_b^2/k^2\f$
       * @param etaf fermion anomalous dimension \f$\eta_f\f$
       * @param etab boson anomalous dimension \f$\eta_b\f$
       * @param k RG scale
       * @param T temperature
       * @param mu chemical potential
       * @param d spacetime dimension
       */
      template <unsigned levf, unsigned levb, typename NT, typename NT1, typename NT2, typename NT3, typename NT4>
      NT LFB(const NT1 &mf2, const NT2 &mb2, const NT3 &etaf, const NT4 &etab, const double &k, const double &T,
             const double &mu, const double &d)
      {
        const auto pref = 2. / (d - 1.);
        const auto preff = (1. - etaf / d);
        const auto prefb = (1. - etab / (d + 1));
        if constexpr (levb == 1 && levf == 1)
          return pref *
                 (preff * FBFermiPiT<2, 1, NT>(mf2, mb2, k, T, mu) + prefb * FBFermiPiT<1, 2, NT>(mf2, mb2, k, T, mu));
        else
          throw std::runtime_error("Threshold function not implemented.");
      }

      /**
       * @brief LFB() at \f$\eta_f = \eta_b = 0\f$ (local potential approximation).
       *
       * @param mf2 dimensionless fermion mass \f$m_f^2/k^2\f$
       * @param mb2 dimensionless boson mass \f$m_b^2/k^2\f$
       * @param k RG scale
       * @param T temperature
       * @param mu chemical potential
       * @param d spacetime dimension
       */
      template <unsigned levf, unsigned levb, typename NT, typename NT1, typename NT2>
      auto LFB(const NT1 &mf2, const NT2 &mb2, const double &k, const double &T, const double &mu, const double &d)
      {
        return LFB<levf, levb, NT>(mf2, mb2, 0., 0., k, T, mu, d);
      }
    } // namespace TFLitimSpatial
  } // namespace fRG
} // namespace DiFfRG