#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZAcbc_kernel
  {
  public:
    using Regulator = _Regulator; // clang-format off


static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const double &cos1, const double &cos2, const double &S0, const double &S1,
                                               const double &SPhi, const double &k,
                                               const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA3,
                                               const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZAcbc,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4SP,
                                               const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA4tadpole,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
                                               const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
{
  using namespace DiFfRG;
                                  // clang-format on

      using namespace DiFfRG::compute;
      const double p1 = sqrt(powr<2>(S0) * (1. - S1 * sin(SPhi)));
      const double p2 = 0.7071067811865475 * sqrt(powr<2>(S0) * (2. + 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi)));
      const double cosp1p2 = 0.7071067811865475 * powr<2>(S0) * (-1. - 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi)) *
        sqrt(powr<-1>(-powr<4>(S0) * (-1. + S1 * sin(SPhi)) * (2. + 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi))));
      const double cosl1p1 = cos1;
      const double cosl1p2 = 1.414213562373095 * powr<-1>(l1) *
        sqrt(powr<-1>(powr<2>(S0) * (2. + 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi)))) *
        (-0.5 * cos1 * l1 * sqrt(powr<2>(S0) * (1. - S1 * sin(SPhi))) -
         0.866025403784439 * l1 * sqrt(powr<2>(S0) * (1. + S1 * sin(SPhi))) *
           (1.414213562373095 * cos2 *
              sqrt((-1. + powr<2>(cos1)) * (-1. + powr<2>(S1)) * powr<-1>(2. - powr<2>(S1) + powr<2>(S1) * cos(2. * SPhi))) +
            cos1 * powr<2>(S0) * S1 * cos(SPhi) * sqrt(powr<-1>(powr<4>(S0) * (1. - powr<2>(S1) * powr<2>(sin(SPhi)))))));
      // clang-format off
using _T = decltype(3. * powr<-1>(-1. + powr<2>(cosp1p2)) * powr<-1>(1. + powr<6>(k)) *
        powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
        (powr<3>(cosl1p1) * cosl1p2 * cosp1p2 * powr<2>(l1) * powr<2>(p1) -
         powr<3>(cosl1p2) * powr<3>(l1) * p2 +
         powr<2>(cosl1p2) * powr<2>(l1) * p1 * (-p1 - cosp1p2 * p2) +
         powr<2>(p1) * ((1. - powr<2>(cosp1p2)) * powr<2>(l1) +
                        cosp1p2 * (-1. + powr<2>(cosp1p2)) * p1 * p2) +
         cosl1p2 * l1 *
             (cosp1p2 * p1 * (powr<2>(l1) + powr<2>(p1)) + powr<2>(l1) * p2 +
              powr<2>(cosp1p2) * powr<2>(p1) * p2) +
         powr<2>(cosl1p1) * l1 * p1 *
             ((-1. - powr<2>(cosp1p2)) * l1 * p1 + cosl1p2 * cosp1p2 * (powr<2>(l1) + powr<2>(p1)) -
              2. * cosp1p2 * l1 * p2 + 2. * cosl1p2 * p1 * p2 -
              cosl1p2 * powr<2>(cosp1p2) * p1 * p2 + powr<2>(cosl1p2) * l1 * (-p1 + cosp1p2 * p2)) +
         cosl1p1 * (-powr<3>(cosl1p2) * powr<2>(l1) * p1 * p2 +
                    cosp1p2 * l1 *
                        (-cosp1p2 * powr<2>(l1) * p1 - cosp1p2 * powr<3>(p1) - powr<2>(l1) * p2 -
                         2. * powr<2>(p1) * p2 + powr<2>(cosp1p2) * powr<2>(p1) * p2) +
                    cosl1p2 * p1 *
                        (3. * cosp1p2 * powr<2>(l1) * p1 +
                         powr<2>(cosp1p2) * (powr<2>(l1) - powr<2>(p1)) * p2 +
                         (2. * powr<2>(l1) + powr<2>(p1)) * p2) +
                    powr<2>(cosl1p2) * (l1 * powr<2>(p1) * (-p1 - cosp1p2 * p2) +
                                        powr<3>(l1) * (-p1 + cosp1p2 * p2)))) *
        ((1. + 1. * powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
             RB(powr<2>(k), powr<2>(l1)) +
         (1. + powr<6>(k)) * RBdot(powr<2>(k), powr<2>(l1)) *
             ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
         powr<6>(k) * RB(powr<2>(k), powr<2>(l1)) *
             (-50. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
              50. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     ZA(sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)),
            sqrt(1. + 3. * (-1. + powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) *
                          powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1))),
            atan2(-1.732050807568877 * p1 * (2. * cosl1p1 * l1 + p1) *
                      powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)),
                  -1. + 3. * l1 * (l1 + cosl1p1 * p1) *
                            powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                   (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 +
                    (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) -
                    2. * cosl1p2 * l1 * powr<3>(p2) + powr<4>(p2))),
              atan2(-1.732050807568877 * l1 * (l1 - 2. * cosl1p2 * p2) *
                        powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                    2. - 3. * powr<2>(l1) *
                             powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) -
                                       cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) -
                                  cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)) *
                         (powr<2>(powr<2>(l1) + 4. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  2. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)) +
                          3. * powr<2>(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 -
                                       p1 * (p1 + 2. * cosp1p2 * p2)))),
              atan2(1.732050807568877 * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                        powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 +
                                 cosp1p2 * p1 * p2 + powr<2>(p2)) *
                        sqrt(powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                             (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) -
                              4. * cosl1p2 * l1 * p2 + 4. * cosp1p2 * p1 * p2 + 4. * powr<2>(p2))) *
                        sqrt(powr<-1>((powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                                      (powr<2>(l1) + powr<2>(p1) + 4. * cosp1p2 * p1 * p2 +
                                       4. * powr<2>(p2) -
                                       2. * l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)))) *
                        (powr<2>(l1) - 2. * cosl1p2 * l1 * p2 - p1 * (p1 + 2. * cosp1p2 * p2)),
                    2. - 3. * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                             powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) -
                                      cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) * Zc(k) +
                 (powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)))) -
    3. * powr<-1>(-1. + powr<2>(cosp1p2)) *
        powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
        (powr<3>(cosl1p1) * l1 * powr<2>(p1) *
             (cosl1p2 * cosp1p2 * l1 + (2. - powr<2>(cosp1p2)) * p1) -
         powr<3>(cosl1p2) * powr<3>(l1) * p2 - powr<2>(cosl1p2) * cosp1p2 * powr<2>(l1) * p1 * p2 +
         cosl1p2 * l1 *
             (cosp1p2 * p1 * (powr<2>(l1) + powr<2>(p1)) + powr<2>(l1) * p2 +
              powr<2>(cosp1p2) * powr<2>(p1) * p2) +
         (-1. + powr<2>(cosp1p2)) * powr<2>(p1) * (powr<2>(l1) + p1 * (p1 + cosp1p2 * p2)) +
         powr<2>(cosl1p1) * p1 *
             ((1. - powr<2>(cosp1p2)) * powr<3>(p1) +
              powr<2>(l1) * (p1 - 2. * powr<2>(cosp1p2) * p1 - 2. * cosp1p2 * p2) +
              powr<2>(cosl1p2) * powr<2>(l1) * (-p1 + cosp1p2 * p2) +
              cosl1p2 * l1 *
                  (cosp1p2 * (powr<2>(l1) - powr<2>(p1)) + 2. * p1 * p2 -
                   powr<2>(cosp1p2) * p1 * p2)) +
         cosl1p1 * (-powr<3>(cosl1p2) * powr<2>(l1) * p1 * p2 +
                    cosl1p2 * p1 *
                        (cosp1p2 * powr<2>(l1) * p1 +
                         powr<2>(cosp1p2) * (powr<2>(l1) - powr<2>(p1)) * p2 +
                         (2. * powr<2>(l1) + powr<2>(p1)) * p2) +
                    powr<2>(cosl1p2) *
                        (-cosp1p2 * l1 * powr<2>(p1) * p2 + powr<3>(l1) * (-p1 + cosp1p2 * p2)) +
                    l1 * (-2. * powr<3>(p1) + powr<2>(cosp1p2) * (-powr<2>(l1) * p1 + powr<3>(p1)) +
                          powr<3>(cosp1p2) * powr<2>(p1) * p2 +
                          cosp1p2 * (-powr<2>(l1) * p2 - 2. * powr<2>(p1) * p2)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     ZA(sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)),
            sqrt(1. + 3. * (-1. + powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) *
                          powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1))),
            atan2(-1.732050807568877 * p1 * (2. * cosl1p1 * l1 + p1) *
                      powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)),
                  -1. + 3. * l1 * (l1 + cosl1p1 * p1) *
                            powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)) *
                         (powr<2>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) -
                                  2. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)) +
                          3. * powr<2>(powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                                       p1 * (p1 + 2. * cosp1p2 * p2)))),
              atan2(-1.732050807568877 * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                        powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                 cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)) *
                        sqrt(powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                             (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                              4. * cosl1p2 * l1 * p2 + 4. * cosp1p2 * p1 * p2 + 4. * powr<2>(p2))) *
                        sqrt(powr<-1>((powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                                      (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       4. * cosl1p2 * l1 * p2 + 4. * cosp1p2 * p1 * p2 +
                                       4. * powr<2>(p2)))) *
                        (powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                         p1 * (p1 + 2. * cosp1p2 * p2)),
                    2. - 3. * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                             powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                      cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                         (3. * powr<2>(l1) * powr<2>(l1 + 2. * cosl1p1 * p1 + 2. * cosl1p2 * p2) +
                          powr<2>(powr<2>(l1) -
                                  2. * (cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                        p2 * (cosl1p2 * l1 + p2))))),
              atan2(1.732050807568877 * l1 * (l1 + 2. * cosl1p1 * p1 + 2. * cosl1p2 * p2) *
                        powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                 2. * cosp1p2 * p1 * p2 + powr<2>(p2)),
                    2. - 3. * powr<2>(l1) *
                             powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                      cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        (dtZc(k) *
             RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
         RBdot(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                               2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
             Zc(k) +
         RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                            2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
             (-50. * Zc(k) + 50. * Zc(1.02 * k))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                    2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     Zc(k) +
                 (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 +
                  2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                             2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))) -
    1.5 * powr<-1>(-1. + powr<2>(cosp1p2)) *
        (-cosl1p1 * cosl1p2 * p1 + cosp1p2 * p1 + p2 - powr<2>(cosl1p2) * p2) *
        (-cosl1p2 * l1 + cosl1p1 * cosp1p2 * l1 + (-1. + powr<2>(cosp1p2)) * p2) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
              sqrt(1. + 3. * (-1. + powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) *
                            powr<-2>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2))),
              atan2(-1.732050807568877 * l1 * (l1 + 2. * cosl1p2 * p2) *
                        powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
                    2. - 3. * powr<2>(l1) *
                             powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)) *
                         (3. * powr<2>(p1) * powr<2>(2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                          powr<2>(2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 - powr<2>(p1) +
                                  4. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 +
                                  2. * powr<2>(p2)))),
              atan2(-1.732050807568877 * p1 * (2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                        powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                 2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
                    2. - 3. * powr<2>(p1) *
                             powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                      2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                         (3. * powr<2>(l1) * powr<2>(l1 + 2. * cosl1p1 * p1 + 2. * cosl1p2 * p2) +
                          powr<2>(powr<2>(l1) -
                                  2. * (cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                        p2 * (cosl1p2 * l1 + p2))))),
              atan2(1.732050807568877 * l1 * (l1 + 2. * cosl1p1 * p1 + 2. * cosl1p2 * p2) *
                        powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                 2. * cosp1p2 * p1 * p2 + powr<2>(p2)),
                    2. - 3. * powr<2>(l1) *
                             powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                      cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        (dtZc(k) *
             RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
         RBdot(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                               2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
             Zc(k) +
         RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                            2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
             (-50. * Zc(k) + 50. * Zc(1.02 * k))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) * Zc(k) +
                 (powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                    2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     Zc(k) +
                 (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 +
                  2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                             2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))) -
    1.5 * powr<-1>(-1. + powr<2>(cosp1p2)) *
        (-cosl1p1 * cosl1p2 * p1 + cosp1p2 * p1 + p2 - powr<2>(cosl1p2) * p2) *
        (-cosl1p2 * l1 + cosl1p1 * cosp1p2 * l1 + (-1. + powr<2>(cosp1p2)) * p2) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
              sqrt(1. + 3. * (-1. + powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) *
                            powr<-2>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2))),
              atan2(-1.732050807568877 * l1 * (l1 + 2. * cosl1p2 * p2) *
                        powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
                    2. - 3. * powr<2>(l1) *
                             powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)) *
                         (3. * powr<2>(p1) * powr<2>(2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                          powr<2>(2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 - powr<2>(p1) +
                                  4. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 +
                                  2. * powr<2>(p2)))),
              atan2(-1.732050807568877 * p1 * (2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                        powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                 2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
                    2. - 3. * powr<2>(p1) *
                             powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                      2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                         (3. * powr<2>(l1) * powr<2>(l1 + 2. * cosl1p1 * p1 + 2. * cosl1p2 * p2) +
                          powr<2>(powr<2>(l1) -
                                  2. * (cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                        p2 * (cosl1p2 * l1 + p2))))),
              atan2(1.732050807568877 * l1 * (l1 + 2. * cosl1p1 * p1 + 2. * cosl1p2 * p2) *
                        powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                 2. * cosp1p2 * p1 * p2 + powr<2>(p2)),
                    2. - 3. * powr<2>(l1) *
                             powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                      cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        (dtZc(k) * RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) +
         RBdot(powr<2>(k), powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) * Zc(k) +
         RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
             (-50. * Zc(k) + 50. * Zc(1.02 * k))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) * Zc(k) +
                 (powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                    2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     Zc(k) +
                 (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 +
                  2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                             2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))) +
    3. * powr<-1>(-1. + powr<2>(cosp1p2)) * powr<-1>(1. + powr<6>(k)) *
        powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
        (powr<3>(cosl1p1) * l1 * powr<2>(p1) *
             (-cosl1p2 * cosp1p2 * l1 + (2. - powr<2>(cosp1p2)) * p1) -
         powr<3>(cosl1p2) * powr<3>(l1) * p2 + powr<2>(cosl1p2) * cosp1p2 * powr<2>(l1) * p1 * p2 +
         cosl1p2 * l1 *
             (cosp1p2 * p1 * (powr<2>(l1) + powr<2>(p1)) + powr<2>(l1) * p2 +
              powr<2>(cosp1p2) * powr<2>(p1) * p2) +
         powr<2>(p1) * ((1. - powr<2>(cosp1p2)) * powr<2>(l1) +
                        p1 * (p1 - powr<2>(cosp1p2) * p1 + cosp1p2 * p2 - powr<3>(cosp1p2) * p2)) +
         powr<2>(cosl1p1) * p1 *
             ((-1. + powr<2>(cosp1p2)) * powr<3>(p1) +
              powr<2>(cosl1p2) * powr<2>(l1) * (p1 - cosp1p2 * p2) +
              powr<2>(l1) * (-p1 + 2. * powr<2>(cosp1p2) * p1 + 2. * cosp1p2 * p2) +
              cosl1p2 * l1 *
                  (cosp1p2 * (powr<2>(l1) - powr<2>(p1)) + 2. * p1 * p2 -
                   powr<2>(cosp1p2) * p1 * p2)) +
         cosl1p1 * (powr<3>(cosl1p2) * powr<2>(l1) * p1 * p2 +
                    cosl1p2 * p1 *
                        (-cosp1p2 * powr<2>(l1) * p1 - 2. * powr<2>(l1) * p2 - powr<2>(p1) * p2 +
                         powr<2>(cosp1p2) * (-powr<2>(l1) + powr<2>(p1)) * p2) +
                    powr<2>(cosl1p2) *
                        (-cosp1p2 * l1 * powr<2>(p1) * p2 + powr<3>(l1) * (-p1 + cosp1p2 * p2)) +
                    l1 * (-2. * powr<3>(p1) + powr<2>(cosp1p2) * (-powr<2>(l1) * p1 + powr<3>(p1)) +
                          powr<3>(cosp1p2) * powr<2>(p1) * p2 +
                          cosp1p2 * (-powr<2>(l1) * p2 - 2. * powr<2>(p1) * p2)))) *
        ((1. + 1. * powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
             RB(powr<2>(k), powr<2>(l1)) +
         (1. + powr<6>(k)) * RBdot(powr<2>(k), powr<2>(l1)) *
             ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
         powr<6>(k) * RB(powr<2>(k), powr<2>(l1)) *
             (-50. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
              50. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     ZA(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
            sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                 (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 +
                  (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) -
                  2. * cosl1p1 * l1 * powr<3>(p1) + powr<4>(p1))),
            atan2(1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) *
                      powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                  -1. + 3. * l1 * (l1 - cosl1p1 * p1) *
                            powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                       powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                  l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                         (3. * powr<2>(l1) * powr<2>(l1 - 2. * (cosl1p1 * p1 + cosl1p2 * p2)) +
                          powr<2>(powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) -
                                  2. * (powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2))))),
              atan2(1.732050807568877 * l1 * (l1 - 2. * (cosl1p1 * p1 + cosl1p2 * p2)) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                    2. - 3. * powr<2>(l1) *
                             powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                      powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                       l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                  l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                         (powr<2>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  2. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)) +
                          3. * powr<2>(powr<2>(l1) - 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                                       p1 * (p1 + 2. * cosp1p2 * p2)))),
              atan2(-1.732050807568877 * (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                        sqrt(powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                             (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) -
                              4. * cosl1p2 * l1 * p2 + 4. * cosp1p2 * p1 * p2 + 4. * powr<2>(p2))) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                        sqrt(powr<-1>((powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                                      (powr<2>(l1) + powr<2>(p1) + 4. * cosp1p2 * p1 * p2 +
                                       4. * powr<2>(p2) -
                                       2. * l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)))) *
                        (powr<2>(l1) - 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                         p1 * (p1 + 2. * cosp1p2 * p2)),
                    2. - 3. * (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                             powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                      l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                    2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                     Zc(k) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                  2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                             2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)))) +
    1.5 * powr<-1>(-1. + powr<2>(cosp1p2)) * powr<-1>(1. + powr<6>(k)) *
        (cosl1p1 * cosl1p2 * p1 - cosp1p2 * p1 + (-1. + powr<2>(cosl1p2)) * p2) *
        (cosl1p2 * l1 - cosl1p1 * cosp1p2 * l1 + (-1. + powr<2>(cosp1p2)) * p2) *
        ((1. + 1. * powr<6>(k)) * dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) *
             RB(powr<2>(k), powr<2>(l1)) +
         (1. + powr<6>(k)) * RBdot(powr<2>(k), powr<2>(l1)) *
             ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
         powr<6>(k) * RB(powr<2>(k), powr<2>(l1)) *
             (-50. * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
              50. * ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667)))) *
        powr<-2>(RB(powr<2>(k), powr<2>(l1)) * ZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
                 powr<2>(l1) * ZA(l1)) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                   (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 +
                    (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) -
                    2. * cosl1p2 * l1 * powr<3>(p2) + powr<4>(p2))),
              atan2(-1.732050807568877 * l1 * (l1 - 2. * cosl1p2 * p2) *
                        powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                    2. - 3. * powr<2>(l1) *
                             powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                       powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                  l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                         (3. * powr<2>(l1) * powr<2>(l1 - 2. * (cosl1p1 * p1 + cosl1p2 * p2)) +
                          powr<2>(powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) -
                                  2. * (powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2))))),
              atan2(1.732050807568877 * l1 * (l1 - 2. * (cosl1p1 * p1 + cosl1p2 * p2)) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                    2. - 3. * powr<2>(l1) *
                             powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                      powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)))) *
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                       l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                  l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)) *
                         (3. * powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                          powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                  4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)))),
              atan2(-1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
                    2. - 3. * powr<2>(p1) *
                             powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                                      2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) * Zc(k) +
                 (powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2)))) *
        powr<-1>(RB(powr<2>(k), powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                    2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                     Zc(k) +
                 (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                  2. * cosp1p2 * p1 * p2 + powr<2>(p2)) *
                     Zc(sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                             2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2)))));
      // clang-format on
      const auto _interp1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = RB(powr<2>(k), fma(2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1)));
      const auto _interp8 = ZA(sqrt(fma(2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1))));
      const auto _interp10 = ZA3(
        0.816496580927726 * sqrt(fma(cosl1p1, l1 * p1, powr<2>(l1) + powr<2>(p1))),
        sqrt(fma(3., (-1. + powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) * powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)), 1.)),
        atan2(-1.732050807568877 * p1 * fma(2., cosl1p1 * l1, p1) * powr<-1>(fma(cosl1p1, l1 * p1, powr<2>(l1) + powr<2>(p1))),
              fma(3., l1 * (l1 + cosl1p1 * p1) * powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)), -1.)));
      const auto _interp12 =
        ZAcbc(0.816496580927726 * sqrt(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))),
              sqrt(fma(-2., cosl1p2 * powr<3>(l1) * p2,
                       fma(-1. + 4. * powr<2>(cosl1p2), powr<2>(l1) * powr<2>(p2),
                           fma(-2., cosl1p2 * l1 * powr<3>(p2), powr<4>(l1) + powr<4>(p2)))) *
                   powr<-2>(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)))),
              atan2(-1.732050807568877 * l1 * fma(-2., cosl1p2 * p2, l1) * powr<-1>(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))),
                    fma(-3., powr<2>(l1) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)), 2.)));
      const auto _interp15 = RB(powr<2>(k), fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)));
      const auto _interp16 = Zc(k);
      const auto _interp17 = Zc(sqrt(fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)))); // clang-format off

const auto _interp21 = ZAcbc(0.816496580927726 *
          sqrt(fma(cosl1p1, l1 *p1,
                   fma(cosl1p2, l1 *p2,
                       fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))),
      0.5 * sqrt(fma(3., powr<2>(l1) * powr<2>(l1 + 2. * cosl1p1 * p1 + 2. * cosl1p2 * p2),
                     powr<2>(powr<2>(l1) -
                             2. * (cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                   p2 * (cosl1p2 * l1 + p2)))) *
                 powr<-2>(fma(cosl1p1, l1 *p1,
                              fma(cosl1p2, l1 *p2,
                                  fma(2., cosp1p2 * p1 * p2,
                                      powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))))),
      atan2(1.732050807568877 * l1 * fma(2., cosl1p1 *p1, fma(2., cosl1p2 *p2, l1)) *
                powr<-1>(fma(cosl1p1, l1 *p1,
                             fma(cosl1p2, l1 *p2,
                                 fma(2., cosp1p2 * p1 * p2,
                                     powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))),
            fma(-3.,
                powr<2>(l1) * powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)),
                2.)));
      // clang-format on

      const auto _interp22 = dtZc(k);
      const auto _interp23 =
        RB(powr<2>(k),
           fma(2., cosl1p1 * l1 * p1, fma(2., cosl1p2 * l1 * p2, fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      const auto _interp24 =
        RBdot(powr<2>(k),
              fma(2., cosl1p1 * l1 * p1, fma(2., cosl1p2 * l1 * p2, fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      const auto _interp25 = Zc(1.02 * k);
      const auto _interp26 = Zc(
        sqrt(fma(2., cosl1p1 * l1 * p1, fma(2., cosl1p2 * l1 * p2, fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))));
      const auto _interp28 = ZAcbc(
        0.816496580927726 * sqrt(fma(cosl1p2, l1 * p2, powr<2>(l1) + powr<2>(p2))),
        sqrt(fma(3., (-1. + powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) * powr<-2>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)), 1.)),
        atan2(-1.732050807568877 * l1 * fma(2., cosl1p2 * p2, l1) * powr<-1>(fma(cosl1p2, l1 * p2, powr<2>(l1) + powr<2>(p2))),
              fma(-3., powr<2>(l1) * powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)), 2.))); // clang-format off

const auto _interp30 = ZAcbc(0.816496580927726 *
          sqrt(fma(cosl1p1, l1 *p1,
                   fma(2., cosl1p2 * l1 * p2,
                       fma(cosp1p2, p1 *p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))),
      0.5 * sqrt(fma(3., powr<2>(p1) * powr<2>(2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2),
                     powr<2>(2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 - powr<2>(p1) +
                             4. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + 2. * powr<2>(p2))) *
                 powr<-2>(fma(cosl1p1, l1 *p1,
                              fma(2., cosl1p2 * l1 * p2,
                                  fma(cosp1p2, p1 *p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))))),
      atan2(-1.732050807568877 * p1 * fma(2., cosl1p1 *l1, fma(2., cosp1p2 *p2, p1)) *
                powr<-1>(fma(cosl1p1, l1 *p1,
                             fma(2., cosl1p2 * l1 * p2,
                                 fma(cosp1p2, p1 *p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))),
            fma(-3.,
                powr<2>(p1) * powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) +
                                       2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
                2.)));
      // clang-format on

      const auto _interp31 = RB(powr<2>(k), fma(2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)));
      const auto _interp32 = Zc(sqrt(fma(2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)))); // clang-format off

const auto _interp39 = ZAcbc(0.816496580927726 * sqrt(fma(2., cosp1p2 * p1 * p2,
                                   fma(-1., l1 *(cosl1p1 *p1 + cosl1p2 * p2),
                                       powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))),
      0.5 * sqrt(powr<-2>(fma(2., cosp1p2 * p1 * p2,
                              fma(-1., l1 *(cosl1p1 *p1 + cosl1p2 * p2),
                                  powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))) *
                 fma(3., powr<2>(l1) * powr<2>(l1 - 2. * (cosl1p1 * p1 + cosl1p2 * p2)),
                     powr<2>(powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) -
                             2. * (powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2))))),
      atan2(1.732050807568877 * l1 * fma(-2., cosl1p1 *p1 + cosl1p2 * p2, l1) *
                powr<-1>(fma(2., cosp1p2 * p1 * p2,
                             fma(-1., l1 *(cosl1p1 *p1 + cosl1p2 * p2),
                                 powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))),
            fma(-3.,
                powr<2>(l1) * powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                       powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                2.)));
      // clang-format on

      const auto _interp42 =
        RB(powr<2>(k), fma(2., cosp1p2 * p1 * p2, fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2))));
      const auto _interp43 =
        Zc(sqrt(fma(2., cosp1p2 * p1 * p2, fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      _T _acc{}; // clang-format off

{ // subkernel 1
const auto _interp45 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
      0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                          l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)) *
                 (3. * powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                  powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                          4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)))),
      atan2(-1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                         l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
            2. - 3. * powr<2>(p1) *
                     powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                              2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2))));
        // clang-format on

        const auto _cse1 = powr<2>(cosp1p2);
        const auto _cse2 = -1. + _cse1;
        const auto _cse3 = powr<2>(l1);
        const auto _cse4 = -2. * cosl1p2 * l1 * p2;
        const auto _cse5 = powr<2>(p2);
        const auto _cse6 = powr<6>(k);
        const auto _cse7 = powr<-1>(_cse2);
        const auto _cse8 = powr<2>(cosl1p2);
        const auto _cse9 = _cse2 * p2;
        const auto _cse10 = powr<2>(p1);
        const auto _cse11 = 2. * cosl1p2 * l1 * p2;
        const auto _cse12 = 2. * cosp1p2 * p1 * p2;
        const auto _cse13 = _interp2 * _interp4;
        const auto _cse14 = _cse3 * _interp6;
        const auto _cse15 = _cse13 + _cse14; // clang-format off
_acc += fma(1.5,
    powr<-2>(_cse15) * powr<-1>(1. + _cse6) * _cse7 * _interp12 *
        powr<-1>(_interp15 * _interp16 + (_cse3 + _cse4 + _cse5) * _interp17) * _interp39 *
        _interp45 *
        ((1. + 1. * _cse6) * _interp1 * _interp2 + (1. + _cse6) * _interp3 * _interp4 +
         _cse6 * _interp2 * (-50. * _interp4 + 50. * _interp5)) *
        (_cse9 + cosl1p2 * l1 - cosl1p1 * cosp1p2 * l1) *
        powr<-1>(_interp16 * _interp42 +
                 _interp43 * (_cse10 + _cse12 + _cse3 + _cse4 + _cse5 - 2. * cosl1p1 * l1 * p1)) *
        (cosl1p1 * cosl1p2 * p1 - cosp1p2 * p1 + (-1. + _cse8) * p2),
    fma(-1.5,
        powr<-1>(_cse15) * _cse7 * _interp21 *
            (_interp22 * _interp23 + _interp16 * _interp24 +
             _interp23 * (-50. * _interp16 + 50. * _interp25)) *
            _interp28 * _interp30 *
            powr<-1>(_interp16 * _interp31 + (_cse11 + _cse3 + _cse5) * _interp32) *
            (_cse9 - cosl1p2 * l1 + cosl1p1 * cosp1p2 * l1) *
            powr<-2>(_interp16 * _interp23 + _interp26 * (_cse10 + _cse11 + _cse12 + _cse3 + _cse5 +
                                                          2. * cosl1p1 * l1 * p1)) *
            (-cosl1p1 * cosl1p2 * p1 + cosp1p2 * p1 + p2 - _cse8 * p2),
        0.));
        // clang-format on
        // clang-format off

}
{ // subkernel 2
const auto _interp14 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 +
                               cosp1p2 * p1 * p2 + powr<2>(p2)),
      0.5 * sqrt(powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 +
                          cosp1p2 * p1 * p2 + powr<2>(p2)) *
                 (powr<2>(powr<2>(l1) + 4. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                          2. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)) +
                  3. * powr<2>(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 -
                               p1 * (p1 + 2. * cosp1p2 * p2)))),
      atan2(1.732050807568877 * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 +
                         cosp1p2 * p1 * p2 + powr<2>(p2)) *
                sqrt(powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 4. * cosl1p2 * l1 * p2 +
                      4. * cosp1p2 * p1 * p2 + 4. * powr<2>(p2))) *
                sqrt(powr<-1>((powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                              (powr<2>(l1) + powr<2>(p1) + 4. * cosp1p2 * p1 * p2 +
                               4. * powr<2>(p2) - 2. * l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)))) *
                (powr<2>(l1) - 2. * cosl1p2 * l1 * p2 - p1 * (p1 + 2. * cosp1p2 * p2)),
            2. - 3. * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 +
                              cosp1p2 * p1 * p2 + powr<2>(p2))));
        // clang-format on

        const auto _interp33 = RBdot(powr<2>(k), powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2));
        const auto _cse1 = powr<2>(cosp1p2);
        const auto _cse2 = -1. + _cse1;
        const auto _cse3 = powr<2>(l1);
        const auto _cse4 = 2. * cosl1p2 * l1 * p2;
        const auto _cse5 = powr<2>(p2);
        const auto _cse6 = powr<-1>(_cse2);
        const auto _cse7 = 2. * cosl1p1 * l1 * p1;
        const auto _cse8 = powr<2>(p1);
        const auto _cse9 = powr<2>(cosl1p2);
        const auto _cse10 = -_cse1;
        const auto _cse11 = _cse3 + _cse8;
        const auto _cse12 = -p1;
        const auto _cse13 = powr<3>(cosl1p2);
        const auto _cse14 = _cse1 * _cse8 * p2;
        const auto _cse15 = -cosp1p2 * p2;
        const auto _cse16 = _cse12 + _cse15;
        const auto _cse17 = powr<3>(l1);
        const auto _cse18 = cosp1p2 * p2;
        const auto _cse19 = _cse12 + _cse18;
        const auto _cse20 = _cse3 + _cse7 + _cse8;
        const auto _cse21 = _cse3 * _interp6;
        const auto _cse22 = _interp2 * _interp4;
        const auto _cse23 = _cse21 + _cse22;
        const auto _cse24 = powr<6>(k); // clang-format off
_acc += fma(-1.5,
    powr<-1>(_cse23) * _cse6 * _interp21 * _interp28 * _interp30 *
        powr<-2>(_interp16 * _interp31 + (_cse3 + _cse4 + _cse5) * _interp32) *
        (_interp22 * _interp31 + (-50. * _interp16 + 50. * _interp25) * _interp31 +
         _interp16 * _interp33) *
        (-cosl1p2 * l1 + cosl1p1 * cosp1p2 * l1 + _cse2 * p2) *
        (-cosl1p1 * cosl1p2 * p1 + cosp1p2 * p1 + p2 - _cse9 * p2) *
        powr<-1>(_interp16 * _interp23 +
                 _interp26 * (_cse3 + _cse4 + _cse5 + _cse7 + _cse8 + 2. * cosp1p2 * p1 * p2)),
    fma(3.,
        powr<-1>(_cse20) * powr<-2>(_cse23) * powr<-1>(1. + _cse24) * _cse6 * _interp10 *
            _interp12 * _interp14 *
            ((1. + 1. * _cse24) * _interp1 * _interp2 + (1. + _cse24) * _interp3 * _interp4 +
             _cse24 * _interp2 * (-50. * _interp4 + 50. * _interp5)) *
            powr<-1>(_interp4 * _interp7 + _cse20 * _interp8) *
            powr<-1>(_interp15 * _interp16 + _interp17 * (_cse3 + _cse5 - 2. * cosl1p2 * l1 * p2)) *
            (_cse3 * _cse8 * powr<3>(cosl1p1) * cosl1p2 * cosp1p2 + _cse16 * _cse3 * _cse9 * p1 -
             _cse13 * _cse17 * p2 + cosl1p2 * l1 * (_cse14 + _cse11 * cosp1p2 * p1 + _cse3 * p2) +
             powr<2>(cosl1p1) * l1 * p1 *
                 (_cse11 * cosl1p2 * cosp1p2 + _cse19 * _cse9 * l1 + (-1. + _cse10) * l1 * p1 -
                  2. * cosp1p2 * l1 * p2 + 2. * cosl1p2 * p1 * p2 - _cse1 * cosl1p2 * p1 * p2) +
             _cse8 * ((1. + _cse10) * _cse3 + _cse2 * cosp1p2 * p1 * p2) +
             cosl1p1 * (_cse9 * (_cse17 * _cse19 + _cse16 * _cse8 * l1) - _cse13 * _cse3 * p1 * p2 +
                        cosp1p2 * l1 *
                            (_cse14 - _cse3 * cosp1p2 * p1 - cosp1p2 * powr<3>(p1) - _cse3 * p2 -
                             2. * _cse8 * p2) +
                        cosl1p2 * p1 *
                            (3. * _cse3 * cosp1p2 * p1 + _cse1 * (_cse3 - _cse8) * p2 +
                             (2. * _cse3 + _cse8) * p2))),
        0.));
        // clang-format on
        // clang-format off

}
{ // subkernel 3
const auto _interp19 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                               cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
      0.5 * sqrt(powr<-2>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                          cosp1p2 * p1 * p2 + powr<2>(p2)) *
                 (powr<2>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) -
                          2. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)) +
                  3. * powr<2>(powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                               p1 * (p1 + 2. * cosp1p2 * p2)))),
      atan2(-1.732050807568877 * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                         cosp1p2 * p1 * p2 + powr<2>(p2)) *
                sqrt(powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 4. * cosl1p2 * l1 * p2 +
                      4. * cosp1p2 * p1 * p2 + 4. * powr<2>(p2))) *
                sqrt(powr<-1>((powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                              (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                               4. * cosl1p2 * l1 * p2 + 4. * cosp1p2 * p1 * p2 +
                               4. * powr<2>(p2)))) *
                (powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                 p1 * (p1 + 2. * cosp1p2 * p2)),
            2. - 3. * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                              cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2))));
        // clang-format on

        const auto _interp34 = RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1));
        const auto _interp35 = ZA(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)));
        const auto _interp37 =
          ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                   (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 + (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) -
                    2. * cosl1p1 * l1 * powr<3>(p1) + powr<4>(p1))),
              atan2(1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                    -1. + 3. * l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)))); // clang-format off

const auto _interp41 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                               l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
      0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                          l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                 (powr<2>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                          2. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)) +
                  3. * powr<2>(powr<2>(l1) - 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                               p1 * (p1 + 2. * cosp1p2 * p2)))),
      atan2(-1.732050807568877 * (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                sqrt(powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 4. * cosl1p2 * l1 * p2 +
                      4. * cosp1p2 * p1 * p2 + 4. * powr<2>(p2))) *
                powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                         l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                sqrt(powr<-1>((powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                              (powr<2>(l1) + powr<2>(p1) + 4. * cosp1p2 * p1 * p2 +
                               4. * powr<2>(p2) - 2. * l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)))) *
                (powr<2>(l1) - 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                 p1 * (p1 + 2. * cosp1p2 * p2)),
            2. - 3. * (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                              l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2))));
        // clang-format on

        const auto _cse1 = powr<2>(p1);
        const auto _cse2 = powr<2>(cosp1p2);
        const auto _cse3 = powr<2>(l1);
        const auto _cse4 = -_cse2;
        const auto _cse5 = -1. + _cse2;
        const auto _cse6 = powr<2>(cosl1p2);
        const auto _cse7 = powr<3>(cosl1p2);
        const auto _cse8 = powr<3>(l1);
        const auto _cse9 = -p1;
        const auto _cse10 = cosp1p2 * p2;
        const auto _cse11 = powr<3>(p1);
        const auto _cse12 = powr<3>(cosp1p2);
        const auto _cse13 = -2. * cosl1p1 * l1 * p1;
        const auto _cse14 = _cse1 + _cse13 + _cse3;
        const auto _cse15 = powr<6>(k);
        const auto _cse16 = powr<-1>(_cse5);
        const auto _cse17 = powr<3>(cosl1p1);
        const auto _cse18 = 2. + _cse4;
        const auto _cse19 = _cse18 * p1;
        const auto _cse20 = -_cse7 * _cse8 * p2;
        const auto _cse21 = _cse1 + _cse3;
        const auto _cse22 = _cse21 * cosp1p2 * p1;
        const auto _cse23 = _cse3 * p2;
        const auto _cse24 = _cse1 * _cse2 * p2;
        const auto _cse25 = _cse22 + _cse23 + _cse24;
        const auto _cse26 = _cse25 * cosl1p2 * l1;
        const auto _cse27 = powr<2>(cosl1p1);
        const auto _cse28 = 1. + _cse4;
        const auto _cse29 = _cse10 + _cse9;
        const auto _cse30 = -_cse1;
        const auto _cse31 = _cse3 + _cse30;
        const auto _cse32 = _cse31 * cosp1p2;
        const auto _cse33 = 2. * p1 * p2; // clang-format off
_acc += fma(3.,
    powr<-1>(_cse14) * powr<-1>(1. + _cse15) * _cse16 * _interp37 * _interp39 *
        powr<-1>(_cse14 * _interp35 + _interp34 * _interp4) * _interp41 *
        ((1. + 1. * _cse15) * _interp1 * _interp2 + (1. + _cse15) * _interp3 * _interp4 +
         _cse15 * _interp2 * (-50. * _interp4 + 50. * _interp5)) *
        powr<-2>(_interp2 * _interp4 + _cse3 * _interp6) *
        powr<-1>(_interp16 * _interp42 +
                 _interp43 * (_cse1 + _cse13 + _cse3 - 2. * cosl1p2 * l1 * p2 +
                              2. * cosp1p2 * p1 * p2 + powr<2>(p2))) *
        (_cse20 + _cse26 + _cse1 * _cse17 * l1 * (_cse19 - cosl1p2 * cosp1p2 * l1) +
         _cse3 * _cse6 * cosp1p2 * p1 * p2 +
         _cse1 * (_cse28 * _cse3 + p1 * (_cse10 + p1 - _cse2 * p1 - _cse12 * p2)) +
         _cse27 * p1 *
             (_cse11 * _cse5 + _cse3 * _cse6 * (p1 - cosp1p2 * p2) +
              _cse3 * (_cse9 + 2. * _cse2 * p1 + 2. * cosp1p2 * p2) +
              cosl1p2 * l1 * (_cse32 + _cse33 - _cse2 * p1 * p2)) +
         cosl1p1 * (_cse3 * _cse7 * p1 * p2 +
                    cosl1p2 * p1 *
                        (-_cse3 * cosp1p2 * p1 - _cse1 * p2 + _cse2 * (_cse1 - _cse3) * p2 -
                         2. * _cse3 * p2) +
                    _cse6 * (_cse29 * _cse8 - _cse1 * cosp1p2 * l1 * p2) +
                    l1 * (-2. * _cse11 + _cse2 * (_cse11 - _cse3 * p1) + _cse1 * _cse12 * p2 +
                          cosp1p2 * (-2. * _cse1 * p2 - _cse3 * p2)))),
    fma(-3.,
        _cse16 * _interp10 * _interp19 * _interp21 *
            (_interp22 * _interp23 + _interp16 * _interp24 +
             _interp23 * (-50. * _interp16 + 50. * _interp25)) *
            powr<-1>(_interp2 * _interp4 + _cse3 * _interp6) *
            powr<-1>(_cse1 + _cse3 + 2. * cosl1p1 * l1 * p1) *
            powr<-1>(_interp4 * _interp7 + _interp8 * (_cse1 + _cse3 + 2. * cosl1p1 * l1 * p1)) *
            powr<-2>(_interp16 * _interp23 +
                     _interp26 * (_cse1 + _cse3 + 2. * cosl1p1 * l1 * p1 + 2. * cosl1p2 * l1 * p2 +
                                  2. * cosp1p2 * p1 * p2 + powr<2>(p2))) *
            (_cse20 + _cse26 + _cse1 * _cse17 * l1 * (_cse19 + cosl1p2 * cosp1p2 * l1) +
             _cse1 * _cse5 * (_cse3 + p1 * (_cse10 + p1)) - _cse3 * _cse6 * cosp1p2 * p1 * p2 +
             _cse27 * p1 *
                 (_cse11 * _cse28 + _cse29 * _cse3 * _cse6 +
                  _cse3 * (p1 - 2. * _cse2 * p1 - 2. * cosp1p2 * p2) +
                  cosl1p2 * l1 * (_cse32 + _cse33 - _cse2 * p1 * p2)) +
             cosl1p1 *
                 (-_cse3 * _cse7 * p1 * p2 +
                  cosl1p2 * p1 *
                      (_cse3 * cosp1p2 * p1 + (_cse1 + 2. * _cse3) * p2 + _cse2 * _cse31 * p2) +
                  _cse6 * (_cse29 * _cse8 - _cse1 * cosp1p2 * l1 * p2) +
                  l1 * (-2. * _cse11 + _cse2 * (_cse11 - _cse3 * p1) + _cse1 * _cse12 * p2 +
                        cosp1p2 * (-2. * _cse1 * p2 - _cse3 * p2)))),
        0.));
        // clang-format on
      }
      return _acc; // clang-format off

}

static KOKKOS_FORCEINLINE_FUNCTION auto constant(const double &S0, const double &S1, const double &SPhi, const double &k,
                                                 const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA3,
                                                 const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZAcbc,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4SP,
                                                 const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA4tadpole,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
                                                 const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
{
  using namespace DiFfRG;
                   // clang-format on

      using namespace DiFfRG::compute;
      return 0.;
    }

  private:
    static KOKKOS_FORCEINLINE_FUNCTION auto RB(const auto &k2, const auto &p2) { return Regulator::RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RF(const auto &k2, const auto &p2) { return Regulator::RF(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RBdot(const auto &k2, const auto &p2) { return Regulator::RBdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto RFdot(const auto &k2, const auto &p2) { return Regulator::RFdot(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RB(const auto &k2, const auto &p2) { return Regulator::dq2RB(k2, p2); }

    static KOKKOS_FORCEINLINE_FUNCTION auto dq2RF(const auto &k2, const auto &p2) { return Regulator::dq2RF(k2, p2); }
  };
} // namespace DiFfRG
using DiFfRG::ZAcbc_kernel;