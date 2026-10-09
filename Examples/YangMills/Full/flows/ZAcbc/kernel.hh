#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZAcbc_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_INLINE_FUNCTION auto
    kernel(const double &l1, const double &cos1, const double &cos2, const double &S0, const double &S1,
           const double &SPhi, const double &k,
           const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA3,
           const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZAcbc,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA4SP,
           const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA4tadpole,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZc,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &Zc,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZA,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const double p1 = sqrt(powr<2>(S0) * (1. - S1 * sin(SPhi)));
      const double p2 =
          0.7071067811865475 * sqrt(powr<2>(S0) * (2. + 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi)));
      const double cosp1p2 = 0.7071067811865475 * powr<2>(S0) *
                             (-1. - 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi)) *
                             sqrt(powr<-1>(-powr<4>(S0) * (-1. + S1 * sin(SPhi)) *
                                           (2. + 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi))));
      const double cosl1p1 = cos1;
      const double cosl1p2 = 1.414213562373095 * powr<-1>(l1) *
                             sqrt(powr<-1>(powr<2>(S0) * (2. + 1.732050807568877 * S1 * cos(SPhi) + S1 * sin(SPhi)))) *
                             (-0.5 * cos1 * l1 * sqrt(powr<2>(S0) * (1. - S1 * sin(SPhi))) -
                              0.866025403784439 * l1 * sqrt(powr<2>(S0) * (1. + S1 * sin(SPhi))) *
                                  (1.414213562373095 * cos2 *
                                       sqrt((-1. + powr<2>(cos1)) * (-1. + powr<2>(S1)) *
                                            powr<-1>(2. - powr<2>(S1) + powr<2>(S1) * cos(2. * SPhi))) +
                                   cos1 * powr<2>(S0) * S1 * cos(SPhi) *
                                       sqrt(powr<-1>(powr<4>(S0) * (1. - powr<2>(S1) * powr<2>(sin(SPhi)))))));
      const auto _interp1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = RB(powr<2>(k), fma(2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1)));
      const auto _interp8 = ZA(sqrt(fma(2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1))));
      const auto _interp10 =
          ZA3(0.816496580927726 * sqrt(fma(cosl1p1, l1 * p1, powr<2>(l1) + powr<2>(p1))),
              sqrt(fma(3.,
                       (-1. + powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) *
                           powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)),
                       1.)),
              atan2(fma(3., l1 * (l1 + cosl1p1 * p1) * powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1)), -1.),
                    -1.732050807568877 * p1 * fma(2., cosl1p1 * l1, p1) *
                        powr<-1>(fma(cosl1p1, l1 * p1, powr<2>(l1) + powr<2>(p1)))));
      const auto _interp12 =
          ZAcbc(0.816496580927726 * sqrt(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))),
                sqrt(fma(-2., cosl1p2 * powr<3>(l1) * p2,
                         fma(-1. + 4. * powr<2>(cosl1p2), powr<2>(l1) * powr<2>(p2),
                             fma(-2., cosl1p2 * l1 * powr<3>(p2), powr<4>(l1) + powr<4>(p2)))) *
                     powr<-2>(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)))),
                atan2(fma(-3., powr<2>(l1) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)), 2.),
                      -1.732050807568877 * l1 * fma(-2., cosl1p2 * p2, l1) *
                          powr<-1>(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)))));
      const auto _interp15 = RB(powr<2>(k), fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)));
      const auto _interp16 = Zc(k);
      const auto _interp17 = Zc(sqrt(fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))));
      const auto _interp21 = ZAcbc(
          0.816496580927726 *
              sqrt(fma(cosl1p1, l1 * p1,
                       fma(cosl1p2, l1 * p2, fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))),
          0.5 * sqrt(fma(3., powr<2>(l1) * powr<2>(l1 + 2. * cosl1p1 * p1 + 2. * cosl1p2 * p2),
                         powr<2>(powr<2>(l1) - 2. * (cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                                     p2 * (cosl1p2 * l1 + p2)))) *
                     powr<-2>(fma(
                         cosl1p1, l1 * p1,
                         fma(cosl1p2, l1 * p2, fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))))),
          atan2(fma(-3.,
                    powr<2>(l1) * powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                           2. * cosp1p2 * p1 * p2 + powr<2>(p2)),
                    2.),
                1.732050807568877 * l1 * fma(2., cosl1p1 * p1, fma(2., cosl1p2 * p2, l1)) *
                    powr<-1>(fma(cosl1p1, l1 * p1,
                                 fma(cosl1p2, l1 * p2,
                                     fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))))));
      const auto _interp22 = dtZc(k);
      const auto _interp23 =
          RB(powr<2>(k),
             fma(2., cosl1p1 * l1 * p1,
                 fma(2., cosl1p2 * l1 * p2, fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      const auto _interp24 =
          RBdot(powr<2>(k),
                fma(2., cosl1p1 * l1 * p1,
                    fma(2., cosl1p2 * l1 * p2, fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      const auto _interp25 = Zc(1.02 * k);
      const auto _interp26 = Zc(
          sqrt(fma(2., cosl1p1 * l1 * p1,
                   fma(2., cosl1p2 * l1 * p2, fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))));
      const auto _interp28 =
          ZAcbc(0.816496580927726 * sqrt(fma(cosl1p2, l1 * p2, powr<2>(l1) + powr<2>(p2))),
                sqrt(fma(3.,
                         (-1. + powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) *
                             powr<-2>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
                         1.)),
                atan2(fma(-3., powr<2>(l1) * powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)), 2.),
                      -1.732050807568877 * l1 * fma(2., cosl1p2 * p2, l1) *
                          powr<-1>(fma(cosl1p2, l1 * p2, powr<2>(l1) + powr<2>(p2)))));
      const auto _interp30 = ZAcbc(
          0.816496580927726 *
              sqrt(fma(cosl1p1, l1 * p1,
                       fma(2., cosl1p2 * l1 * p2, fma(cosp1p2, p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))),
          0.5 * sqrt(fma(3., powr<2>(p1) * powr<2>(2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2),
                         powr<2>(2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 - powr<2>(p1) + 4. * cosl1p2 * l1 * p2 +
                                 2. * cosp1p2 * p1 * p2 + 2. * powr<2>(p2))) *
                     powr<-2>(fma(
                         cosl1p1, l1 * p1,
                         fma(2., cosl1p2 * l1 * p2, fma(cosp1p2, p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))))),
          atan2(fma(-3.,
                    powr<2>(p1) * powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 +
                                           cosp1p2 * p1 * p2 + powr<2>(p2)),
                    2.),
                -1.732050807568877 * p1 * fma(2., cosl1p1 * l1, fma(2., cosp1p2 * p2, p1)) *
                    powr<-1>(fma(cosl1p1, l1 * p1,
                                 fma(2., cosl1p2 * l1 * p2,
                                     fma(cosp1p2, p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))))));
      const auto _interp31 = RB(powr<2>(k), fma(2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)));
      const auto _interp32 = Zc(sqrt(fma(2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))));
      const auto _interp39 = ZAcbc(
          0.816496580927726 *
              sqrt(fma(2., cosp1p2 * p1 * p2,
                       fma(-1., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))),
          0.5 * sqrt(powr<-2>(
                         fma(2., cosp1p2 * p1 * p2,
                             fma(-1., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))) *
                     fma(3., powr<2>(l1) * powr<2>(l1 - 2. * (cosl1p1 * p1 + cosl1p2 * p2)),
                         powr<2>(powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) -
                                 2. * (powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2))))),
          atan2(fma(-3.,
                    powr<2>(l1) * powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                           l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                    2.),
                1.732050807568877 * l1 * fma(-2., cosl1p1 * p1 + cosl1p2 * p2, l1) *
                    powr<-1>(
                        fma(2., cosp1p2 * p1 * p2,
                            fma(-1., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))));
      const auto _interp42 =
          RB(powr<2>(k), fma(2., cosp1p2 * p1 * p2,
                             fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2))));
      const auto _interp43 =
          Zc(sqrt(fma(2., cosp1p2 * p1 * p2,
                      fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      const auto _den1 = powr<-1>(-1. + powr<2>(cosp1p2));
      const auto _den2 = powr<-1>(1. + powr<6>(k));
      const auto _den3 = powr<-2>(fma(_interp2, _interp4, fma(_interp6, powr<2>(l1), 0.)));
      const auto _den4 = powr<-1>(fma(_interp2, _interp4, fma(_interp6, powr<2>(l1), 0.)));
      const auto _den6 = powr<-1>(fma(2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1)));
      const auto _den8 =
          powr<-1>(fma(_interp4, _interp7, fma(_interp8, powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1), 0.)));
      const auto _den9 =
          powr<-1>(fma(_interp15, _interp16, fma(_interp17, powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2), 0.)));
      const auto _den12 = powr<-1>(fma(_interp16, _interp42,
                                       fma(_interp43,
                                           powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                                               2. * cosp1p2 * p1 * p2 + powr<2>(p2),
                                           0.)));
      const auto _den13 = powr<-2>(fma(_interp16, _interp23,
                                       fma(_interp26,
                                           powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 +
                                               2. * cosp1p2 * p1 * p2 + powr<2>(p2),
                                           0.)));
      const auto _interp45 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 +
                                                            powr<2>(p2) - l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
                                   0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                                       l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)) *
                                              (3. * powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                                               powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                                       4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)))),
                                   atan2(2. - 3. * powr<2>(p1) *
                                                  powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                                                           2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
                                         -1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                                             powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                                      l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2))));
      const auto _interp14 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 +
                                         cosp1p2 * p1 * p2 + powr<2>(p2)),
                0.5 * sqrt(powr<-2>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 +
                                    cosp1p2 * p1 * p2 + powr<2>(p2)) *
                           (powr<2>(powr<2>(l1) + 4. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 -
                                    2. * p2 * (cosp1p2 * p1 + p2)) +
                            3. * powr<2>(powr<2>(l1) - 2. * cosl1p2 * l1 * p2 - p1 * (p1 + 2. * cosp1p2 * p2)))),
                atan2(2. - 3. * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                               powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 +
                                        cosp1p2 * p1 * p2 + powr<2>(p2)),
                      1.732050807568877 *
                          powr<-1>(powr<2>(l1) + cosl1p1 * l1 * p1 + powr<2>(p1) - cosl1p2 * l1 * p2 +
                                   cosp1p2 * p1 * p2 + powr<2>(p2)) *
                          (powr<2>(l1) - 2. * cosl1p2 * l1 * p2 - p1 * (p1 + 2. * cosp1p2 * p2))));
      const auto _interp33 = RBdot(powr<2>(k), powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2));
      const auto _interp19 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                         cosp1p2 * p1 * p2 + powr<2>(p2)),
                0.5 * sqrt(powr<-2>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                    cosp1p2 * p1 * p2 + powr<2>(p2)) *
                           (powr<2>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 -
                                    2. * p2 * (cosp1p2 * p1 + p2)) +
                            3. * powr<2>(powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                                         p1 * (p1 + 2. * cosp1p2 * p2)))),
                atan2(2. - 3. * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                               powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                        cosp1p2 * p1 * p2 + powr<2>(p2)),
                      -1.732050807568877 *
                          powr<-1>(powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                   cosp1p2 * p1 * p2 + powr<2>(p2)) *
                          (powr<2>(l1) + 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) + p1 * (p1 + 2. * cosp1p2 * p2))));
      const auto _interp34 = RB(powr<2>(k), powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1));
      const auto _interp35 = ZA(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)));
      const auto _interp37 =
          ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                   (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 +
                    (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) - 2. * cosl1p1 * l1 * powr<3>(p1) +
                    powr<4>(p1))),
              atan2(-1. + 3. * l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                    1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) *
                        powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1))));
      const auto _interp41 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                         l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
                0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                    l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                           (powr<2>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 -
                                    2. * p2 * (cosp1p2 * p1 + p2)) +
                            3. * powr<2>(powr<2>(l1) - 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) +
                                         p1 * (p1 + 2. * cosp1p2 * p2)))),
                atan2(2. - 3. * (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)) *
                               powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                        l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
                      -1.732050807568877 *
                          powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                   l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                          (powr<2>(l1) - 2. * l1 * (cosl1p1 * p1 + cosl1p2 * p2) + p1 * (p1 + 2. * cosp1p2 * p2))));
      // clang-format off
      using _T = decltype(_den1 + _den12 + _den13 + _den2 + _den3 + _den4 + _den6 + _den8 + _den9 + _interp1 + _interp10 + _interp12 + _interp14 + _interp15 + _interp16 + _interp17 + _interp19 + _interp2 + _interp21 + _interp22 + _interp23 + _interp24 + _interp25 + _interp26 + _interp28 + _interp3 + _interp30 + _interp31 + _interp32 + _interp33 + _interp34 + _interp35 + _interp37 + _interp39 + _interp4 + _interp41 + _interp42 + _interp43 + _interp45 + _interp5 + _interp6 + _interp7 + _interp8 + cosl1p1 + cosl1p2 + cosp1p2 + k + l1 + p1 + p2);
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _den11 =
            powr<-1>(_interp16 * _interp31 + _interp32 * (powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)));
        const auto _cse1_k1 = -1. + powr<2>(cosp1p2);
        const auto _cse2_k1 = _cse1_k1 * p2;
        _acc +=
            fma(-1.5,
                _den1 * _den11 * _den13 * _den4 * _interp21 *
                    (_interp22 * _interp23 + _interp16 * _interp24 + _interp23 * (-50. * _interp16 + 50. * _interp25)) *
                    _interp28 * _interp30 * (_cse2_k1 - cosl1p2 * l1 + cosl1p1 * cosp1p2 * l1) *
                    (-cosl1p1 * cosl1p2 * p1 + cosp1p2 * p1 + p2 - powr<2>(cosl1p2) * p2),
                fma(1.5,
                    _den1 * _den12 * _den2 * _den3 * _den9 * _interp12 * _interp39 * _interp45 *
                        (_interp2 * (-50. * _interp4 + 50. * _interp5) * powr<6>(k) +
                         _interp3 * _interp4 * (1. + powr<6>(k)) + _interp1 * _interp2 * (1. + 1. * powr<6>(k))) *
                        (_cse2_k1 + cosl1p2 * l1 - cosl1p1 * cosp1p2 * l1) *
                        (cosl1p1 * cosl1p2 * p1 - cosp1p2 * p1 + (-1. + powr<2>(cosl1p2)) * p2),
                    0.));
      }
      { // subkernel 2
        const auto _den10 =
            powr<-2>(_interp16 * _interp31 + _interp32 * (powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)));
        const auto _den14 = powr<-1>(_interp16 * _interp23 +
                                     _interp26 * (powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                                  2. * cosl1p2 * l1 * p2 + 2. * cosp1p2 * p1 * p2 + powr<2>(p2)));
        const auto _cse1_k2 = -powr<2>(cosp1p2);
        const auto _cse2_k2 = powr<2>(l1) + powr<2>(p1);
        const auto _cse3_k2 = -p1;
        const auto _cse4_k2 = powr<2>(cosp1p2) * powr<2>(p1) * p2;
        const auto _cse5_k2 = -cosp1p2 * p2;
        const auto _cse6_k2 = _cse3_k2 + _cse5_k2;
        const auto _cse7_k2 = cosp1p2 * p2;
        const auto _cse8_k2 = _cse3_k2 + _cse7_k2;
        const auto _cse9_k2 = -1. + powr<2>(cosp1p2);
        // clang-format off
        _acc += fma(-1.5, _den1 * _den10 * _den14 * _den4 * _interp21 * _interp28 * _interp30 * (_interp22 * _interp31 + (-50. * _interp16 + 50. * _interp25) * _interp31 + _interp16 * _interp33) * (-cosl1p2 * l1 + cosl1p1 * cosp1p2 * l1 + _cse9_k2 * p2) * (-cosl1p1 * cosl1p2 * p1 + cosp1p2 * p1 + p2 - powr<2>(cosl1p2) * p2), fma(3., _den1 * _den2 * _den3 * _den6 * _den8 * _den9 * _interp10 * _interp12 * _interp14 * (_interp2 * (-50. * _interp4 + 50. * _interp5) * powr<6>(k) + _interp3 * _interp4 * (1. + powr<6>(k)) + _interp1 * _interp2 * (1. + 1. * powr<6>(k))) * (_cse6_k2 * powr<2>(cosl1p2) * powr<2>(l1) * p1 + powr<3>(cosl1p1) * cosl1p2 * cosp1p2 * powr<2>(l1) * powr<2>(p1) - powr<3>(cosl1p2) * powr<3>(l1) * p2 + cosl1p2 * l1 * (_cse4_k2 + _cse2_k2 * cosp1p2 * p1 + powr<2>(l1) * p2) + powr<2>(p1) * ((1. + _cse1_k2) * powr<2>(l1) + _cse9_k2 * cosp1p2 * p1 * p2) + powr<2>(cosl1p1) * l1 * p1 * (_cse2_k2 * cosl1p2 * cosp1p2 + _cse8_k2 * powr<2>(cosl1p2) * l1 + (-1. + _cse1_k2) * l1 * p1 - 2. * cosp1p2 * l1 * p2 + 2. * cosl1p2 * p1 * p2 - cosl1p2 * powr<2>(cosp1p2) * p1 * p2) + cosl1p1 * (powr<2>(cosl1p2) * (_cse8_k2 * powr<3>(l1) + _cse6_k2 * l1 * powr<2>(p1)) - powr<3>(cosl1p2) * powr<2>(l1) * p1 * p2 + cosp1p2 * l1 * (_cse4_k2 - cosp1p2 * powr<2>(l1) * p1 - cosp1p2 * powr<3>(p1) - powr<2>(l1) * p2 - 2. * powr<2>(p1) * p2) + cosl1p2 * p1 * (3. * cosp1p2 * powr<2>(l1) * p1 + powr<2>(cosp1p2) * (powr<2>(l1) - powr<2>(p1)) * p2 + (2. * powr<2>(l1) + powr<2>(p1)) * p2))), 0.));
        // clang-format on
      }
      { // subkernel 3
        const auto _den5 = powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1));
        const auto _den7 =
            powr<-1>(_interp34 * _interp4 + _interp35 * (powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1)));
        const auto _cse1_k3 = -powr<2>(cosp1p2);
        const auto _cse2_k3 = cosp1p2 * p2;
        const auto _cse3_k3 = -powr<2>(p1);
        const auto _cse4_k3 = _cse3_k3 + powr<2>(l1);
        const auto _cse5_k3 = -p1;
        const auto _cse6_k3 = _cse2_k3 + _cse5_k3;
        const auto _cse7_k3 = 2. + _cse1_k3;
        const auto _cse8_k3 = _cse7_k3 * p1;
        const auto _cse9_k3 = -powr<3>(cosl1p2) * powr<3>(l1) * p2;
        const auto _cse10_k3 = powr<2>(l1) + powr<2>(p1);
        const auto _cse11_k3 = _cse10_k3 * cosp1p2 * p1;
        const auto _cse12_k3 = powr<2>(l1) * p2;
        const auto _cse13_k3 = powr<2>(cosp1p2) * powr<2>(p1) * p2;
        const auto _cse14_k3 = _cse11_k3 + _cse12_k3 + _cse13_k3;
        const auto _cse15_k3 = _cse14_k3 * cosl1p2 * l1;
        const auto _cse16_k3 = 1. + _cse1_k3;
        const auto _cse17_k3 = -1. + powr<2>(cosp1p2);
        const auto _cse18_k3 = _cse4_k3 * cosp1p2;
        const auto _cse19_k3 = 2. * p1 * p2;
        const auto _cse20_k3 = -powr<2>(cosp1p2) * p1 * p2;
        const auto _cse21_k3 = _cse18_k3 + _cse19_k3 + _cse20_k3;
        const auto _cse22_k3 = _cse21_k3 * cosl1p2 * l1;
        // clang-format off
        _acc += fma(3., _den1 * _den12 * _den2 * _den3 * _den5 * _den7 * _interp37 * _interp39 * _interp41 * (_interp2 * (-50. * _interp4 + 50. * _interp5) * powr<6>(k) + _interp3 * _interp4 * (1. + powr<6>(k)) + _interp1 * _interp2 * (1. + 1. * powr<6>(k))) * (_cse15_k3 + _cse9_k3 + powr<3>(cosl1p1) * l1 * (_cse8_k3 - cosl1p2 * cosp1p2 * l1) * powr<2>(p1) + powr<2>(cosl1p2) * cosp1p2 * powr<2>(l1) * p1 * p2 + powr<2>(cosl1p1) * p1 * (_cse22_k3 + _cse17_k3 * powr<3>(p1) + powr<2>(cosl1p2) * powr<2>(l1) * (p1 - cosp1p2 * p2) + powr<2>(l1) * (_cse5_k3 + 2. * powr<2>(cosp1p2) * p1 + 2. * cosp1p2 * p2)) + powr<2>(p1) * (_cse16_k3 * powr<2>(l1) + p1 * (_cse2_k3 + p1 - powr<2>(cosp1p2) * p1 - powr<3>(cosp1p2) * p2)) + cosl1p1 * (powr<3>(cosl1p2) * powr<2>(l1) * p1 * p2 + powr<2>(cosl1p2) * (_cse6_k3 * powr<3>(l1) - cosp1p2 * l1 * powr<2>(p1) * p2) + cosl1p2 * p1 * (-cosp1p2 * powr<2>(l1) * p1 - 2. * powr<2>(l1) * p2 - powr<2>(p1) * p2 + powr<2>(cosp1p2) * (-powr<2>(l1) + powr<2>(p1)) * p2) + l1 * (-2. * powr<3>(p1) + powr<2>(cosp1p2) * (-powr<2>(l1) * p1 + powr<3>(p1)) + powr<3>(cosp1p2) * powr<2>(p1) * p2 + cosp1p2 * (-powr<2>(l1) * p2 - 2. * powr<2>(p1) * p2)))), fma(-3., _den1 * _den13 * _den4 * _den6 * _den8 * _interp10 * _interp19 * _interp21 * (_interp22 * _interp23 + _interp16 * _interp24 + _interp23 * (-50. * _interp16 + 50. * _interp25)) * (_cse15_k3 + _cse9_k3 + powr<3>(cosl1p1) * l1 * (_cse8_k3 + cosl1p2 * cosp1p2 * l1) * powr<2>(p1) + _cse17_k3 * powr<2>(p1) * (powr<2>(l1) + p1 * (_cse2_k3 + p1)) - powr<2>(cosl1p2) * cosp1p2 * powr<2>(l1) * p1 * p2 + powr<2>(cosl1p1) * p1 * (_cse22_k3 + _cse6_k3 * powr<2>(cosl1p2) * powr<2>(l1) + _cse16_k3 * powr<3>(p1) + powr<2>(l1) * (p1 - 2. * powr<2>(cosp1p2) * p1 - 2. * cosp1p2 * p2)) + cosl1p1 * (-powr<3>(cosl1p2) * powr<2>(l1) * p1 * p2 + powr<2>(cosl1p2) * (_cse6_k3 * powr<3>(l1) - cosp1p2 * l1 * powr<2>(p1) * p2) + cosl1p2 * p1 * (cosp1p2 * powr<2>(l1) * p1 + _cse4_k3 * powr<2>(cosp1p2) * p2 + (2. * powr<2>(l1) + powr<2>(p1)) * p2) + l1 * (-2. * powr<3>(p1) + powr<2>(cosp1p2) * (-powr<2>(l1) * p1 + powr<3>(p1)) + powr<3>(cosp1p2) * powr<2>(p1) * p2 + cosp1p2 * (-powr<2>(l1) * p2 - 2. * powr<2>(p1) * p2)))), 0.));
        // clang-format on
      }
      return _acc;
    }

    static KOKKOS_INLINE_FUNCTION auto
    constant(const double &S0, const double &S1, const double &SPhi, const double &k,
             const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA3,
             const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZAcbc,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA4SP,
             const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA4tadpole,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZc,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &Zc,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZA,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      return 0.;
    }

  private:
    static KOKKOS_INLINE_FUNCTION auto RB(const auto &k2, const auto &p2) { return Regulator::RB(k2, p2); }

    static KOKKOS_INLINE_FUNCTION auto RF(const auto &k2, const auto &p2) { return Regulator::RF(k2, p2); }

    static KOKKOS_INLINE_FUNCTION auto RBdot(const auto &k2, const auto &p2) { return Regulator::RBdot(k2, p2); }

    static KOKKOS_INLINE_FUNCTION auto RFdot(const auto &k2, const auto &p2) { return Regulator::RFdot(k2, p2); }

    static KOKKOS_INLINE_FUNCTION auto dq2RB(const auto &k2, const auto &p2) { return Regulator::dq2RB(k2, p2); }

    static KOKKOS_INLINE_FUNCTION auto dq2RF(const auto &k2, const auto &p2) { return Regulator::dq2RF(k2, p2); }
  };
} // namespace DiFfRG
using DiFfRG::ZAcbc_kernel;