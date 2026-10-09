#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA3_kernel
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
      const auto _interp7 = RB(powr<2>(k), fma(-2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1)));
      const auto _interp8 = ZA(sqrt(fma(-2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1))));
      const auto _interp9 = RB(powr<2>(k), fma(2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)));
      const auto _interp17 =
          RB(powr<2>(k), fma(2., cosp1p2 * p1 * p2,
                             fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2))));
      const auto _interp18 =
          ZA(sqrt(fma(2., cosp1p2 * p1 * p2,
                      fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      const auto _interp20 =
          ZA3(0.816496580927726 * sqrt(fma(-1., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1))),
              sqrt(fma(-2., cosl1p1 * powr<3>(l1) * p1,
                       fma(-1. + 4. * powr<2>(cosl1p1), powr<2>(l1) * powr<2>(p1),
                           fma(-2., cosl1p1 * l1 * powr<3>(p1), powr<4>(l1) + powr<4>(p1)))) *
                   powr<-2>(fma(-1., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1)))),
              atan2(fma(3., l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)), -1.),
                    1.732050807568877 * p1 * fma(-2., cosl1p1 * l1, p1) *
                        powr<-1>(fma(-1., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1)))));
      const auto _interp22 = ZA3(
          0.816496580927726 *
              sqrt(fma(2., cosp1p2 * p1 * p2,
                       fma(-1., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))),
          0.5 * sqrt(powr<-2>(
                         fma(2., cosp1p2 * p1 * p2,
                             fma(-1., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))) *
                     fma(3.,
                         powr<2>(-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                 p2 * (-2. * cosl1p2 * l1 + p2)),
                         powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 +
                                 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))),
          atan2(fma(3.,
                    l1 * (l1 - cosl1p1 * p1 - cosl1p2 * p2) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                    -1.),
                -1.732050807568877 *
                    fma(-2., cosl1p1 * l1 * p1,
                        fma(2., cosp1p2 * p1 * p2, fma(p2, -2. * cosl1p2 * l1 + p2, powr<2>(p1)))) *
                    powr<-1>(
                        fma(2., cosp1p2 * p1 * p2,
                            fma(-1., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))));
      const auto _interp25 = RB(powr<2>(k), fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)));
      const auto _interp26 = ZA(sqrt(fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))));
      const auto _interp28 =
          ZA3(0.816496580927726 * sqrt(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))),
              sqrt(fma(-2., cosl1p2 * powr<3>(l1) * p2,
                       fma(-1. + 4. * powr<2>(cosl1p2), powr<2>(l1) * powr<2>(p2),
                           fma(-2., cosl1p2 * l1 * powr<3>(p2), powr<4>(l1) + powr<4>(p2)))) *
                   powr<-2>(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)))),
              atan2(fma(3., l1 * (l1 - cosl1p2 * p2) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)), -1.),
                    1.732050807568877 * p2 * fma(-2., cosl1p2 * l1, p2) *
                        powr<-1>(fma(-1., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)))));
      const auto _interp34 =
          ZAcbc(0.816496580927726 * sqrt(fma(-1., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1))),
                sqrt(fma(-2., cosl1p1 * powr<3>(l1) * p1,
                         fma(-1. + 4. * powr<2>(cosl1p1), powr<2>(l1) * powr<2>(p1),
                             fma(-2., cosl1p1 * l1 * powr<3>(p1), powr<4>(l1) + powr<4>(p1)))) *
                     powr<-2>(fma(-1., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1)))),
                atan2(fma(3., l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)), -1.),
                      1.732050807568877 * p1 * powr<-1>(fma(-1., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1))) *
                          fma(2., cosl1p1 * l1, fma(-1., p1, 0.))));
      const auto _interp37 = dtZc(k);
      const auto _interp38 = Zc(k);
      const auto _interp39 = Zc(1.02 * k);
      const auto _interp40 = Zc(l1);
      const auto _interp41 = Zc(sqrt(fma(-2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1))));
      const auto _interp42 = Zc(sqrt(fma(2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))));
      const auto _interp49 = ZAcbc(
          0.816496580927726 *
              sqrt(fma(2., cosp1p2 * p1 * p2,
                       fma(-1., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))),
          0.5 * sqrt(powr<-2>(
                         fma(2., cosp1p2 * p1 * p2,
                             fma(-1., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))) *
                     fma(3.,
                         powr<2>(-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                 p2 * (-2. * cosl1p2 * l1 + p2)),
                         powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 +
                                 2. * cosp1p2 * p1 * p2 + powr<2>(p2)))),
          atan2(fma(3.,
                    l1 * (l1 - cosl1p1 * p1 - cosl1p2 * p2) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                    -1.),
                1.732050807568877 *
                    fma(-2., cosl1p1 * l1 * p1,
                        fma(2., cosp1p2 * p1 * p2, fma(p2, -2. * cosl1p2 * l1 + p2, powr<2>(p1)))) *
                    powr<-1>(
                        fma(2., cosp1p2 * p1 * p2,
                            fma(-1., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2))))));
      const auto _interp52 =
          Zc(sqrt(fma(2., cosp1p2 * p1 * p2,
                      fma(-2., l1 * (cosl1p1 * p1 + cosl1p2 * p2), powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      const auto _interp57 = Zc(sqrt(fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2))));
      const auto _den1 = powr<-1>(-1. + powr<2>(cosp1p2));
      const auto _den2 = powr<-1>(1. + powr<6>(k));
      const auto _den3 = powr<-2>(fma(_interp2, _interp38, fma(_interp40, powr<2>(l1), 0.)));
      const auto _den4 = powr<-2>(fma(_interp2, _interp4, fma(_interp6, powr<2>(l1), 0.)));
      const auto _den5 = powr<-1>(fma(-2., cosl1p1 * l1 * p1, powr<2>(l1) + powr<2>(p1)));
      const auto _den6 = powr<-1>(fma(-2., cosl1p2 * l1 * p2, powr<2>(l1) + powr<2>(p2)));
      const auto _den8 =
          powr<-1>(fma(_interp38, _interp7, fma(_interp41, powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1), 0.)));
      const auto _den9 =
          powr<-1>(fma(_interp4, _interp7, fma(_interp8, powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1), 0.)));
      const auto _den10 =
          powr<-1>(fma(_interp25, _interp4, fma(_interp26, powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2), 0.)));
      const auto _den11 =
          powr<-1>(fma(_interp25, _interp38, fma(_interp57, powr<2>(l1) - 2. * cosl1p2 * l1 * p2 + powr<2>(p2), 0.)));
      const auto _den13 =
          powr<-1>(fma(_interp38, _interp9, fma(_interp42, powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2), 0.)));
      const auto _den14 = powr<-1>(
          fma(-2., cosl1p1 * l1 * p1,
              fma(-2., cosl1p2 * l1 * p2, fma(2., cosp1p2 * p1 * p2, powr<2>(l1) + powr<2>(p1) + powr<2>(p2)))));
      const auto _den15 = powr<-1>(fma(_interp17, _interp4,
                                       fma(_interp18,
                                           powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                                               2. * cosp1p2 * p1 * p2 + powr<2>(p2),
                                           0.)));
      const auto _den16 = powr<-1>(fma(_interp17, _interp38,
                                       fma(_interp52,
                                           powr<2>(l1) - 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                                               2. * cosp1p2 * p1 * p2 + powr<2>(p2),
                                           0.)));
      const auto _den17 = powr<-1>(fma(3., powr<4>(p1),
                                       fma(6., cosp1p2 * powr<3>(p1) * p2,
                                           fma(8. + powr<2>(cosp1p2), powr<2>(p1) * powr<2>(p2),
                                               fma(6., cosp1p2 * p1 * powr<3>(p2), fma(3., powr<4>(p2), 0.))))));
      const auto _interp32 = ZA4SP(
          0.7071067811865475 * sqrt(powr<2>(l1) + powr<2>(p1) - cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)));
      const auto _interp31 = ZA4SP(
          0.7071067811865475 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2)));
      const auto _interp35 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
                sqrt(1. + 3. * (-1. + powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) *
                              powr<-2>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2))),
                atan2(-1. + 3. * l1 * (l1 + cosl1p2 * p2) * powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
                      1.732050807568877 * p2 * (2. * cosl1p2 * l1 + p2) *
                          powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2))));
      const auto _interp36 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                         cosp1p2 * p1 * p2 + powr<2>(p2)),
                0.5 * sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                    cosp1p2 * p1 * p2 + powr<2>(p2)) *
                           (powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                                    4. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
                            3. * powr<2>(2. * cosl1p1 * l1 * p1 - powr<2>(p1) + p2 * (2. * cosl1p2 * l1 + p2)))),
                atan2(-1. + 3. * (powr<2>(l1) - cosl1p1 * l1 * p1 + cosl1p2 * l1 * p2 - cosp1p2 * p1 * p2) *
                                powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                         cosp1p2 * p1 * p2 + powr<2>(p2)),
                      1.732050807568877 *
                          powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                   cosp1p2 * p1 * p2 + powr<2>(p2)) *
                          (-2. * cosl1p1 * l1 * p1 + powr<2>(p1) - p2 * (2. * cosl1p2 * l1 + p2))));
      const auto _interp43 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                     (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 +
                      (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) - 2. * cosl1p1 * l1 * powr<3>(p1) +
                      powr<4>(p1))),
                atan2(-1. + 3. * l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                      1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1) *
                          powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1))));
      const auto _interp45 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
                sqrt(1. + 3. * (-1. + powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) *
                              powr<-2>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2))),
                atan2(-1. + 3. * l1 * (l1 + cosl1p2 * p2) * powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
                      -1.732050807568877 * p2 * (2. * cosl1p2 * l1 + p2) *
                          powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2))));
      const auto _interp47 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                         cosp1p2 * p1 * p2 + powr<2>(p2)),
                0.5 * sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                    cosp1p2 * p1 * p2 + powr<2>(p2)) *
                           (powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                                    4. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
                            3. * powr<2>(2. * cosl1p1 * l1 * p1 - powr<2>(p1) + p2 * (2. * cosl1p2 * l1 + p2)))),
                atan2(-1. + 3. * (powr<2>(l1) - cosl1p1 * l1 * p1 + cosl1p2 * l1 * p2 - cosp1p2 * p1 * p2) *
                                powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                         cosp1p2 * p1 * p2 + powr<2>(p2)),
                      1.732050807568877 *
                          powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                   cosp1p2 * p1 * p2 + powr<2>(p2)) *
                          (2. * cosl1p1 * l1 * p1 - powr<2>(p1) + p2 * (2. * cosl1p2 * l1 + p2))));
      const auto _interp54 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 +
                      (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) - 2. * cosl1p2 * l1 * powr<3>(p2) +
                      powr<4>(p2))),
                atan2(-1. + 3. * l1 * (l1 - cosl1p2 * p2) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                      1.732050807568877 * (2. * cosl1p2 * l1 - p2) * p2 *
                          powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2))));
      const auto _interp56 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 +
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
      const auto _interp58 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                sqrt(powr<-2>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)) *
                     (powr<4>(l1) - 2. * cosl1p2 * powr<3>(l1) * p2 +
                      (-1. + 4. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) - 2. * cosl1p2 * l1 * powr<3>(p2) +
                      powr<4>(p2))),
                atan2(-1. + 3. * l1 * (l1 - cosl1p2 * p2) * powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2)),
                      1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + p2) *
                          powr<-1>(powr<2>(l1) - cosl1p2 * l1 * p2 + powr<2>(p2))));
      const auto _interp59 = ZAcbc(
          0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                   l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
          0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                              l1 * (cosl1p1 * p1 + cosl1p2 * p2)) *
                     (powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosl1p2 * l1 * p2 +
                              2. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
                      3. * powr<2>(-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 +
                                   p2 * (-2. * cosl1p2 * l1 + p2)))),
          atan2(-1. + 3. * l1 * (l1 - cosl1p1 * p1 - cosl1p2 * p2) *
                          powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                                   l1 * (cosl1p1 * p1 + cosl1p2 * p2)),
                -1.732050807568877 *
                    (-2. * cosl1p1 * l1 * p1 + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + p2 * (-2. * cosl1p2 * l1 + p2)) *
                    powr<-1>(powr<2>(l1) + powr<2>(p1) + 2. * cosp1p2 * p1 * p2 + powr<2>(p2) -
                             l1 * (cosl1p1 * p1 + cosl1p2 * p2))));
      const auto _interp60 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 +
                                                            powr<2>(p2) - l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
                                   0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                                       l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)) *
                                              (3. * powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                                               powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                                       4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)))),
                                   atan2(2. - 3. * powr<2>(p1) *
                                                  powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                                                           2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
                                         1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                                             powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                                      l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2))));
      const auto _interp51 =
          ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                         l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
                0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                    l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                           (3. * powr<2>(p2) * powr<2>(-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) +
                            powr<2>(powr<2>(p2) - 2. * (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                                                        p1 * (p1 + cosp1p2 * p2))))),
                atan2(-1. + 3. *
                                powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                         l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                                (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) + p1 * (p1 + cosp1p2 * p2)),
                      1.732050807568877 * (2. * cosl1p2 * l1 - 2. * cosp1p2 * p1 - p2) * p2 *
                          powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                   l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2))));
      const auto _interp33 = ZA4SP(0.7071067811865475 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 +
                                                             powr<2>(p2) - l1 * (cosl1p1 * p1 + cosl1p2 * p2)));
      const auto _interp24 =
          ZA3(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                       l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                  l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                         (3. * powr<2>(p2) * powr<2>(-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) +
                          powr<2>(powr<2>(p2) - 2. * (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) +
                                                      p1 * (p1 + cosp1p2 * p2))))),
              atan2(-1. + 3. *
                              powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                       l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2)) *
                              (powr<2>(l1) - l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2) + p1 * (p1 + cosp1p2 * p2)),
                    1.732050807568877 * p2 * (-2. * cosl1p2 * l1 + 2. * cosp1p2 * p1 + p2) *
                        powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                 l1 * (2. * cosl1p1 * p1 + cosl1p2 * p2))));
      const auto _interp30 = ZA3(0.816496580927726 * sqrt(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                                          l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)),
                                 0.5 * sqrt(powr<-2>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                                     l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2)) *
                                            (3. * powr<2>(p1) * powr<2>(-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) +
                                             powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) +
                                                     4. * cosl1p2 * l1 * p2 - 2. * p2 * (cosp1p2 * p1 + p2)))),
                                 atan2(2. - 3. * powr<2>(p1) *
                                                powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) -
                                                         2. * cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 + powr<2>(p2)),
                                       1.732050807568877 * p1 * (-2. * cosl1p1 * l1 + p1 + 2. * cosp1p2 * p2) *
                                           powr<-1>(powr<2>(l1) + powr<2>(p1) + cosp1p2 * p1 * p2 + powr<2>(p2) -
                                                    l1 * (cosl1p1 * p1 + 2. * cosl1p2 * p2))));
      const auto _interp10 = ZA(sqrt(powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)));
      const auto _interp12 =
          ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
              sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)) *
                   (powr<4>(l1) - 2. * cosl1p1 * powr<3>(l1) * p1 +
                    (-1. + 4. * powr<2>(cosl1p1)) * powr<2>(l1) * powr<2>(p1) - 2. * cosl1p1 * l1 * powr<3>(p1) +
                    powr<4>(p1))),
              atan2(-1. + 3. * l1 * (l1 - cosl1p1 * p1) * powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1)),
                    1.732050807568877 * (2. * cosl1p1 * l1 - p1) * p1 *
                        powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1))));
      const auto _interp14 =
          ZA3(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
              sqrt(1. + 3. * (-1. + powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p2) *
                            powr<-2>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2))),
              atan2(-1. + 3. * l1 * (l1 + cosl1p2 * p2) * powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2)),
                    1.732050807568877 * p2 * (2. * cosl1p2 * l1 + p2) *
                        powr<-1>(powr<2>(l1) + cosl1p2 * l1 * p2 + powr<2>(p2))));
      const auto _interp16 =
          ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                       cosp1p2 * p1 * p2 + powr<2>(p2)),
              0.5 * sqrt(powr<-2>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                  cosp1p2 * p1 * p2 + powr<2>(p2)) *
                         (powr<2>(-2. * powr<2>(l1) + 2. * cosl1p1 * l1 * p1 + powr<2>(p1) - 2. * cosl1p2 * l1 * p2 +
                                  4. * cosp1p2 * p1 * p2 + powr<2>(p2)) +
                          3. * powr<2>(2. * cosl1p1 * l1 * p1 - powr<2>(p1) + p2 * (2. * cosl1p2 * l1 + p2)))),
              atan2(-1. + 3. * (powr<2>(l1) - cosl1p1 * l1 * p1 + cosl1p2 * l1 * p2 - cosp1p2 * p1 * p2) *
                              powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 +
                                       cosp1p2 * p1 * p2 + powr<2>(p2)),
                    1.732050807568877 *
                        powr<-1>(powr<2>(l1) - cosl1p1 * l1 * p1 + powr<2>(p1) + cosl1p2 * l1 * p2 + cosp1p2 * p1 * p2 +
                                 powr<2>(p2)) *
                        (-2. * cosl1p1 * l1 * p1 + powr<2>(p1) - p2 * (2. * cosl1p2 * l1 + p2))));
      // clang-format off
      using _T = decltype(_den1 + _den10 + _den11 + _den13 + _den14 + _den15 + _den16 + _den17 + _den2 + _den3 + _den4 + _den5 + _den6 + _den8 + _den9 + _interp1 + _interp10 + _interp12 + _interp14 + _interp16 + _interp17 + _interp18 + _interp2 + _interp20 + _interp22 + _interp24 + _interp25 + _interp26 + _interp28 + _interp3 + _interp30 + _interp31 + _interp32 + _interp33 + _interp34 + _interp35 + _interp36 + _interp37 + _interp38 + _interp39 + _interp4 + _interp40 + _interp41 + _interp42 + _interp43 + _interp45 + _interp47 + _interp49 + _interp5 + _interp51 + _interp52 + _interp54 + _interp56 + _interp57 + _interp58 + _interp59 + _interp6 + _interp60 + _interp7 + _interp8 + _interp9 + cosl1p1 + cosl1p2 + cosp1p2 + k + l1 + p1 + p2);
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _cse1_k1 = -8. * powr<2>(cosp1p2);
        const auto _cse2_k1 = 4. * cosl1p1 * cosl1p2 * cosp1p2;
        const auto _cse3_k1 = -5. * powr<2>(cosp1p2);
        const auto _cse4_k1 = -5. + powr<2>(cosp1p2);
        const auto _cse5_k1 = _cse4_k1 * powr<2>(cosl1p2);
        const auto _cse6_k1 = 5. + _cse2_k1 + _cse3_k1 + _cse5_k1;
        const auto _cse7_k1 = -powr<2>(cosp1p2);
        const auto _cse8_k1 = -powr<2>(cosl1p1);
        const auto _cse9_k1 = 12. * powr<2>(cosp1p2);
        // clang-format off
        _acc += -4.5 * _den1 * _den10 * _den17 * _den2 * _den4 * _den6 * _interp28 * _interp32 * powr<2>(p2) * fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k), fma(_interp3, _interp4 * (1. + powr<6>(k)), fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.))) * fma(powr<2>(l1), (8. + _cse1_k1 + powr<2>(cosl1p1) + 6. * cosl1p1 * cosl1p2 * cosp1p2 + powr<2>(cosl1p2) * (-8. + powr<2>(cosp1p2))) * powr<2>(p1) + (powr<2>(cosl1p1) * cosp1p2 + cosl1p1 * cosl1p2 * (4. + 6. * powr<2>(cosp1p2)) + cosp1p2 * (8. + _cse1_k1 + powr<2>(cosl1p2) * (-12. + powr<2>(cosp1p2)))) * p1 * p2 + _cse6_k1 * powr<2>(p2), fma(powr<2>(p2), _cse6_k1 * powr<2>(p1) + (cosl1p1 * cosl1p2 * (2. + 4. * powr<2>(cosp1p2)) + cosp1p2 * (5. + _cse3_k1 + powr<2>(cosl1p2) * (-7. + powr<2>(cosp1p2)))) * p1 * p2 + (3. + 2. * cosl1p1 * cosl1p2 * cosp1p2 - 3. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-3. + powr<2>(cosp1p2))) * powr<2>(p2), fma(l1, p2 * (powr<3>(cosl1p2) * ((8. + _cse7_k1) * powr<2>(p1) + (12. + _cse7_k1) * cosp1p2 * p1 * p2 + (5. + _cse7_k1) * powr<2>(p2)) + cosl1p1 * powr<2>(cosl1p2) * (-6. * cosp1p2 * powr<2>(p1) - 4. * p1 * p2 - 6. * powr<2>(cosp1p2) * p1 * p2 - 4. * cosp1p2 * powr<2>(p2)) + cosl1p1 * (-4. * cosp1p2 * powr<2>(p1) - 2. * p1 * p2 - 4. * powr<2>(cosp1p2) * p1 * p2 - 2. * cosp1p2 * powr<2>(p2)) + cosl1p2 * ((-8. + _cse8_k1 + _cse9_k1) * powr<2>(p1) + (-6. + _cse8_k1 + _cse9_k1) * cosp1p2 * p1 * p2 + (-5. + 7. * powr<2>(cosp1p2)) * powr<2>(p2))), 0.)));
        // clang-format on
      }
      { // subkernel 2
        const auto _cse1_k2 = -8. * powr<2>(cosp1p2);
        const auto _cse2_k2 = 4. * cosl1p1 * cosl1p2 * cosp1p2;
        const auto _cse3_k2 = -5. * powr<2>(cosp1p2);
        const auto _cse4_k2 = -5. + powr<2>(cosp1p2);
        const auto _cse5_k2 = _cse4_k2 * powr<2>(cosl1p1);
        const auto _cse6_k2 = 5. + _cse2_k2 + _cse3_k2 + _cse5_k2;
        const auto _cse7_k2 = -powr<2>(cosp1p2);
        const auto _cse8_k2 = -powr<2>(cosl1p2);
        const auto _cse9_k2 = 12. * powr<2>(cosp1p2);
        // clang-format off
        _acc += -4.5 * _den1 * _den17 * _den2 * _den4 * _den5 * _den9 * _interp20 * _interp31 * powr<2>(p1) * fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k), fma(_interp3, _interp4 * (1. + powr<6>(k)), fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.))) * fma(powr<2>(p1), (3. + 2. * cosl1p1 * cosl1p2 * cosp1p2 - 3. * powr<2>(cosp1p2) + powr<2>(cosl1p1) * (-3. + powr<2>(cosp1p2))) * powr<2>(p1) + (5. * cosp1p2 - 5. * powr<3>(cosp1p2) + cosl1p1 * cosl1p2 * (2. + 4. * powr<2>(cosp1p2)) + powr<2>(cosl1p1) * (-7. * cosp1p2 + powr<3>(cosp1p2))) * p1 * p2 + _cse6_k2 * powr<2>(p2), fma(powr<2>(l1), _cse6_k2 * powr<2>(p1) + ((8. + _cse1_k2 + powr<2>(cosl1p2)) * cosp1p2 + cosl1p1 * cosl1p2 * (4. + 6. * powr<2>(cosp1p2)) + powr<2>(cosl1p1) * (-12. * cosp1p2 + powr<3>(cosp1p2))) * p1 * p2 + (8. + _cse1_k2 + powr<2>(cosl1p2) + 6. * cosl1p1 * cosl1p2 * cosp1p2 + powr<2>(cosl1p1) * (-8. + powr<2>(cosp1p2))) * powr<2>(p2), fma(l1, p1 * (powr<3>(cosl1p1) * ((5. + _cse7_k2) * powr<2>(p1) + (12. + _cse7_k2) * cosp1p2 * p1 * p2 + (8. + _cse7_k2) * powr<2>(p2)) + cosl1p1 * ((-5. + 7. * powr<2>(cosp1p2)) * powr<2>(p1) + (-6. + _cse8_k2 + _cse9_k2) * cosp1p2 * p1 * p2 + (-8. + _cse8_k2 + _cse9_k2) * powr<2>(p2)) + powr<2>(cosl1p1) * cosl1p2 * (-4. * cosp1p2 * powr<2>(p1) - 4. * p1 * p2 - 6. * powr<2>(cosp1p2) * p1 * p2 - 6. * cosp1p2 * powr<2>(p2)) + cosl1p2 * (-2. * cosp1p2 * powr<2>(p1) - 2. * p1 * p2 - 4. * powr<2>(cosp1p2) * p1 * p2 - 4. * cosp1p2 * powr<2>(p2))), 0.)));
        // clang-format on
      }
      { // subkernel 3
        const auto _cse1_k3 = -cosp1p2 * powr<2>(p1);
        const auto _cse2_k3 = -p1 * p2;
        const auto _cse3_k3 = -powr<2>(cosp1p2) * p1 * p2;
        const auto _cse4_k3 = -cosp1p2 * powr<2>(p2);
        const auto _cse5_k3 = _cse1_k3 + _cse2_k3 + _cse3_k3 + _cse4_k3;
        const auto _cse6_k3 = 1. + powr<2>(cosp1p2);
        // clang-format off
        _acc += -0.75 * _den1 * _den13 * _den17 * _den3 * _den8 * _interp34 * _interp35 * _interp36 * powr<2>(l1) * fma(_cse5_k3, powr<2>(cosl1p2) * p1 * p2, fma(1. - powr<2>(cosp1p2), powr<2>(p1) * powr<2>(p2), fma(powr<3>(cosl1p2), l1 * powr<2>(p2) * (cosp1p2 * p1 + p2), fma(powr<3>(cosl1p1), l1 * powr<2>(p1) * (-p1 - cosp1p2 * p2), fma(cosl1p2, l1 * (-cosp1p2 * powr<3>(p1) - 2. * powr<2>(cosp1p2) * powr<2>(p1) * p2 - 2. * cosp1p2 * p1 * powr<2>(p2) - powr<3>(p2)), fma(powr<2>(cosl1p1), p1 * (_cse5_k3 * p2 + cosl1p2 * l1 * (_cse2_k3 + _cse4_k3 + cosp1p2 * powr<2>(p1) + powr<2>(cosp1p2) * p1 * p2)), fma(cosl1p1, cosl1p2 * p1 * p2 * (_cse6_k3 * powr<2>(p1) + cosp1p2 * (3. + powr<2>(cosp1p2)) * p1 * p2 + _cse6_k3 * powr<2>(p2)) + l1 * (powr<3>(p1) + (2. + powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (powr<2>(cosl1p2) + 2. * powr<2>(cosp1p2) - powr<2>(cosl1p2) * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + (1. - powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)), 0.))))))) * fma(_interp2, _interp37, fma(_interp3, _interp38, fma(_interp2, -50. * _interp38 + 50. * _interp39, 0.)));
        // clang-format on
      }
      { // subkernel 4
        const auto _cse1_k4 = -cosp1p2 * powr<2>(p1);
        const auto _cse2_k4 = -p1 * p2;
        const auto _cse3_k4 = -powr<2>(cosp1p2) * p1 * p2;
        const auto _cse4_k4 = -cosp1p2 * powr<2>(p2);
        const auto _cse5_k4 = _cse1_k4 + _cse2_k4 + _cse3_k4 + _cse4_k4;
        const auto _cse6_k4 = 1. + powr<2>(cosp1p2);
        // clang-format off
        _acc += -0.75 * _den1 * _den13 * _den17 * _den3 * _den8 * _interp43 * _interp45 * _interp47 * powr<2>(l1) * fma(_cse5_k4, powr<2>(cosl1p2) * p1 * p2, fma(1. - powr<2>(cosp1p2), powr<2>(p1) * powr<2>(p2), fma(powr<3>(cosl1p2), l1 * powr<2>(p2) * (cosp1p2 * p1 + p2), fma(powr<3>(cosl1p1), l1 * powr<2>(p1) * (-p1 - cosp1p2 * p2), fma(cosl1p2, l1 * (-cosp1p2 * powr<3>(p1) - 2. * powr<2>(cosp1p2) * powr<2>(p1) * p2 - 2. * cosp1p2 * p1 * powr<2>(p2) - powr<3>(p2)), fma(powr<2>(cosl1p1), p1 * (_cse5_k4 * p2 + cosl1p2 * l1 * (_cse2_k4 + _cse4_k4 + cosp1p2 * powr<2>(p1) + powr<2>(cosp1p2) * p1 * p2)), fma(cosl1p1, cosl1p2 * p1 * p2 * (_cse6_k4 * powr<2>(p1) + cosp1p2 * (3. + powr<2>(cosp1p2)) * p1 * p2 + _cse6_k4 * powr<2>(p2)) + l1 * (powr<3>(p1) + (2. + powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (powr<2>(cosl1p2) + 2. * powr<2>(cosp1p2) - powr<2>(cosl1p2) * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + (1. - powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)), 0.))))))) * fma(_interp2, _interp37, fma(_interp3, _interp38, fma(_interp2, -50. * _interp38 + 50. * _interp39, 0.)));
        // clang-format on
      }
      { // subkernel 5
        const auto _cse1_k5 = -1. + powr<2>(cosp1p2);
        const auto _cse2_k5 = 1. + powr<2>(cosp1p2);
        // clang-format off
        _acc += 0.75 * _den1 * _den11 * _den16 * _den17 * _den3 * _interp49 * _interp54 * _interp56 * powr<2>(l1) * fma(_interp2, _interp37, fma(_interp3, _interp38, fma(_interp2, -50. * _interp38 + 50. * _interp39, 0.))) * fma(powr<3>(cosl1p2), l1 * (-cosp1p2 * p1 - p2) * powr<2>(p2), fma(powr<3>(cosl1p1), l1 * powr<2>(p1) * (p1 + cosp1p2 * p2), fma(powr<2>(p2), _cse1_k5 * powr<2>(p1) + cosp1p2 * (-2. + 2. * powr<2>(cosp1p2)) * p1 * p2 + _cse1_k5 * powr<2>(p2), fma(powr<2>(cosl1p2), p2 * (-cosp1p2 * powr<3>(p1) + powr<2>(p1) * p2 - 2. * powr<2>(cosp1p2) * powr<2>(p1) * p2 + cosp1p2 * p1 * powr<2>(p2) - powr<3>(cosp1p2) * p1 * powr<2>(p2) + powr<3>(p2) - powr<2>(cosp1p2) * powr<3>(p2)), fma(powr<2>(cosl1p1), p1 * (cosp1p2 * p1 * p2 * (-p1 - cosp1p2 * p2) + cosl1p2 * l1 * (-cosp1p2 * powr<2>(p1) + p1 * p2 - powr<2>(cosp1p2) * p1 * p2 + cosp1p2 * powr<2>(p2))), fma(cosl1p2, l1 * (2. * powr<2>(cosp1p2) * powr<2>(p1) * p2 + powr<3>(p2) + cosp1p2 * (powr<3>(p1) + 2. * p1 * powr<2>(p2))), fma(cosl1p1, cosl1p2 * p1 * p2 * (_cse2_k5 * powr<2>(p1) + _cse2_k5 * cosp1p2 * p1 * p2 + (1. - powr<2>(cosp1p2)) * powr<2>(p2)) + l1 * (-powr<3>(p1) + (-2. - powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (_cse1_k5 * powr<2>(cosl1p2) - 2. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + (-1. + powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)), 0.)))))));
        // clang-format on
      }
      { // subkernel 6
        const auto _cse1_k6 = -1. + powr<2>(cosp1p2);
        const auto _cse2_k6 = 1. + powr<2>(cosp1p2);
        // clang-format off
        _acc += 0.75 * _den1 * _den11 * _den16 * _den17 * _den3 * _interp58 * _interp59 * _interp60 * powr<2>(l1) * fma(_interp2, _interp37, fma(_interp3, _interp38, fma(_interp2, -50. * _interp38 + 50. * _interp39, 0.))) * fma(powr<3>(cosl1p2), l1 * (-cosp1p2 * p1 - p2) * powr<2>(p2), fma(powr<3>(cosl1p1), l1 * powr<2>(p1) * (p1 + cosp1p2 * p2), fma(powr<2>(p2), _cse1_k6 * powr<2>(p1) + cosp1p2 * (-2. + 2. * powr<2>(cosp1p2)) * p1 * p2 + _cse1_k6 * powr<2>(p2), fma(powr<2>(cosl1p2), p2 * (-cosp1p2 * powr<3>(p1) + powr<2>(p1) * p2 - 2. * powr<2>(cosp1p2) * powr<2>(p1) * p2 + cosp1p2 * p1 * powr<2>(p2) - powr<3>(cosp1p2) * p1 * powr<2>(p2) + powr<3>(p2) - powr<2>(cosp1p2) * powr<3>(p2)), fma(powr<2>(cosl1p1), p1 * (cosp1p2 * p1 * p2 * (-p1 - cosp1p2 * p2) + cosl1p2 * l1 * (-cosp1p2 * powr<2>(p1) + p1 * p2 - powr<2>(cosp1p2) * p1 * p2 + cosp1p2 * powr<2>(p2))), fma(cosl1p2, l1 * (2. * powr<2>(cosp1p2) * powr<2>(p1) * p2 + powr<3>(p2) + cosp1p2 * (powr<3>(p1) + 2. * p1 * powr<2>(p2))), fma(cosl1p1, cosl1p2 * p1 * p2 * (_cse2_k6 * powr<2>(p1) + _cse2_k6 * cosp1p2 * p1 * p2 + (1. - powr<2>(cosp1p2)) * powr<2>(p2)) + l1 * (-powr<3>(p1) + (-2. - powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (_cse1_k6 * powr<2>(cosl1p2) - 2. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + (-1. + powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)), 0.)))))));
        // clang-format on
      }
      { // subkernel 7
        const auto _cse1_k7 = -powr<2>(cosp1p2);
        const auto _cse2_k7 = 1. + _cse1_k7;
        const auto _cse3_k7 = -1. + _cse1_k7;
        const auto _cse4_k7 = -2. * powr<2>(cosp1p2);
        const auto _cse5_k7 = -1. + powr<2>(cosp1p2);
        // clang-format off
        _acc += -4.5 * _den1 * _den16 * _den17 * _den3 * _den8 * _interp34 * _interp49 * _interp51 * powr<2>(l1) * fma(_interp2, _interp37, fma(_interp3, _interp38, fma(_interp2, -50. * _interp38 + 50. * _interp39, 0.))) * fma(powr<3>(cosl1p2), l1 * (-cosp1p2 * p1 - p2) * powr<2>(p2), fma(powr<2>(cosl1p2), cosp1p2 * p1 * powr<2>(p2) * (cosp1p2 * p1 + p2), fma(powr<3>(cosl1p1), l1 * powr<2>(p1) * (p1 + cosp1p2 * p2), fma(powr<2>(p1), _cse2_k7 * powr<2>(p1) + (2. + _cse4_k7) * cosp1p2 * p1 * p2 + _cse2_k7 * powr<2>(p2), fma(cosl1p2, l1 * (2. * powr<2>(cosp1p2) * powr<2>(p1) * p2 + powr<3>(p2) + cosp1p2 * (powr<3>(p1) + 2. * p1 * powr<2>(p2))), fma(cosl1p1, cosl1p2 * p1 * p2 * (_cse5_k7 * powr<2>(p1) + _cse3_k7 * cosp1p2 * p1 * p2 + _cse3_k7 * powr<2>(p2)) + l1 * (-powr<3>(p1) + (-2. - powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (_cse4_k7 + _cse5_k7 * powr<2>(cosl1p2)) * p1 * powr<2>(p2) + (-1. + powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)), fma(powr<2>(cosl1p1), p1 * (_cse5_k7 * powr<3>(p1) + _cse5_k7 * cosp1p2 * powr<2>(p1) * p2 + (-1. + 2. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + cosp1p2 * powr<3>(p2) + cosl1p2 * l1 * (p1 * p2 - powr<2>(cosp1p2) * p1 * p2 + cosp1p2 * (-powr<2>(p1) + powr<2>(p2)))), 0.)))))));
        // clang-format on
      }
      { // subkernel 8
        const auto _cse1_k8 = 7. * powr<2>(cosp1p2);
        const auto _cse2_k8 = -16. * powr<4>(cosp1p2);
        const auto _cse3_k8 = 9. * powr<2>(cosp1p2);
        const auto _cse4_k8 = -3. + _cse3_k8;
        const auto _cse5_k8 = 14. * powr<2>(cosp1p2);
        const auto _cse6_k8 = -16. + _cse5_k8;
        const auto _cse7_k8 = _cse6_k8 * cosl1p1 * cosl1p2 * cosp1p2;
        const auto _cse8_k8 = 2. * powr<4>(cosp1p2);
        const auto _cse9_k8 = -6. + _cse8_k8;
        const auto _cse10_k8 = 10. * powr<2>(cosp1p2);
        const auto _cse11_k8 = -6. + _cse10_k8;
        const auto _cse12_k8 = _cse11_k8 * cosl1p1 * cosl1p2;
        const auto _cse13_k8 = -13. * powr<2>(cosp1p2);
        const auto _cse14_k8 = 2. * cosl1p1 * cosl1p2 * cosp1p2;
        const auto _cse15_k8 = -3. * powr<2>(cosp1p2);
        const auto _cse16_k8 = -3. + powr<2>(cosp1p2);
        const auto _cse17_k8 = -4. * powr<4>(cosp1p2);
        const auto _cse18_k8 = -5. + _cse1_k8;
        const auto _cse19_k8 = 12. * powr<2>(cosp1p2);
        const auto _cse20_k8 = -10. + _cse19_k8;
        const auto _cse21_k8 = _cse20_k8 * cosl1p1 * cosl1p2;
        const auto _cse22_k8 = -12. * powr<2>(cosp1p2);
        // clang-format off
        _acc += -18. * _den1 * _den14 * _den15 * _den17 * _den2 * _den4 * _interp22 * _interp33 * fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k), fma(_interp3, _interp4 * (1. + powr<6>(k)), fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.))) * fma(3. + _cse14_k8 + _cse15_k8 + _cse16_k8 * powr<2>(cosl1p1), powr<6>(p1), fma(_cse12_k8 + (13. + _cse13_k8 + 2. * powr<2>(cosl1p2)) * cosp1p2 + powr<2>(cosl1p1) * (-9. * cosp1p2 + 3. * powr<3>(cosp1p2)), powr<5>(p1) * p2, fma(9. + _cse1_k8 + _cse2_k8 + _cse7_k8 + _cse9_k8 * powr<2>(cosl1p1) + _cse4_k8 * powr<2>(cosl1p2), powr<4>(p1) * powr<2>(p2), fma(powr<2>(cosl1p1) * (-7. * cosp1p2 + 11. * powr<3>(cosp1p2)) + cosl1p1 * cosl1p2 * (-12. + 4. * powr<4>(cosp1p2)) + cosp1p2 * (26. + _cse17_k8 - 22. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-7. + 11. * powr<2>(cosp1p2))), powr<3>(p1) * powr<3>(p2), fma(9. + _cse1_k8 + _cse2_k8 + _cse7_k8 + _cse4_k8 * powr<2>(cosl1p1) + _cse9_k8 * powr<2>(cosl1p2), powr<2>(p1) * powr<4>(p2), fma(_cse12_k8 + 2. * powr<2>(cosl1p1) * cosp1p2 + cosp1p2 * (13. + _cse13_k8 + powr<2>(cosl1p2) * (-9. + 3. * powr<2>(cosp1p2))), p1 * powr<5>(p2), fma(3. + _cse14_k8 + _cse15_k8 + _cse16_k8 * powr<2>(cosl1p2), powr<6>(p2), fma(powr<2>(l1), (5. + 4. * cosl1p1 * cosl1p2 * cosp1p2 - 5. * powr<2>(cosp1p2) + powr<2>(cosl1p1) * (-5. + powr<2>(cosp1p2))) * powr<4>(p1) + (_cse21_k8 + (12. + _cse22_k8 + 3. * powr<2>(cosl1p2)) * cosp1p2 + powr<2>(cosl1p1) * (-6. * cosp1p2 + powr<3>(cosp1p2))) * powr<3>(p1) * p2 + (10. + _cse17_k8 + _cse18_k8 * powr<2>(cosl1p1) + _cse18_k8 * powr<2>(cosl1p2) - 6. * powr<2>(cosp1p2) + cosl1p1 * cosl1p2 * cosp1p2 * (-10. + 6. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<2>(p2) + (_cse21_k8 + 3. * powr<2>(cosl1p1) * cosp1p2 + cosp1p2 * (12. + _cse22_k8 + powr<2>(cosl1p2) * (-6. + powr<2>(cosp1p2)))) * p1 * powr<3>(p2) + (5. + 4. * cosl1p1 * cosl1p2 * cosp1p2 - 5. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-5. + powr<2>(cosp1p2))) * powr<4>(p2), fma(l1, powr<3>(cosl1p1) * powr<2>(p1) * ((5. - powr<2>(cosp1p2)) * powr<3>(p1) + cosp1p2 * (6. - powr<2>(cosp1p2)) * powr<2>(p1) * p2 + (5. - 7. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) - 3. * cosp1p2 * powr<3>(p2)) + cosl1p1 * (_cse18_k8 * powr<5>(p1) + cosp1p2 * (-10. - 7. * powr<2>(cosl1p2) + 16. * powr<2>(cosp1p2)) * powr<4>(p1) * p2 + (-10. + _cse10_k8 + 4. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (15. - 19. * powr<2>(cosp1p2))) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (-12. + 8. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (16. - 7. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<3>(p2) + (-5. + (15. + _cse13_k8) * powr<2>(cosl1p2) - powr<2>(cosp1p2)) * p1 * powr<4>(p2) + (-2. - 4. * powr<2>(cosl1p2)) * cosp1p2 * powr<5>(p2)) + powr<2>(cosl1p1) * cosl1p2 * p1 * (15. * powr<3>(p1) * p2 - 7. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) + 15. * p1 * powr<3>(p2) + powr<2>(cosp1p2) * (-13. * powr<3>(p1) * p2 - 19. * p1 * powr<3>(p2)) + cosp1p2 * (-4. * powr<4>(p1) + 16. * powr<2>(p1) * powr<2>(p2) - 7. * powr<4>(p2))) + cosl1p2 * (-5. * powr<4>(p1) * p2 + (-10. + 5. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<3>(p2) + 4. * powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) + (-5. + 5. * powr<2>(cosl1p2)) * powr<5>(p2) + powr<3>(cosp1p2) * (8. * powr<3>(p1) * powr<2>(p2) + (16. - powr<2>(cosl1p2)) * p1 * powr<4>(p2)) + cosp1p2 * (-2. * powr<5>(p1) + (-12. - 3. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<2>(p2) + (-10. + 6. * powr<2>(cosl1p2)) * p1 * powr<4>(p2)) + powr<2>(cosp1p2) * (-powr<4>(p1) * p2 + (10. - 7. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<3>(p2) + (7. - powr<2>(cosl1p2)) * powr<5>(p2))), 0.)))))))));
        // clang-format on
      }
      { // subkernel 9
        const auto _cse1_k9 = -2. * powr<2>(cosp1p2);
        const auto _cse2_k9 = -3. * powr<2>(cosp1p2);
        const auto _cse3_k9 = 3. + _cse2_k9;
        const auto _cse4_k9 = -11. * powr<2>(cosp1p2);
        const auto _cse5_k9 = 11. + _cse4_k9;
        const auto _cse6_k9 = 14. * powr<2>(cosp1p2);
        const auto _cse7_k9 = -10. * powr<2>(cosp1p2);
        const auto _cse8_k9 = -3. * powr<4>(l1);
        const auto _cse9_k9 = -6. * powr<2>(p1);
        const auto _cse10_k9 = -14. * powr<4>(p1);
        const auto _cse11_k9 = 13. * powr<2>(cosp1p2);
        const auto _cse12_k9 = -12. * powr<2>(cosp1p2);
        const auto _cse13_k9 = -26. * powr<2>(cosl1p2);
        const auto _cse14_k9 = -52. * powr<2>(cosp1p2);
        const auto _cse15_k9 = -13. * powr<4>(cosp1p2);
        const auto _cse16_k9 = 16. * powr<2>(p1) * powr<4>(p2);
        const auto _cse17_k9 = 3. * powr<6>(p2);
        const auto _cse18_k9 = -6. * powr<2>(cosp1p2);
        const auto _cse19_k9 = 3. * powr<2>(cosp1p2);
        const auto _cse20_k9 = -3. + _cse19_k9;
        const auto _cse21_k9 = 12. * powr<2>(cosp1p2);
        const auto _cse22_k9 = 26. * powr<2>(cosp1p2);
        // clang-format off
        _acc += 18. * _den1 * _den14 * _den15 * _den17 * _den2 * _den4 * _den5 * _den9 * _interp20 * _interp22 * _interp24 * fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k), fma(_interp3, _interp4 * (1. + powr<6>(k)), fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.))) * fma(powr<5>(cosl1p2), powr<5>(l1) * (-cosp1p2 * p1 - p2) * powr<4>(p2), fma(powr<6>(cosl1p1), powr<4>(l1) * powr<5>(p1) * (-2. * p1 - 2. * cosp1p2 * p2), fma(powr<5>(cosl1p1), powr<3>(l1) * powr<4>(p1) * ((15. + _cse1_k9) * powr<3>(p1) + (28. + _cse1_k9) * cosp1p2 * powr<2>(p1) * p2 + (14. + powr<2>(cosp1p2)) * p1 * powr<2>(p2) + 2. * cosp1p2 * powr<3>(p2) + powr<2>(l1) * (13. * p1 + 13. * cosp1p2 * p2) + cosl1p2 * l1 * (2. * cosp1p2 * powr<2>(p1) - 5. * p1 * p2 + 2. * powr<2>(cosp1p2) * p1 * p2 - 5. * cosp1p2 * powr<2>(p2))), fma(powr<4>(cosl1p2), powr<4>(l1) * powr<3>(p2) * (6. * powr<2>(l1) * p2 + 11. * powr<2>(p1) * p2 + 4. * powr<2>(cosp1p2) * powr<2>(p1) * p2 + 6. * powr<3>(p2) + cosp1p2 * (6. * powr<2>(l1) * p1 + 7. * powr<3>(p1) + 14. * p1 * powr<2>(p2))), fma(powr<2>(p1), powr<6>(l1) * (_cse3_k9 * powr<2>(p1) + (6. + _cse18_k9) * cosp1p2 * p1 * p2 + _cse3_k9 * powr<2>(p2)) + powr<4>(l1) * (_cse5_k9 * powr<4>(p1) + cosp1p2 * (33. - 33. * powr<2>(cosp1p2)) * powr<3>(p1) * p2 + (21. + powr<2>(cosp1p2) - 22. * powr<4>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) + cosp1p2 * (31. - 31. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) + (10. + _cse7_k9) * powr<4>(p2)) + powr<2>(l1) * (_cse5_k9 * powr<6>(p1) + cosp1p2 * (44. - 44. * powr<2>(cosp1p2)) * powr<5>(p1) * p2 + (29. + 22. * powr<2>(cosp1p2) - 51. * powr<4>(cosp1p2)) * powr<4>(p1) * powr<2>(p2) + cosp1p2 * (72. - 58. * powr<2>(cosp1p2) - 14. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<3>(p2) + (21. + _cse6_k9 - 35. * powr<4>(cosp1p2)) * powr<2>(p1) * powr<4>(p2) + cosp1p2 * (20. - 20. * powr<2>(cosp1p2)) * p1 * powr<5>(p2) + _cse3_k9 * powr<6>(p2)) + p1 * (_cse3_k9 * powr<7>(p1) + cosp1p2 * (15. - 15. * powr<2>(cosp1p2)) * powr<6>(p1) * p2 + (11. + _cse6_k9 - 25. * powr<4>(cosp1p2)) * powr<5>(p1) * powr<2>(p2) + cosp1p2 * (39. - 24. * powr<2>(cosp1p2) - 15. * powr<4>(cosp1p2)) * powr<4>(p1) * powr<3>(p2) + (11. + 30. * powr<2>(cosp1p2) - 39. * powr<4>(cosp1p2) - 2. * powr<6>(cosp1p2)) * powr<3>(p1) * powr<4>(p2) + (23. + _cse15_k9 + _cse7_k9) * cosp1p2 * powr<2>(p1) * powr<5>(p2) + (3. + 9. * powr<2>(cosp1p2) - 12. * powr<4>(cosp1p2)) * p1 * powr<6>(p2) + _cse3_k9 * cosp1p2 * powr<7>(p2)), fma(powr<3>(cosl1p2), powr<3>(l1) * powr<2>(p2) * (-4. * powr<3>(cosp1p2) * powr<3>(p1) * powr<2>(p2) + powr<2>(cosp1p2) * powr<2>(p1) * p2 * (-11. * powr<2>(l1) - 17. * powr<2>(p1) - 25. * powr<2>(p2)) + cosp1p2 * p1 * (_cse8_k9 - 5. * powr<4>(p1) - 41. * powr<2>(p1) * powr<2>(p2) - 18. * powr<4>(p2) + powr<2>(l1) * (_cse9_k9 - 23. * powr<2>(p2))) + p2 * (_cse8_k9 - 12. * powr<4>(p1) - 11. * powr<2>(p1) * powr<2>(p2) - 3. * powr<4>(p2) + powr<2>(l1) * (-11. * powr<2>(p1) - 7. * powr<2>(p2)))), fma(powr<2>(cosl1p2), powr<2>(l1) * p2 * (5. * powr<6>(p1) * p2 - 6. * powr<4>(l1) * powr<3>(p2) + 3. * powr<4>(p1) * powr<3>(p2) - 2. * powr<4>(cosp1p2) * powr<4>(p1) * powr<3>(p2) + powr<3>(cosp1p2) * powr<3>(p1) * powr<2>(p2) * (_cse9_k9 - 14. * powr<2>(l1) - powr<2>(p2)) + powr<2>(l1) * (3. * powr<4>(p1) * p2 - 12. * powr<2>(p1) * powr<3>(p2) - 6. * powr<5>(p2)) + powr<2>(cosp1p2) * powr<2>(p1) * p2 * (_cse10_k9 - 9. * powr<4>(l1) + 8. * powr<2>(p1) * powr<2>(p2) + 6. * powr<4>(p2) + powr<2>(l1) * (-31. * powr<2>(p1) - 13. * powr<2>(p2))) + cosp1p2 * p1 * (_cse17_k9 - 6. * powr<6>(p1) + 6. * powr<4>(p1) * powr<2>(p2) + 10. * powr<2>(p1) * powr<4>(p2) + powr<4>(l1) * (_cse9_k9 - 9. * powr<2>(p2)) + powr<2>(l1) * (_cse10_k9 - 20. * powr<2>(p1) * powr<2>(p2) - 11. * powr<4>(p2)))), fma(powr<4>(cosl1p1), powr<2>(l1) * powr<3>(p1) * (powr<4>(l1) * (-12. * p1 - 12. * cosp1p2 * p2) + cosl1p2 * powr<3>(l1) * (-13. * cosp1p2 * powr<2>(p1) + 26. * p1 * p2 - 13. * powr<2>(cosp1p2) * p1 * p2 + 26. * cosp1p2 * powr<2>(p2)) + powr<2>(l1) * ((-39. + _cse11_k9) * powr<3>(p1) + (-65. + _cse11_k9 + 5. * powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (-35. + _cse6_k9 + powr<2>(cosl1p2) * (-2. + powr<2>(cosp1p2))) * p1 * powr<2>(p2) + (5. - 6. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) + p1 * ((-25. + _cse11_k9) * powr<4>(p1) + (-62. + _cse22_k9) * cosp1p2 * powr<3>(p1) * p2 + (-51. - 5. * powr<2>(cosp1p2) + 5. * powr<4>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) + cosp1p2 * (-58. + 16. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) + (-22. + 7. * powr<2>(cosp1p2)) * powr<4>(p2)) + cosl1p2 * l1 * (31. * powr<3>(p1) * p2 - 6. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) + 33. * p1 * powr<3>(p2) + powr<2>(cosp1p2) * (-31. * powr<3>(p1) * p2 + 10. * p1 * powr<3>(p2)) + cosp1p2 * (-13. * powr<4>(p1) + 53. * powr<2>(p1) * powr<2>(p2) + 9. * powr<4>(p2)))), fma(powr<3>(cosl1p1), l1 * powr<2>(p1) * (powr<6>(l1) * (3. * p1 + 3. * cosp1p2 * p2) + cosl1p2 * powr<5>(l1) * (12. * cosp1p2 * powr<2>(p1) - 18. * p1 * p2 + 12. * powr<2>(cosp1p2) * p1 * p2 - 18. * cosp1p2 * powr<2>(p2)) + powr<4>(l1) * ((10. + _cse12_k9) * powr<3>(p1) + (8. + _cse12_k9 + _cse13_k9) * cosp1p2 * powr<2>(p1) * p2 + (20. + powr<2>(cosl1p2) - 43. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + (-21. + 27. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) + powr<2>(l1) * ((24. - 26. * powr<2>(cosp1p2)) * powr<5>(p1) + (46. + _cse13_k9 + _cse14_k9) * cosp1p2 * powr<4>(p1) * p2 + (58. + (6. + _cse14_k9) * powr<2>(cosl1p2) - 72. * powr<2>(cosp1p2) - 16. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (24. + (4. + _cse1_k9) * powr<2>(cosl1p2) - 74. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) + (31. - 77. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (14. + 21. * powr<2>(cosp1p2))) * p1 * powr<4>(p2) + (-22. + 13. * powr<2>(cosl1p2)) * cosp1p2 * powr<5>(p2)) + p1 * ((15. + _cse12_k9) * powr<6>(p1) + cosp1p2 * (48. - 36. * powr<2>(cosp1p2)) * powr<5>(p1) * p2 + (44. + 6. * powr<2>(cosp1p2) - 26. * powr<4>(cosp1p2)) * powr<4>(p1) * powr<2>(p2) + cosp1p2 * (84. + _cse14_k9 - 2. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<3>(p2) + (33. + _cse15_k9 + 4. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<4>(p2) + (24. + _cse12_k9) * cosp1p2 * p1 * powr<5>(p2) + (6. + _cse2_k9) * powr<6>(p2)) + cosl1p2 * powr<3>(l1) * (-65. * powr<3>(p1) * p2 + 16. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) + (-69. + 4. * powr<2>(cosl1p2)) * p1 * powr<3>(p2) + powr<2>(cosp1p2) * (78. * powr<3>(p1) * p2 + (-11. - 2. * powr<2>(cosl1p2)) * p1 * powr<3>(p2)) + cosp1p2 * (26. * powr<4>(p1) + (-102. + 6. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) + (-7. - 4. * powr<2>(cosl1p2)) * powr<4>(p2))) + cosl1p2 * l1 * (-44. * powr<5>(p1) * p2 - 88. * powr<3>(p1) * powr<3>(p2) + 6. * powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) - 34. * p1 * powr<5>(p2) + powr<3>(cosp1p2) * (_cse16_k9 + 72. * powr<4>(p1) * powr<2>(p2)) + powr<2>(cosp1p2) * (62. * powr<5>(p1) * p2 + 40. * powr<3>(p1) * powr<3>(p2) + p1 * powr<5>(p2)) + cosp1p2 * (12. * powr<6>(p1) - 78. * powr<4>(p1) * powr<2>(p2) - 76. * powr<2>(p1) * powr<4>(p2) - 3. * powr<6>(p2)))), fma(cosl1p2, l1 * (5. * powr<5>(cosp1p2) * powr<5>(p1) * powr<4>(p2) + powr<4>(cosp1p2) * powr<4>(p1) * powr<3>(p2) * (33. * powr<2>(l1) + 31. * powr<2>(p1) + 26. * powr<2>(p2)) + powr<3>(cosp1p2) * powr<3>(p1) * powr<2>(p2) * (34. * powr<4>(l1) + 44. * powr<4>(p1) + 65. * powr<2>(p1) * powr<2>(p2) + 18. * powr<4>(p2) + powr<2>(l1) * (88. * powr<2>(p1) + 69. * powr<2>(p2))) + powr<2>(cosp1p2) * powr<2>(p1) * p2 * (_cse16_k9 + _cse17_k9 + 6. * powr<6>(l1) + 21. * powr<6>(p1) + 37. * powr<4>(p1) * powr<2>(p2) + powr<4>(l1) * (39. * powr<2>(p1) + 44. * powr<2>(p2)) + powr<2>(l1) * (58. * powr<4>(p1) + 97. * powr<2>(p1) * powr<2>(p2) + 43. * powr<4>(p2))) + p2 * (-6. * powr<8>(p1) + 3. * powr<6>(l1) * powr<2>(p2) - 14. * powr<6>(p1) * powr<2>(p2) - 6. * powr<4>(p1) * powr<4>(p2) + powr<4>(l1) * (-6. * powr<4>(p1) + 5. * powr<2>(p1) * powr<2>(p2) + 8. * powr<4>(p2)) + powr<2>(l1) * (_cse17_k9 - 14. * powr<6>(p1) - 14. * powr<4>(p1) * powr<2>(p2) + 5. * powr<2>(p1) * powr<4>(p2))) + cosp1p2 * p1 * (3. * powr<8>(p1) - 8. * powr<6>(p1) * powr<2>(p2) - 16. * powr<4>(p1) * powr<4>(p2) - 3. * powr<2>(p1) * powr<6>(p2) + powr<6>(l1) * (3. * powr<2>(p1) + 6. * powr<2>(p2)) + powr<4>(l1) * (11. * powr<4>(p1) + 18. * powr<2>(p1) * powr<2>(p2) + 27. * powr<4>(p2)) + powr<2>(l1) * (11. * powr<6>(p1) + 2. * powr<4>(p1) * powr<2>(p2) + 26. * powr<2>(p1) * powr<4>(p2) + 15. * powr<6>(p2)))), fma(cosl1p1, powr<7>(l1) * (-3. * powr<3>(p1) + (-6. - 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (_cse18_k9 + _cse20_k9 * powr<2>(cosl1p2)) * p1 * powr<2>(p2) + (-3. + 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) + powr<5>(l1) * ((-23. + _cse21_k9) * powr<5>(p1) + cosp1p2 * (-57. + 2. * powr<2>(cosl1p2) + 24. * powr<2>(cosp1p2)) * powr<4>(p1) * p2 + (-20. - 10. * powr<2>(cosl1p2) - 32. * powr<2>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (-27. + (2. + _cse11_k9) * powr<2>(cosl1p2) - 14. * powr<4>(cosl1p2) - 22. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) + (-27. * powr<2>(cosp1p2) + powr<4>(cosl1p2) * (-14. + powr<2>(cosp1p2)) + powr<2>(cosl1p2) * (11. + 23. * powr<2>(cosp1p2))) * p1 * powr<4>(p2) + (-8. + 7. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * cosp1p2 * powr<5>(p2)) + cosl1p2 * powr<3>(p1) * p2 * (_cse20_k9 * powr<6>(p1) + (-12. + _cse21_k9) * cosp1p2 * powr<5>(p1) * p2 + (-11. + _cse1_k9 + 13. * powr<4>(cosp1p2)) * powr<4>(p1) * powr<2>(p2) + cosp1p2 * (-28. + _cse22_k9 + 2. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<3>(p2) + (-11. + _cse1_k9 + 13. * powr<4>(cosp1p2)) * powr<2>(p1) * powr<4>(p2) + (-12. + _cse21_k9) * cosp1p2 * p1 * powr<5>(p2) + _cse20_k9 * powr<6>(p2)) + l1 * powr<2>(p1) * ((-15. + _cse21_k9) * powr<7>(p1) + cosp1p2 * (-63. - 3. * powr<2>(cosl1p2) + 48. * powr<2>(cosp1p2)) * powr<6>(p1) * p2 + (-44. - 54. * powr<2>(cosp1p2) + 62. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (6. - 18. * powr<2>(cosp1p2))) * powr<5>(p1) * powr<2>(p2) + cosp1p2 * (-128. + 46. * powr<2>(cosp1p2) + 28. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (2. - 26. * powr<2>(cosp1p2))) * powr<4>(p1) * powr<3>(p2) + (-33. - 88. * powr<2>(cosp1p2) + 65. * powr<4>(cosp1p2) + 2. * powr<6>(cosp1p2) + powr<2>(cosl1p2) * (14. - 39. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2))) * powr<3>(p1) * powr<4>(p2) + cosp1p2 * (-57. + 8. * powr<2>(cosp1p2) + 13. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (2. - 26. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<5>(p2) + (-6. - 21. * powr<2>(cosp1p2) + 12. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (6. - 18. * powr<2>(cosp1p2))) * p1 * powr<6>(p2) + (-6. + _cse19_k9 - 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<7>(p2)) + powr<3>(l1) * ((-39. + 28. * powr<2>(cosp1p2)) * powr<7>(p1) + cosp1p2 * (-128. + 2. * powr<2>(cosl1p2) + 84. * powr<2>(cosp1p2)) * powr<6>(p1) * p2 + (-72. + (-6. + _cse4_k9) * powr<2>(cosl1p2) - 76. * powr<2>(cosp1p2) + 58. * powr<4>(cosp1p2)) * powr<5>(p1) * powr<2>(p2) + cosp1p2 * (-142. - 3. * powr<4>(cosl1p2) + 24. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (-15. + 4. * powr<2>(cosp1p2))) * powr<4>(p1) * powr<3>(p2) + (-31. - 59. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2) + powr<4>(cosl1p2) * (-7. - 4. * powr<2>(cosp1p2)) + powr<2>(cosl1p2) * (20. - 13. * powr<2>(cosp1p2) + 6. * powr<4>(cosp1p2))) * powr<3>(p1) * powr<4>(p2) + cosp1p2 * (-27. - 14. * powr<4>(cosl1p2) - 21. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (2. + 27. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<5>(p2) + (-6. * powr<4>(cosl1p2) - 15. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (9. + 18. * powr<2>(cosp1p2))) * p1 * powr<6>(p2) + (-3. + 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<7>(p2)) + cosl1p2 * powr<6>(l1) * (3. * powr<3>(p1) * p2 - 3. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) + (-15. + 18. * powr<2>(cosl1p2)) * p1 * powr<3>(p2) + powr<2>(cosp1p2) * (-21. * powr<3>(p1) * p2 + (9. - 6. * powr<2>(cosl1p2)) * p1 * powr<3>(p2)) + cosp1p2 * (-12. * powr<4>(p1) + (-15. + 18. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) + (6. - 6. * powr<2>(cosl1p2)) * powr<4>(p2))) + cosl1p2 * powr<4>(l1) * (16. * powr<5>(p1) * p2 + (-26. + 41. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) - 9. * powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) + (-27. + 23. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * p1 * powr<5>(p2) + powr<3>(cosp1p2) * (-76. * powr<4>(p1) * powr<2>(p2) + (-7. - 4. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<4>(p2)) + powr<2>(cosp1p2) * (-86. * powr<5>(p1) * p2 + (-67. + 35. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) + (11. - 14. * powr<2>(cosl1p2)) * p1 * powr<5>(p2)) + cosp1p2 * (-28. * powr<6>(p1) + (-34. + 27. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<2>(p2) + (-49. + 68. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * powr<2>(p1) * powr<4>(p2) + (6. - 6. * powr<2>(cosl1p2)) * powr<6>(p2))) + cosl1p2 * powr<2>(l1) * p1 * (8. * powr<6>(p1) * p2 + (-2. + 5. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<3>(p2) - 5. * powr<5>(cosp1p2) * powr<3>(p1) * powr<4>(p2) + (-18. + 6. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<5>(p2) + (-6. + 3. * powr<2>(cosl1p2)) * powr<7>(p2) + powr<4>(cosp1p2) * (-53. * powr<4>(p1) * powr<3>(p2) - 26. * powr<2>(p1) * powr<5>(p2)) + powr<3>(cosp1p2) * (-78. * powr<5>(p1) * powr<2>(p2) + (-102. + 6. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<4>(p2) - 18. * p1 * powr<6>(p2)) + cosp1p2 * (-12. * powr<7>(p1) + (-18. + 6. * powr<2>(cosl1p2)) * powr<5>(p1) * powr<2>(p2) + (-34. + 27. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<4>(p2) + (-15. + 18. * powr<2>(cosl1p2)) * p1 * powr<6>(p2)) + powr<2>(cosp1p2) * (-50. * powr<6>(p1) * p2 + (-92. + 16. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<3>(p2) + (-40. + 27. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<5>(p2) - 3. * powr<7>(p2))), fma(powr<2>(cosl1p1), p1 * (powr<4>(cosl1p2) * powr<4>(l1) * powr<3>(p2) * (4. * cosp1p2 * powr<2>(p1) + 4. * p1 * p2 - powr<2>(cosp1p2) * p1 * p2 - cosp1p2 * powr<2>(p2)) + powr<6>(l1) * ((9. + _cse19_k9) * powr<3>(p1) + (21. + _cse19_k9) * cosp1p2 * powr<2>(p1) * p2 + (-3. + 30. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + 15. * cosp1p2 * powr<3>(p2)) + powr<4>(l1) * ((30. + _cse1_k9) * powr<5>(p1) + cosp1p2 * (88. - 4. * powr<2>(cosp1p2)) * powr<4>(p1) * p2 + (14. + 117. * powr<2>(cosp1p2) + 7. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (59. + 77. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) + (-10. + 91. * powr<2>(cosp1p2)) * p1 * powr<4>(p2) + 27. * cosp1p2 * powr<5>(p2)) + powr<3>(p1) * (_cse20_k9 * powr<6>(p1) + (-12. + _cse21_k9) * cosp1p2 * powr<5>(p1) * p2 + (-11. + _cse1_k9 + 13. * powr<4>(cosp1p2)) * powr<4>(p1) * powr<2>(p2) + cosp1p2 * (-28. + _cse22_k9 + 2. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<3>(p2) + (-11. + _cse1_k9 + 13. * powr<4>(cosp1p2)) * powr<2>(p1) * powr<4>(p2) + (-12. + _cse21_k9) * cosp1p2 * p1 * powr<5>(p2) + _cse20_k9 * powr<6>(p2)) + powr<2>(l1) * ((14. + _cse1_k9) * powr<7>(p1) + (54. + _cse18_k9) * cosp1p2 * powr<6>(p1) * p2 + (22. + 88. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2)) * powr<5>(p1) * powr<2>(p2) + cosp1p2 * (76. + 72. * powr<2>(cosp1p2) - powr<4>(cosp1p2)) * powr<4>(p1) * powr<3>(p2) + (1. + 117. * powr<2>(cosp1p2) + 14. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<4>(p2) + cosp1p2 * (32. + 43. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<5>(p2) + (-3. + 30. * powr<2>(cosp1p2)) * p1 * powr<6>(p2) + 6. * cosp1p2 * powr<7>(p2)) + powr<3>(cosl1p2) * powr<3>(l1) * powr<2>(p2) * (2. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) + p1 * p2 * (-25. * powr<2>(l1) - 17. * powr<2>(p1) - 11. * powr<2>(p2)) + powr<2>(cosp1p2) * p1 * p2 * (12. * powr<2>(l1) - 27. * powr<2>(p1) + 12. * powr<2>(p2)) + cosp1p2 * (-16. * powr<4>(p1) - 35. * powr<2>(p1) * powr<2>(p2) + 6. * powr<4>(p2) + powr<2>(l1) * (-27. * powr<2>(p1) + 14. * powr<2>(p2)))) + powr<2>(cosl1p2) * powr<2>(l1) * p2 * (powr<3>(cosp1p2) * powr<2>(p1) * (-21. * powr<2>(l1) + 52. * powr<2>(p1)) * powr<2>(p2) + powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) + powr<2>(cosp1p2) * p1 * p2 * (-6. * powr<4>(l1) + 65. * powr<4>(p1) + 72. * powr<2>(p1) * powr<2>(p2) - 6. * powr<4>(p2) + powr<2>(l1) * (72. * powr<2>(p1) - 59. * powr<2>(p2))) + p1 * p2 * (_cse10_k9 + 6. * powr<4>(l1) - 31. * powr<2>(p1) * powr<2>(p2) - 9. * powr<4>(p2) + powr<2>(l1) * (8. * powr<2>(p1) - 13. * powr<2>(p2))) + cosp1p2 * (18. * powr<6>(p1) + 11. * powr<4>(p1) * powr<2>(p2) - 3. * powr<6>(p2) + powr<4>(l1) * (18. * powr<2>(p1) - 18. * powr<2>(p2)) + powr<2>(l1) * (39. * powr<4>(p1) + 13. * powr<2>(p1) * powr<2>(p2) - 23. * powr<4>(p2)))) + cosl1p2 * l1 * (-2. * powr<5>(cosp1p2) * powr<4>(p1) * powr<4>(p2) + powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) * (10. * powr<2>(l1) - 31. * powr<2>(p1) - 13. * powr<2>(p2)) + powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) * (-powr<4>(l1) - 62. * powr<4>(p1) - 78. * powr<2>(p1) * powr<2>(p2) - 12. * powr<4>(p2) + powr<2>(l1) * (-40. * powr<2>(p1) + 11. * powr<2>(p2))) + powr<2>(cosp1p2) * p1 * p2 * (-3. * powr<6>(l1) - 30. * powr<6>(p1) - 33. * powr<4>(p1) * powr<2>(p2) - 14. * powr<2>(p1) * powr<4>(p2) - 3. * powr<6>(p2) + powr<4>(l1) * (-14. * powr<2>(p1) - 15. * powr<2>(p2)) + powr<2>(l1) * (-33. * powr<4>(p1) - 17. * powr<2>(p1) * powr<2>(p2) - 15. * powr<4>(p2))) + p1 * p2 * (3. * powr<6>(l1) + 21. * powr<6>(p1) + 58. * powr<4>(p1) * powr<2>(p2) + 39. * powr<2>(p1) * powr<4>(p2) + 6. * powr<6>(p2) + powr<4>(l1) * (16. * powr<2>(p1) + 43. * powr<2>(p2)) + powr<2>(l1) * (37. * powr<4>(p1) + 97. * powr<2>(p1) * powr<2>(p2) + 44. * powr<4>(p2))) + cosp1p2 * (-3. * powr<8>(p1) + 50. * powr<6>(p1) * powr<2>(p2) + 86. * powr<4>(p1) * powr<4>(p2) + 21. * powr<2>(p1) * powr<6>(p2) + powr<6>(l1) * (-3. * powr<2>(p1) + 3. * powr<2>(p2)) + powr<4>(l1) * (2. * powr<4>(p1) + 40. * powr<2>(p1) * powr<2>(p2) - 11. * powr<4>(p2)) + powr<2>(l1) * (2. * powr<6>(p1) + 92. * powr<4>(p1) * powr<2>(p2) + 67. * powr<2>(p1) * powr<4>(p2) - 9. * powr<6>(p2))))), 0.))))))))))));
        // clang-format on
      }
      { // subkernel 10
        const auto _cse1_k10 = -12. * powr<4>(l1);
        const auto _cse2_k10 = 14. * powr<2>(p1) * p2;
        const auto _cse3_k10 = 3. * powr<6>(l1);
        const auto _cse4_k10 = -22. * powr<4>(p1);
        const auto _cse5_k10 = 12. * powr<2>(cosp1p2);
        const auto _cse6_k10 = -4. * powr<2>(cosp1p2);
        const auto _cse7_k10 = 4. * powr<2>(cosl1p2);
        const auto _cse8_k10 = 6. * powr<2>(cosl1p2);
        const auto _cse9_k10 = -72. * powr<2>(p1) * powr<2>(p2);
        const auto _cse10_k10 = 48. * powr<6>(p2);
        const auto _cse11_k10 = 24. * powr<2>(p1) * powr<2>(p2);
        const auto _cse12_k10 = -3. * powr<6>(p1);
        const auto _cse13_k10 = -3. * powr<2>(p1);
        const auto _cse14_k10 = 6. * powr<6>(p1);
        const auto _cse15_k10 = -3. * powr<2>(cosp1p2);
        const auto _cse16_k10 = 3. + _cse15_k10;
        const auto _cse17_k10 = -11. * powr<2>(cosp1p2);
        const auto _cse18_k10 = 11. + _cse17_k10;
        const auto _cse19_k10 = 6. * powr<2>(cosp1p2);
        const auto _cse20_k10 = -powr<2>(cosp1p2);
        const auto _cse21_k10 = -2. * powr<2>(cosp1p2);
        const auto _cse22_k10 = 13. * powr<4>(cosp1p2);
        // clang-format off
        _acc += 6. * _den1 * _den10 * _den14 * _den15 * _den17 * _den2 * _den4 * _den6 * _interp22 * _interp28 * _interp30 * fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k), fma(_interp3, _interp4 * (1. + powr<6>(k)), fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.))) * fma(powr<6>(cosl1p2), powr<4>(l1) * (-2. * cosp1p2 * p1 - 2. * p2) * powr<5>(p2), fma(powr<5>(cosl1p1), powr<4>(l1) * powr<4>(p1) * (l1 * (-p1 - cosp1p2 * p2) + cosl1p2 * p2 * (p1 + cosp1p2 * p2)), fma(powr<5>(cosl1p2), powr<3>(l1) * powr<4>(p2) * (_cse2_k10 + 13. * powr<2>(l1) * p2 - 2. * powr<3>(cosp1p2) * p1 * powr<2>(p2) + 15. * powr<3>(p2) + cosp1p2 * (13. * powr<2>(l1) * p1 + 2. * powr<3>(p1) + 28. * p1 * powr<2>(p2)) + powr<2>(cosp1p2) * (powr<2>(p1) * p2 - 2. * powr<3>(p2))), fma(powr<4>(cosl1p1), powr<3>(l1) * powr<3>(p1) * (powr<3>(l1) * (6. * p1 + 6. * cosp1p2 * p2) + cosl1p2 * p2 * (-6. * powr<3>(p1) - 14. * cosp1p2 * powr<2>(p1) * p2 + (-7. + _cse6_k10) * p1 * powr<2>(p2) - 3. * cosp1p2 * powr<3>(p2)) + l1 * (6. * powr<3>(p1) + (14. - powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (11. + (4. + _cse20_k10) * powr<2>(cosl1p2) + 4. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + (7. + _cse7_k10) * cosp1p2 * powr<3>(p2)) + cosl1p2 * powr<2>(l1) * (-14. * p1 * p2 + powr<2>(cosp1p2) * p1 * p2 + cosp1p2 * (powr<2>(p1) - 14. * powr<2>(p2)))), fma(powr<4>(cosl1p2), powr<2>(l1) * powr<3>(p2) * (5. * powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) + powr<3>(cosp1p2) * p1 * powr<2>(p2) * (13. * powr<2>(l1) + 16. * powr<2>(p1) + 26. * powr<2>(p2)) + cosp1p2 * p1 * (_cse1_k10 - 58. * powr<2>(p1) * powr<2>(p2) - 62. * powr<4>(p2) + powr<2>(l1) * (5. * powr<2>(p1) - 65. * powr<2>(p2))) + p2 * (_cse1_k10 + _cse4_k10 - 51. * powr<2>(p1) * powr<2>(p2) - 25. * powr<4>(p2) + powr<2>(l1) * (-35. * powr<2>(p1) - 39. * powr<2>(p2))) + powr<2>(cosp1p2) * (7. * powr<4>(p1) * p2 - 5. * powr<2>(p1) * powr<3>(p2) + 13. * powr<5>(p2) + powr<2>(l1) * (_cse2_k10 + 13. * powr<3>(p2)))), fma(powr<3>(cosl1p2), l1 * powr<2>(p2) * (-2. * powr<5>(cosp1p2) * powr<3>(p1) * powr<4>(p2) + powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) * (-16. * powr<2>(l1) - 13. * powr<2>(p1) - 26. * powr<2>(p2)) + powr<3>(cosp1p2) * p1 * powr<2>(p2) * (_cse1_k10 - 12. * powr<4>(p1) - 52. * powr<2>(p1) * powr<2>(p2) - 36. * powr<4>(p2) + powr<2>(l1) * (-74. * powr<2>(p1) - 52. * powr<2>(p2))) + powr<2>(cosp1p2) * p2 * (_cse12_k10 + 4. * powr<4>(p1) * powr<2>(p2) + 6. * powr<2>(p1) * powr<4>(p2) - 12. * powr<6>(p2) + powr<4>(l1) * (-43. * powr<2>(p1) - 12. * powr<2>(p2)) + powr<2>(l1) * (_cse9_k10 - 77. * powr<4>(p1) - 26. * powr<4>(p2))) + p2 * (_cse14_k10 + _cse3_k10 + 33. * powr<4>(p1) * powr<2>(p2) + 44. * powr<2>(p1) * powr<4>(p2) + 15. * powr<6>(p2) + powr<4>(l1) * (20. * powr<2>(p1) + 10. * powr<2>(p2)) + powr<2>(l1) * (31. * powr<4>(p1) + 58. * powr<2>(p1) * powr<2>(p2) + 24. * powr<4>(p2))) + cosp1p2 * p1 * (_cse10_k10 + _cse3_k10 + 24. * powr<4>(p1) * powr<2>(p2) + 84. * powr<2>(p1) * powr<4>(p2) + powr<4>(l1) * (-21. * powr<2>(p1) + 8. * powr<2>(p2)) + powr<2>(l1) * (_cse11_k10 + _cse4_k10 + 46. * powr<4>(p2)))), fma(powr<3>(cosl1p1), powr<2>(l1) * powr<2>(p1) * (powr<5>(l1) * (-3. * p1 - 3. * cosp1p2 * p2) + cosl1p2 * powr<4>(l1) * (-6. * cosp1p2 * powr<2>(p1) + 18. * p1 * p2 - 6. * powr<2>(cosp1p2) * p1 * p2 + 18. * cosp1p2 * powr<2>(p2)) + powr<3>(l1) * (-7. * powr<3>(p1) + (-23. + 14. * powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (-11. + _cse17_k10 + (-25. + _cse5_k10) * powr<2>(cosl1p2)) * p1 * powr<2>(p2) + (-6. - 27. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) + cosl1p2 * p2 * (3. * powr<5>(p1) + 18. * cosp1p2 * powr<4>(p1) * p2 + (6. + 27. * powr<2>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) + (27. + _cse19_k10) * cosp1p2 * powr<2>(p1) * powr<3>(p2) + (5. + 16. * powr<2>(cosp1p2)) * p1 * powr<4>(p2) + 6. * cosp1p2 * powr<5>(p2)) + l1 * (-3. * powr<5>(p1) + (-18. + _cse8_k10) * cosp1p2 * powr<4>(p1) * p2 + (-11. + (-11. + _cse5_k10) * powr<2>(cosl1p2) - 25. * powr<2>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (-41. + _cse6_k10 + powr<2>(cosl1p2) * (-35. + 2. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<3>(p2) + (-12. - 17. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-17. - 27. * powr<2>(cosp1p2))) * p1 * powr<4>(p2) + (-5. - 16. * powr<2>(cosl1p2)) * cosp1p2 * powr<5>(p2)) + cosl1p2 * powr<2>(l1) * (23. * powr<3>(p1) * p2 - 4. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) + (41. + _cse7_k10) * p1 * powr<3>(p2) + powr<2>(cosp1p2) * (-14. * powr<3>(p1) * p2 + (35. - 2. * powr<2>(cosl1p2)) * p1 * powr<3>(p2)) + cosp1p2 * (-6. * powr<4>(p1) + (68. - 4. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) + (27. + _cse8_k10) * powr<4>(p2)))), fma(cosl1p2, l1 * (2. * powr<6>(cosp1p2) * powr<4>(p1) * powr<5>(p2) + powr<5>(cosp1p2) * powr<3>(p1) * powr<4>(p2) * (2. * powr<2>(l1) + 13. * powr<2>(p1) + 28. * powr<2>(p2)) + powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) * (12. * powr<4>(p1) + 65. * powr<2>(p1) * powr<2>(p2) + 62. * powr<4>(p2) + powr<2>(l1) * (-5. * powr<2>(p1) + 58. * powr<2>(p2))) + powr<3>(p2) * (-3. * powr<6>(l1) - 6. * powr<6>(p1) - 33. * powr<4>(p1) * powr<2>(p2) - 44. * powr<2>(p1) * powr<4>(p2) - 15. * powr<6>(p2) + powr<4>(l1) * (-20. * powr<2>(p1) - 23. * powr<2>(p2)) + powr<2>(l1) * (_cse9_k10 - 31. * powr<4>(p1) - 39. * powr<4>(p2))) + powr<3>(cosp1p2) * p1 * powr<2>(p2) * (_cse10_k10 + 3. * powr<6>(p1) + 8. * powr<4>(p1) * powr<2>(p2) + 46. * powr<2>(p1) * powr<4>(p2) + powr<4>(l1) * (-22. * powr<2>(p1) + 24. * powr<2>(p2)) + powr<2>(l1) * (_cse11_k10 - 21. * powr<4>(p1) + 84. * powr<4>(p2))) + cosp1p2 * p1 * (-6. * powr<6>(p1) * powr<2>(p2) - 57. * powr<4>(p1) * powr<4>(p2) - 128. * powr<2>(p1) * powr<6>(p2) - 63. * powr<8>(p2) + powr<6>(l1) * (_cse13_k10 - 6. * powr<2>(p2)) + powr<4>(l1) * (-8. * powr<4>(p1) - 27. * powr<2>(p1) * powr<2>(p2) - 57. * powr<4>(p2)) + powr<2>(l1) * (_cse12_k10 - 27. * powr<4>(p1) * powr<2>(p2) - 142. * powr<2>(p1) * powr<4>(p2) - 128. * powr<6>(p2))) + powr<2>(cosp1p2) * p2 * (-6. * powr<6>(l1) * powr<2>(p1) - 21. * powr<6>(p1) * powr<2>(p2) - 88. * powr<4>(p1) * powr<4>(p2) - 54. * powr<2>(p1) * powr<6>(p2) + 12. * powr<8>(p2) + powr<4>(l1) * (-27. * powr<4>(p1) - 32. * powr<2>(p1) * powr<2>(p2) + 12. * powr<4>(p2)) + powr<2>(l1) * (-15. * powr<6>(p1) - 59. * powr<4>(p1) * powr<2>(p2) - 76. * powr<2>(p1) * powr<4>(p2) + 28. * powr<6>(p2)))), fma(powr<2>(cosl1p2), p2 * (powr<5>(cosp1p2) * powr<3>(p1) * powr<4>(p2) * (-powr<2>(l1) + 2. * powr<2>(p2)) + powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) * (7. * powr<4>(l1) + 13. * powr<2>(p1) * powr<2>(p2) + 13. * powr<4>(p2) + powr<2>(l1) * (14. * powr<2>(p1) - 5. * powr<2>(p2))) + powr<3>(cosp1p2) * p1 * powr<2>(p2) * (_cse3_k10 + 12. * powr<4>(p1) * powr<2>(p2) + 26. * powr<2>(p1) * powr<4>(p2) + 12. * powr<6>(p2) + powr<4>(l1) * (77. * powr<2>(p1) - 4. * powr<2>(p2)) + powr<2>(l1) * (43. * powr<4>(p1) + 72. * powr<2>(p1) * powr<2>(p2) - 6. * powr<4>(p2))) + powr<2>(cosp1p2) * p2 * (3. * powr<6>(p1) * powr<2>(p2) - 2. * powr<4>(p1) * powr<4>(p2) - 2. * powr<2>(p1) * powr<6>(p2) + 3. * powr<8>(p2) + powr<6>(l1) * (30. * powr<2>(p1) + 3. * powr<2>(p2)) + powr<4>(l1) * (91. * powr<4>(p1) + 117. * powr<2>(p1) * powr<2>(p2) - 2. * powr<4>(p2)) + powr<2>(l1) * (30. * powr<6>(p1) + 117. * powr<4>(p1) * powr<2>(p2) + 88. * powr<2>(p1) * powr<4>(p2) - 2. * powr<6>(p2))) + p2 * (-3. * powr<6>(p1) * powr<2>(p2) - 11. * powr<4>(p1) * powr<4>(p2) - 11. * powr<2>(p1) * powr<6>(p2) - 3. * powr<8>(p2) + powr<6>(l1) * (_cse13_k10 + 9. * powr<2>(p2)) + powr<4>(l1) * (-10. * powr<4>(p1) + 14. * powr<2>(p1) * powr<2>(p2) + 30. * powr<4>(p2)) + powr<2>(l1) * (_cse12_k10 + powr<4>(p1) * powr<2>(p2) + 22. * powr<2>(p1) * powr<4>(p2) + 14. * powr<6>(p2))) + cosp1p2 * p1 * (-12. * powr<4>(p1) * powr<4>(p2) - 28. * powr<2>(p1) * powr<6>(p2) - 12. * powr<8>(p2) + powr<6>(l1) * (15. * powr<2>(p1) + 21. * powr<2>(p2)) + powr<4>(l1) * (27. * powr<4>(p1) + 59. * powr<2>(p1) * powr<2>(p2) + 88. * powr<4>(p2)) + powr<2>(l1) * (_cse14_k10 + 32. * powr<4>(p1) * powr<2>(p2) + 76. * powr<2>(p1) * powr<4>(p2) + 54. * powr<6>(p2)))), fma(powr<2>(p2), powr<6>(l1) * (_cse16_k10 * powr<2>(p1) + cosp1p2 * (6. - 6. * powr<2>(cosp1p2)) * p1 * p2 + _cse16_k10 * powr<2>(p2)) + powr<4>(l1) * ((10. - 10. * powr<2>(cosp1p2)) * powr<4>(p1) + cosp1p2 * (31. - 31. * powr<2>(cosp1p2)) * powr<3>(p1) * p2 + (21. + powr<2>(cosp1p2) - 22. * powr<4>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) + cosp1p2 * (33. - 33. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) + _cse18_k10 * powr<4>(p2)) + powr<2>(l1) * (_cse16_k10 * powr<6>(p1) + cosp1p2 * (20. - 20. * powr<2>(cosp1p2)) * powr<5>(p1) * p2 + (21. + 14. * powr<2>(cosp1p2) - 35. * powr<4>(cosp1p2)) * powr<4>(p1) * powr<2>(p2) + cosp1p2 * (72. - 58. * powr<2>(cosp1p2) - 14. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<3>(p2) + (29. + 22. * powr<2>(cosp1p2) - 51. * powr<4>(cosp1p2)) * powr<2>(p1) * powr<4>(p2) + cosp1p2 * (44. - 44. * powr<2>(cosp1p2)) * p1 * powr<5>(p2) + _cse18_k10 * powr<6>(p2)) + p2 * (3. * powr<6>(p1) * p2 + 11. * powr<4>(p1) * powr<3>(p2) - 2. * powr<6>(cosp1p2) * powr<4>(p1) * powr<3>(p2) + 11. * powr<2>(p1) * powr<5>(p2) + 3. * powr<7>(p2) + powr<5>(cosp1p2) * (-13. * powr<5>(p1) * powr<2>(p2) - 15. * powr<3>(p1) * powr<4>(p2)) + powr<4>(cosp1p2) * (-12. * powr<6>(p1) * p2 - 39. * powr<4>(p1) * powr<3>(p2) - 25. * powr<2>(p1) * powr<5>(p2)) + powr<3>(cosp1p2) * (-3. * powr<7>(p1) - 10. * powr<5>(p1) * powr<2>(p2) - 24. * powr<3>(p1) * powr<4>(p2) - 15. * p1 * powr<6>(p2)) + cosp1p2 * (3. * powr<7>(p1) + 23. * powr<5>(p1) * powr<2>(p2) + 39. * powr<3>(p1) * powr<4>(p2) + 15. * p1 * powr<6>(p2)) + powr<2>(cosp1p2) * (9. * powr<6>(p1) * p2 + 30. * powr<4>(p1) * powr<3>(p2) + 14. * powr<2>(p1) * powr<5>(p2) - 3. * powr<7>(p2))), fma(cosl1p1, powr<7>(l1) * (3. * powr<3>(p1) + (6. + 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (_cse19_k10 + _cse16_k10 * powr<2>(cosl1p2)) * p1 * powr<2>(p2) + (3. - 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) + powr<5>(l1) * (8. * powr<5>(p1) + (27. - 11. * powr<2>(cosl1p2)) * cosp1p2 * powr<4>(p1) * p2 + (5. + 44. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (43. - 15. * powr<2>(cosp1p2))) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (18. + (40. + _cse20_k10) * powr<2>(cosl1p2) + 26. * powr<4>(cosl1p2) + 34. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) + (-6. + 39. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (16. - 14. * powr<2>(cosp1p2)) + powr<4>(cosl1p2) * (26. - 13. * powr<2>(cosp1p2))) * p1 * powr<4>(p2) + (11. + 2. * powr<2>(cosl1p2) - 13. * powr<4>(cosl1p2)) * cosp1p2 * powr<5>(p2)) + cosl1p2 * p1 * powr<3>(p2) * ((-3. + 3. * powr<2>(cosp1p2)) * powr<6>(p1) + (-12. + _cse5_k10) * cosp1p2 * powr<5>(p1) * p2 + (-11. + _cse21_k10 + _cse22_k10) * powr<4>(p1) * powr<2>(p2) + cosp1p2 * (-28. + 26. * powr<2>(cosp1p2) + 2. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<3>(p2) + (-11. + _cse21_k10 + _cse22_k10) * powr<2>(p1) * powr<4>(p2) + (-12. + _cse5_k10) * cosp1p2 * p1 * powr<5>(p2) + (-3. + 3. * powr<2>(cosp1p2)) * powr<6>(p2)) + cosl1p2 * powr<2>(l1) * p2 * ((-6. + _cse15_k10) * powr<7>(p1) + cosp1p2 * (-15. - 3. * powr<2>(cosl1p2) - 18. * powr<2>(cosp1p2)) * powr<6>(p1) * p2 + (-18. - 40. * powr<2>(cosp1p2) - 26. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (-34. + powr<2>(cosp1p2))) * powr<5>(p1) * powr<2>(p2) + cosp1p2 * (-34. - 102. * powr<2>(cosp1p2) - 5. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (-76. + 16. * powr<2>(cosp1p2))) * powr<4>(p1) * powr<3>(p2) + (-2. - 92. * powr<2>(cosp1p2) - 53. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (-88. + 40. * powr<2>(cosp1p2) + 6. * powr<4>(cosp1p2))) * powr<3>(p1) * powr<4>(p2) + cosp1p2 * (-18. - 78. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-78. + 72. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<5>(p2) + (8. - 50. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-44. + 62. * powr<2>(cosp1p2))) * p1 * powr<6>(p2) + (-12. + 12. * powr<2>(cosl1p2)) * cosp1p2 * powr<7>(p2)) + powr<3>(l1) * (3. * powr<7>(p1) + (15. - 9. * powr<2>(cosl1p2)) * cosp1p2 * powr<6>(p1) * p2 + (5. + 43. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (44. - 15. * powr<2>(cosp1p2))) * powr<5>(p1) * powr<2>(p2) + cosp1p2 * (26. + 9. * powr<4>(cosl1p2) + 69. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (67. + 11. * powr<2>(cosp1p2))) * powr<4>(p1) * powr<3>(p2) + (-14. + 97. * powr<2>(cosp1p2) + 33. * powr<4>(cosp1p2) + powr<4>(cosl1p2) * (33. + 10. * powr<2>(cosp1p2)) + powr<2>(cosl1p2) * (97. - 17. * powr<2>(cosp1p2) + 10. * powr<4>(cosp1p2))) * powr<3>(p1) * powr<4>(p2) + cosp1p2 * (2. + 88. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (92. - 40. * powr<2>(cosp1p2)) + powr<4>(cosl1p2) * (53. - 6. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<5>(p2) + (-14. + 58. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (37. - 33. * powr<2>(cosp1p2)) + powr<4>(cosl1p2) * (31. - 31. * powr<2>(cosp1p2))) * p1 * powr<6>(p2) + (11. + 2. * powr<2>(cosl1p2) - 13. * powr<4>(cosl1p2)) * cosp1p2 * powr<7>(p2)) + cosl1p2 * powr<6>(l1) * (-15. * powr<3>(p1) * p2 - 3. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) + (3. - 18. * powr<2>(cosl1p2)) * p1 * powr<3>(p2) + powr<2>(cosp1p2) * (9. * powr<3>(p1) * p2 + (-21. + 12. * powr<2>(cosl1p2)) * p1 * powr<3>(p2)) + cosp1p2 * (6. * powr<4>(p1) + (-15. - 18. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<2>(p2) + (-12. + 12. * powr<2>(cosl1p2)) * powr<4>(p2))) + cosl1p2 * powr<4>(l1) * (-27. * powr<5>(p1) * p2 + (-26. - 69. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) - 9. * powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) + (16. - 65. * powr<2>(cosl1p2) - 5. * powr<4>(cosl1p2)) * p1 * powr<5>(p2) + powr<3>(cosp1p2) * (-7. * powr<4>(p1) * powr<2>(p2) + (-76. + 16. * powr<2>(cosl1p2)) * powr<2>(p1) * powr<4>(p2)) + powr<2>(cosp1p2) * (11. * powr<5>(p1) * p2 + (-67. - 11. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) + (-86. + 78. * powr<2>(cosl1p2) + 2. * powr<4>(cosl1p2)) * p1 * powr<5>(p2)) + cosp1p2 * (_cse14_k10 + (-49. - 7. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<2>(p2) + (-34. - 102. * powr<2>(cosl1p2) - 5. * powr<4>(cosl1p2)) * powr<2>(p1) * powr<4>(p2) + (-28. + 26. * powr<2>(cosl1p2) + 2. * powr<4>(cosl1p2)) * powr<6>(p2))) + l1 * powr<2>(p2) * (-6. * powr<5>(p1) * powr<2>(p2) + 5. * powr<5>(cosp1p2) * powr<4>(p1) * powr<3>(p2) - 14. * powr<3>(p1) * powr<4>(p2) - 6. * p1 * powr<6>(p2) + powr<4>(cosp1p2) * (26. * powr<5>(p1) * powr<2>(p2) + 31. * powr<3>(p1) * powr<4>(p2)) + powr<3>(cosp1p2) * (18. * powr<6>(p1) * p2 + 65. * powr<4>(p1) * powr<3>(p2) + 44. * powr<2>(p1) * powr<5>(p2)) + powr<2>(cosp1p2) * (3. * powr<7>(p1) + 16. * powr<5>(p1) * powr<2>(p2) + 37. * powr<3>(p1) * powr<4>(p2) + 21. * p1 * powr<6>(p2)) + cosp1p2 * (-3. * powr<6>(p1) * p2 - 16. * powr<4>(p1) * powr<3>(p2) - 8. * powr<2>(p1) * powr<5>(p2) + 3. * powr<7>(p2)) + powr<2>(cosl1p2) * ((6. + _cse15_k10) * powr<7>(p1) + cosp1p2 * (21. - 12. * powr<2>(cosp1p2)) * powr<6>(p1) * p2 + (39. - 14. * powr<2>(cosp1p2) - 13. * powr<4>(cosp1p2)) * powr<5>(p1) * powr<2>(p2) + cosp1p2 * (86. - 78. * powr<2>(cosp1p2) - 2. * powr<4>(cosp1p2)) * powr<4>(p1) * powr<3>(p2) + (58. - 33. * powr<2>(cosp1p2) - 31. * powr<4>(cosp1p2)) * powr<3>(p1) * powr<4>(p2) + cosp1p2 * (50. - 62. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<5>(p2) + (21. - 30. * powr<2>(cosp1p2)) * p1 * powr<6>(p2) - 3. * cosp1p2 * powr<7>(p2))), fma(powr<2>(cosl1p1), l1 * p1 * (powr<4>(cosl1p2) * powr<3>(l1) * powr<3>(p2) * (-6. * cosp1p2 * powr<2>(p1) - 2. * p1 * p2 + powr<2>(cosp1p2) * p1 * p2 + 5. * cosp1p2 * powr<2>(p2)) + powr<5>(l1) * (-6. * powr<3>(p1) - 9. * cosp1p2 * powr<2>(p1) * p2 - 9. * powr<2>(cosp1p2) * p1 * powr<2>(p2) - 6. * cosp1p2 * powr<3>(p2)) + powr<3>(l1) * (-6. * powr<5>(p1) - 11. * cosp1p2 * powr<4>(p1) * p2 + (-12. - 13. * powr<2>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (-20. - 14. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) + (3. - 31. * powr<2>(cosp1p2)) * p1 * powr<4>(p2) - 14. * cosp1p2 * powr<5>(p2)) + l1 * p2 * (3. * powr<3>(p1) * powr<3>(p2) - 2. * powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) + 5. * p1 * powr<5>(p2) + powr<3>(cosp1p2) * (-powr<4>(p1) * powr<2>(p2) - 6. * powr<2>(p1) * powr<4>(p2)) + powr<2>(cosp1p2) * (6. * powr<5>(p1) * p2 + 8. * powr<3>(p1) * powr<3>(p2) - 14. * p1 * powr<5>(p2)) + cosp1p2 * (3. * powr<6>(p1) + 10. * powr<4>(p1) * powr<2>(p2) + 6. * powr<2>(p1) * powr<4>(p2) - 6. * powr<6>(p2))) + powr<3>(cosl1p2) * powr<2>(l1) * powr<2>(p2) * (-2. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) + p1 * p2 * (powr<2>(l1) + 14. * powr<2>(p1) + 6. * powr<2>(p2)) + powr<2>(cosp1p2) * (21. * powr<3>(p1) * p2 - 52. * p1 * powr<3>(p2)) + cosp1p2 * (13. * powr<4>(p1) + 4. * powr<2>(p1) * powr<2>(p2) - 26. * powr<4>(p2) + powr<2>(l1) * (27. * powr<2>(p1) - 26. * powr<2>(p2)))) + powr<2>(cosl1p2) * l1 * p2 * (powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) + powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) * (-21. * powr<2>(l1) + 52. * powr<2>(p2)) + p1 * p2 * (6. * powr<4>(l1) - 9. * powr<4>(p1) - 31. * powr<2>(p1) * powr<2>(p2) - 14. * powr<4>(p2) + powr<2>(l1) * (-13. * powr<2>(p1) + 8. * powr<2>(p2))) + powr<2>(cosp1p2) * p1 * p2 * (-6. * powr<4>(l1) - 6. * powr<4>(p1) + 72. * powr<2>(p1) * powr<2>(p2) + 65. * powr<4>(p2) + powr<2>(l1) * (-59. * powr<2>(p1) + 72. * powr<2>(p2))) + cosp1p2 * (_cse12_k10 + 11. * powr<2>(p1) * powr<4>(p2) + 18. * powr<6>(p2) + powr<4>(l1) * (-18. * powr<2>(p1) + 18. * powr<2>(p2)) + powr<2>(l1) * (-23. * powr<4>(p1) + 13. * powr<2>(p1) * powr<2>(p2) + 39. * powr<4>(p2)))) + cosl1p2 * (powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) * (6. * powr<2>(l1) - 5. * powr<2>(p2)) + powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) * (13. * powr<4>(l1) - 26. * powr<2>(p1) * powr<2>(p2) - 26. * powr<4>(p2) + powr<2>(l1) * (27. * powr<2>(p1) + 4. * powr<2>(p2))) + powr<2>(cosp1p2) * p1 * p2 * (_cse3_k10 + 23. * powr<4>(l1) * powr<2>(p1) - 18. * powr<4>(p1) * powr<2>(p2) - 39. * powr<2>(p1) * powr<4>(p2) - 18. * powr<6>(p2) + powr<2>(l1) * (18. * powr<4>(p1) - 13. * powr<2>(p1) * powr<2>(p2) - 11. * powr<4>(p2))) + p1 * p2 * (-3. * powr<6>(l1) + 6. * powr<4>(p1) * powr<2>(p2) + 14. * powr<2>(p1) * powr<4>(p2) + 6. * powr<6>(p2) + powr<4>(l1) * (11. * powr<2>(p1) - 10. * powr<2>(p2)) + powr<2>(l1) * (9. * powr<4>(p1) + 20. * powr<2>(p1) * powr<2>(p2) - 6. * powr<4>(p2))) + cosp1p2 * (-3. * powr<6>(p1) * powr<2>(p2) + 2. * powr<4>(p1) * powr<4>(p2) + 2. * powr<2>(p1) * powr<6>(p2) - 3. * powr<8>(p2) + powr<6>(l1) * (3. * powr<2>(p1) - 3. * powr<2>(p2)) + powr<4>(l1) * (7. * powr<4>(p1) + 2. * powr<2>(p1) * powr<2>(p2) + 2. * powr<4>(p2)) + powr<2>(l1) * (3. * powr<6>(p1) + 2. * powr<4>(p1) * powr<2>(p2) - 15. * powr<2>(p1) * powr<4>(p2) + 2. * powr<6>(p2))))), 0.))))))))))));
        // clang-format on
      }
      { // subkernel 11
        const auto _den7 = powr<-1>(powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2));
        const auto _den12 =
            powr<-1>(_interp4 * _interp9 + _interp10 * (powr<2>(l1) + 2. * cosl1p2 * l1 * p2 + powr<2>(p2)));
        const auto _cse1_k11 = -6. * powr<3>(p1);
        const auto _cse2_k11 = 10. * powr<2>(cosp1p2);
        const auto _cse3_k11 = -10. + _cse2_k11;
        const auto _cse4_k11 = -6. * powr<2>(cosp1p2);
        const auto _cse5_k11 = 6. + _cse4_k11;
        const auto _cse6_k11 = -3. * powr<2>(cosp1p2);
        const auto _cse7_k11 = 3. + _cse6_k11;
        const auto _cse8_k11 = 3. * powr<2>(cosp1p2);
        const auto _cse9_k11 = -3. + _cse8_k11;
        const auto _cse10_k11 = -2. * powr<2>(cosp1p2);
        const auto _cse11_k11 = 2. + _cse10_k11;
        const auto _cse12_k11 = -5. * p1 * powr<2>(p2);
        const auto _cse13_k11 = -7. * powr<2>(cosp1p2);
        const auto _cse14_k11 = 3. * powr<5>(p1);
        const auto _cse15_k11 = 12. * powr<2>(cosp1p2);
        const auto _cse16_k11 = -5. * powr<2>(cosp1p2);
        const auto _cse17_k11 = 11. * powr<2>(p2);
        // clang-format off
        _acc += -6. * _den1 * _den12 * _den17 * _den2 * _den4 * _den5 * _den7 * _den9 * _interp12 * _interp14 * _interp16 * fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k), fma(_interp3, _interp4 * (1. + powr<6>(k)), fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.))) * fma(powr<5>(cosl1p2), powr<5>(l1) * (-cosp1p2 * p1 - p2) * powr<4>(p2), fma(powr<5>(cosl1p1), powr<4>(l1) * powr<4>(p1) * (l1 + cosl1p2 * p2) * (p1 + cosp1p2 * p2), fma(powr<4>(cosl1p2), powr<4>(l1) * powr<3>(p2) * (-6. * powr<2>(l1) * p2 - 9. * powr<2>(p1) * p2 - 6. * powr<3>(p2) + cosp1p2 * (-6. * powr<2>(l1) * p1 - 5. * powr<3>(p1) - 10. * p1 * powr<2>(p2))), fma(powr<4>(cosl1p1), powr<3>(l1) * powr<3>(p1) * (powr<3>(l1) * (-6. * p1 - 6. * cosp1p2 * p2) + cosl1p2 * powr<2>(l1) * (-cosp1p2 * powr<2>(p1) - 10. * p1 * p2 - powr<2>(cosp1p2) * p1 * p2 - 10. * cosp1p2 * powr<2>(p2)) + l1 * (_cse1_k11 + (-10. - powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (-9. - powr<2>(cosl1p2) * powr<2>(cosp1p2)) * p1 * powr<2>(p2) - 5. * cosp1p2 * powr<3>(p2)) + cosl1p2 * p2 * (_cse12_k11 + _cse1_k11 - 10. * cosp1p2 * powr<2>(p1) * p2 - cosp1p2 * powr<3>(p2))), fma(powr<2>(p1), powr<2>(p2) * (_cse9_k11 * powr<6>(l1) + powr<4>(l1) * (_cse3_k11 * powr<2>(p1) + cosp1p2 * (-9. + 9. * powr<2>(cosp1p2)) * p1 * p2 + _cse3_k11 * powr<2>(p2)) + cosp1p2 * p1 * p2 * (_cse7_k11 * powr<4>(p1) + _cse5_k11 * cosp1p2 * powr<3>(p1) * p2 + (8. + _cse13_k11 - powr<4>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) + _cse5_k11 * cosp1p2 * p1 * powr<3>(p2) + _cse7_k11 * powr<4>(p2)) + powr<2>(l1) * (_cse9_k11 * powr<4>(p1) + _cse11_k11 * cosp1p2 * powr<3>(p1) * p2 + (-10. + 19. * powr<2>(cosp1p2) - 9. * powr<4>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) + _cse11_k11 * cosp1p2 * p1 * powr<3>(p2) + _cse9_k11 * powr<4>(p2))), fma(powr<3>(cosl1p2), powr<3>(l1) * powr<2>(p2) * (powr<2>(cosp1p2) * powr<2>(p1) * (_cse17_k11 + 7. * powr<2>(l1) + 16. * powr<2>(p1)) * p2 + 2. * powr<3>(cosp1p2) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (-3. * powr<4>(l1) * p1 + 10. * powr<5>(p1) + powr<2>(l1) * (_cse12_k11 + 3. * powr<3>(p1)) + 19. * powr<3>(p1) * powr<2>(p2)) + p2 * (-3. * powr<4>(l1) + 9. * powr<4>(p1) - 2. * powr<2>(p1) * powr<2>(p2) - 3. * powr<4>(p2) + powr<2>(l1) * (-2. * powr<2>(p1) - 7. * powr<2>(p2)))), fma(powr<3>(cosl1p1), powr<2>(l1) * powr<2>(p1) * (powr<5>(l1) * (3. * p1 + 3. * cosp1p2 * p2) + cosl1p2 * cosp1p2 * powr<4>(l1) * p1 * (6. * p1 + 6. * cosp1p2 * p2) + powr<3>(l1) * (7. * powr<3>(p1) + (5. + 10. * powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (2. + _cse13_k11 + (-11. + _cse15_k11) * powr<2>(cosl1p2)) * p1 * powr<2>(p2) + (-3. - 9. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) + cosl1p2 * p2 * (_cse14_k11 + (-3. - 9. * powr<2>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) - 21. * cosp1p2 * powr<2>(p1) * powr<3>(p2) + (-10. + _cse16_k11) * p1 * powr<4>(p2) - 3. * cosp1p2 * powr<5>(p2)) + l1 * (_cse14_k11 + 6. * powr<2>(cosl1p2) * cosp1p2 * powr<4>(p1) * p2 + (2. + (-7. + _cse15_k11) * powr<2>(cosl1p2) - 11. * powr<2>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (-19. + _cse10_k11 + powr<2>(cosl1p2) * (-19. + 2. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<3>(p2) + (-9. + (-16. + _cse16_k11) * powr<2>(cosl1p2) - 16. * powr<2>(cosp1p2)) * p1 * powr<4>(p2) + (-10. - 5. * powr<2>(cosl1p2)) * cosp1p2 * powr<5>(p2)) + cosl1p2 * powr<2>(l1) * (5. * powr<3>(p1) * p2 + (-19. - 2. * powr<2>(cosl1p2)) * p1 * powr<3>(p2) + powr<2>(cosp1p2) * (10. * powr<3>(p1) * p2 + (-19. + 2. * powr<2>(cosl1p2)) * p1 * powr<3>(p2)) + cosp1p2 * (6. * powr<4>(p1) - 8. * powr<2>(p1) * powr<2>(p2) - 21. * powr<4>(p2)))), fma(powr<2>(cosl1p2), powr<2>(l1) * p2 * (powr<3>(cosp1p2) * powr<3>(p1) * (_cse17_k11 + 16. * powr<2>(l1) + 7. * powr<2>(p1)) * powr<2>(p2) + powr<2>(cosp1p2) * powr<2>(p1) * p2 * (15. * powr<4>(l1) + 15. * powr<4>(p1) + 41. * powr<2>(p1) * powr<2>(p2) + 12. * powr<4>(p2) + powr<2>(l1) * (46. * powr<2>(p1) + 41. * powr<2>(p2))) + p2 * (3. * powr<6>(p1) + 10. * powr<4>(p1) * powr<2>(p2) + 3. * powr<2>(p1) * powr<4>(p2) + powr<4>(l1) * (3. * powr<2>(p1) + 6. * powr<2>(p2)) + powr<2>(l1) * (10. * powr<4>(p1) + 19. * powr<2>(p1) * powr<2>(p2) + 6. * powr<4>(p2))) + cosp1p2 * p1 * (6. * powr<6>(p1) + 32. * powr<4>(p1) * powr<2>(p2) + 25. * powr<2>(p1) * powr<4>(p2) + 3. * powr<6>(p2) + powr<4>(l1) * (9. * powr<2>(p1) + 15. * powr<2>(p2)) + powr<2>(l1) * (21. * powr<4>(p1) + 60. * powr<2>(p1) * powr<2>(p2) + 25. * powr<4>(p2)))), fma(cosl1p2, l1 * (-powr<5>(cosp1p2) * powr<5>(p1) * powr<4>(p2) + powr<4>(cosp1p2) * powr<4>(p1) * powr<3>(p2) * (-5. * powr<2>(l1) - 6. * powr<2>(p1) - 10. * powr<2>(p2)) + powr<2>(l1) * powr<3>(p2) * (3. * powr<4>(l1) - 9. * powr<4>(p1) + 2. * powr<2>(p1) * powr<2>(p2) + 3. * powr<4>(p2) + powr<2>(l1) * (2. * powr<2>(p1) + 8. * powr<2>(p2))) + powr<3>(cosp1p2) * powr<3>(p1) * powr<2>(p2) * (10. * powr<4>(l1) - 3. * powr<4>(p1) - 5. * powr<2>(p1) * powr<2>(p2) + powr<2>(l1) * (3. * powr<2>(p1) + 19. * powr<2>(p2))) + powr<2>(cosp1p2) * powr<2>(p1) * p2 * (6. * powr<6>(l1) + 15. * powr<4>(p1) * powr<2>(p2) + 25. * powr<2>(p1) * powr<4>(p2) + 3. * powr<6>(p2) + powr<4>(l1) * (21. * powr<2>(p1) + 32. * powr<2>(p2)) + powr<2>(l1) * (9. * powr<4>(p1) + 60. * powr<2>(p1) * powr<2>(p2) + 25. * powr<4>(p2))) + cosp1p2 * p1 * (6. * powr<6>(p1) * powr<2>(p2) + 21. * powr<4>(p1) * powr<4>(p2) + 9. * powr<2>(p1) * powr<6>(p2) + powr<6>(l1) * (3. * powr<2>(p1) + 6. * powr<2>(p2)) + powr<4>(l1) * (8. * powr<4>(p1) + 24. * powr<2>(p1) * powr<2>(p2) + 21. * powr<4>(p2)) + powr<2>(l1) * (3. * powr<6>(p1) + 24. * powr<4>(p1) * powr<2>(p2) + 27. * powr<2>(p1) * powr<4>(p2) + 9. * powr<6>(p2)))), fma(cosl1p1, powr<7>(l1) * (-3. * powr<3>(p1) + (-6. - 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<2>(p1) * p2 + (_cse4_k11 + _cse9_k11 * powr<2>(cosl1p2)) * p1 * powr<2>(p2) + (-3. + 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<3>(p2)) + cosl1p2 * powr<3>(p1) * powr<3>(p2) * (_cse9_k11 * powr<4>(p1) + cosp1p2 * (-6. + 6. * powr<2>(cosp1p2)) * powr<3>(p1) * p2 + (-8. + 7. * powr<2>(cosp1p2) + powr<4>(cosp1p2)) * powr<2>(p1) * powr<2>(p2) + cosp1p2 * (-6. + 6. * powr<2>(cosp1p2)) * p1 * powr<3>(p2) + _cse9_k11 * powr<4>(p2)) + powr<5>(l1) * (-8. * powr<5>(p1) + (-21. - 25. * powr<2>(cosl1p2)) * cosp1p2 * powr<4>(p1) * p2 + (-2. - 32. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-25. - 39. * powr<2>(cosp1p2))) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (-24. + (-61. + _cse16_k11) * powr<2>(cosl1p2) + 10. * powr<4>(cosl1p2) - 10. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) + (-21. * powr<2>(cosp1p2) + powr<4>(cosl1p2) * (10. + powr<2>(cosp1p2)) + powr<2>(cosl1p2) * (-25. + 5. * powr<2>(cosp1p2))) * p1 * powr<4>(p2) + (-8. + 7. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * cosp1p2 * powr<5>(p2)) + cosl1p2 * powr<2>(l1) * p1 * p2 * ((-6. + _cse6_k11) * powr<6>(p1) + (-33. - 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<5>(p1) * p2 + (-24. + (-10. + _cse16_k11) * powr<2>(cosl1p2) - 61. * powr<2>(cosp1p2) + 10. * powr<4>(cosp1p2)) * powr<4>(p1) * powr<2>(p2) + cosp1p2 * (-95. - 21. * powr<2>(cosl1p2) - 8. * powr<2>(cosp1p2) + powr<4>(cosp1p2)) * powr<3>(p1) * powr<3>(p2) + (-24. - 61. * powr<2>(cosp1p2) + 10. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (-3. - 9. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<4>(p2) - 33. * cosp1p2 * p1 * powr<5>(p2) + (-6. + _cse6_k11 + 3. * powr<2>(cosl1p2)) * powr<6>(p2)) + powr<3>(l1) * (-3. * powr<7>(p1) + (-9. - 15. * powr<2>(cosl1p2)) * cosp1p2 * powr<6>(p1) * p2 + (-2. - 25. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-32. - 39. * powr<2>(cosp1p2))) * powr<5>(p1) * powr<2>(p2) + cosp1p2 * (-27. + powr<4>(cosl1p2) - 19. * powr<2>(cosp1p2) + powr<2>(cosl1p2) * (-126. - 19. * powr<2>(cosp1p2))) * powr<4>(p1) * powr<3>(p2) + (9. + 5. * powr<4>(cosl1p2) - 60. * powr<2>(cosp1p2) + 5. * powr<4>(cosp1p2) + powr<2>(cosl1p2) * (-60. - 81. * powr<2>(cosp1p2))) * powr<3>(p1) * powr<4>(p2) + cosp1p2 * (-24. + _cse6_k11 + 10. * powr<4>(cosl1p2) + powr<2>(cosl1p2) * (-61. - 9. * powr<2>(cosp1p2))) * powr<2>(p1) * powr<5>(p2) + (-15. * powr<2>(cosl1p2) + 6. * powr<4>(cosl1p2) - 9. * powr<2>(cosp1p2)) * p1 * powr<6>(p2) + (-3. + 3. * powr<2>(cosl1p2)) * cosp1p2 * powr<7>(p2)) + cosl1p2 * powr<6>(l1) * (-9. * powr<3>(p1) * p2 - 3. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) - 9. * p1 * powr<3>(p2) + powr<2>(cosp1p2) * (-15. * powr<3>(p1) * p2 + (-15. + 6. * powr<2>(cosl1p2)) * p1 * powr<3>(p2)) + cosp1p2 * (-6. * powr<4>(p1) - 33. * powr<2>(p1) * powr<2>(p2) + (-6. + 6. * powr<2>(cosl1p2)) * powr<4>(p2))) + cosl1p2 * powr<4>(l1) * (-21. * powr<5>(p1) * p2 + (-27. - 19. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) + powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) + (-21. + 5. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * p1 * powr<5>(p2) + powr<3>(cosp1p2) * (-21. * powr<4>(p1) * powr<2>(p2) - 21. * powr<2>(p1) * powr<4>(p2)) + powr<2>(cosp1p2) * (-25. * powr<5>(p1) * p2 + (-126. - 19. * powr<2>(cosl1p2)) * powr<3>(p1) * powr<3>(p2) + (-25. + 10. * powr<2>(cosl1p2)) * p1 * powr<5>(p2)) + cosp1p2 * (-6. * powr<6>(p1) + (-95. - 21. * powr<2>(cosl1p2)) * powr<4>(p1) * powr<2>(p2) + (-95. - 8. * powr<2>(cosl1p2) + powr<4>(cosl1p2)) * powr<2>(p1) * powr<4>(p2) + (-6. + 6. * powr<2>(cosl1p2)) * powr<6>(p2))) + l1 * powr<2>(p1) * powr<2>(p2) * (powr<2>(cosl1p2) * ((-6. + _cse8_k11) * powr<5>(p1) + cosp1p2 * (-15. + 6. * powr<2>(cosp1p2)) * powr<4>(p1) * p2 + (-21. + 5. * powr<2>(cosp1p2) + powr<4>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) + (-25. + _cse2_k11) * cosp1p2 * powr<2>(p1) * powr<3>(p2) - 9. * p1 * powr<4>(p2) - 3. * cosp1p2 * powr<5>(p2)) + cosp1p2 * (-9. * powr<4>(p1) * p2 - 21. * powr<2>(p1) * powr<3>(p2) + powr<4>(cosp1p2) * powr<2>(p1) * powr<3>(p2) - 6. * powr<5>(p2) + cosp1p2 * (-3. * powr<5>(p1) - 25. * powr<3>(p1) * powr<2>(p2) - 15. * p1 * powr<4>(p2)) + powr<3>(cosp1p2) * (10. * powr<3>(p1) * powr<2>(p2) + 6. * p1 * powr<4>(p2)) + powr<2>(cosp1p2) * (5. * powr<2>(p1) * powr<3>(p2) + 3. * powr<5>(p2)))), fma(powr<2>(cosl1p1), l1 * p1 * (powr<4>(cosl1p2) * cosp1p2 * powr<3>(l1) * (-cosp1p2 * p1 - p2) * powr<4>(p2) + powr<5>(l1) * (6. * powr<3>(p1) + 15. * cosp1p2 * powr<2>(p1) * p2 + (3. + 15. * powr<2>(cosp1p2)) * p1 * powr<2>(p2) + 9. * cosp1p2 * powr<3>(p2)) + powr<3>(l1) * (6. * powr<5>(p1) + 25. * cosp1p2 * powr<4>(p1) * p2 + (19. + 41. * powr<2>(cosp1p2)) * powr<3>(p1) * powr<2>(p2) + cosp1p2 * (60. + 16. * powr<2>(cosp1p2)) * powr<2>(p1) * powr<3>(p2) + (10. + 46. * powr<2>(cosp1p2)) * p1 * powr<4>(p2) + 21. * cosp1p2 * powr<5>(p2)) + l1 * p2 * (3. * powr<5>(p1) * p2 + 10. * powr<3>(p1) * powr<3>(p2) + 3. * p1 * powr<5>(p2) + powr<3>(cosp1p2) * (11. * powr<4>(p1) * powr<2>(p2) + 7. * powr<2>(p1) * powr<4>(p2)) + powr<2>(cosp1p2) * (12. * powr<5>(p1) * p2 + 41. * powr<3>(p1) * powr<3>(p2) + 15. * p1 * powr<5>(p2)) + cosp1p2 * (3. * powr<6>(p1) + 25. * powr<4>(p1) * powr<2>(p2) + 32. * powr<2>(p1) * powr<4>(p2) + 6. * powr<6>(p2))) + powr<3>(cosl1p2) * powr<2>(l1) * powr<2>(p2) * (-2. * powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) + powr<2>(cosp1p2) * p1 * p2 * (-12. * powr<2>(l1) + 5. * powr<2>(p1) - 12. * powr<2>(p2)) + p1 * p2 * (11. * powr<2>(l1) + 16. * powr<2>(p1) + 7. * powr<2>(p2)) + cosp1p2 * (5. * powr<4>(p1) + 19. * powr<2>(p1) * powr<2>(p2) - 6. * powr<4>(p2) + powr<2>(l1) * (9. * powr<2>(p1) - 10. * powr<2>(p2)))) + powr<2>(cosl1p2) * l1 * p2 * (-powr<4>(cosp1p2) * powr<3>(p1) * powr<3>(p2) + powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) * (5. * powr<2>(l1) - 12. * powr<2>(p1) - 12. * powr<2>(p2)) + powr<2>(cosp1p2) * p1 * p2 * (-12. * powr<4>(l1) - 12. * powr<4>(p1) - 3. * powr<2>(p1) * powr<2>(p2) - 12. * powr<4>(p2) + powr<2>(l1) * (-3. * powr<2>(p1) - 3. * powr<2>(p2))) + p1 * p2 * (12. * powr<4>(l1) + 15. * powr<4>(p1) + 46. * powr<2>(p1) * powr<2>(p2) + 15. * powr<4>(p2) + powr<2>(l1) * (41. * powr<2>(p1) + 41. * powr<2>(p2))) + cosp1p2 * (-3. * powr<6>(p1) + 39. * powr<4>(p1) * powr<2>(p2) + 39. * powr<2>(p1) * powr<4>(p2) - 3. * powr<6>(p2) + powr<2>(l1) * (-5. * powr<4>(p1) + 81. * powr<2>(p1) * powr<2>(p2) - 5. * powr<4>(p2)))) + cosl1p2 * (-powr<4>(cosp1p2) * powr<3>(p1) * powr<5>(p2) + powr<3>(cosp1p2) * powr<2>(p1) * powr<2>(p2) * (5. * powr<4>(l1) - 10. * powr<2>(p1) * powr<2>(p2) - 6. * powr<4>(p2) + powr<2>(l1) * (9. * powr<2>(p1) + 19. * powr<2>(p2))) + p1 * p2 * (3. * powr<6>(l1) + 9. * powr<4>(p1) * powr<2>(p2) + 21. * powr<2>(p1) * powr<4>(p2) + 6. * powr<6>(p2) + powr<4>(l1) * (25. * powr<2>(p1) + 25. * powr<2>(p2)) + powr<2>(l1) * (15. * powr<4>(p1) + 60. * powr<2>(p1) * powr<2>(p2) + 32. * powr<4>(p2))) + powr<2>(cosp1p2) * p1 * p2 * (-3. * powr<6>(l1) - 5. * powr<2>(p1) * powr<4>(p2) - 3. * powr<6>(p2) + powr<4>(l1) * (-5. * powr<2>(p1) + 39. * powr<2>(p2)) + powr<2>(l1) * (81. * powr<2>(p1) * powr<2>(p2) + 39. * powr<4>(p2))) + cosp1p2 * (3. * powr<6>(p1) * powr<2>(p2) + 25. * powr<4>(p1) * powr<4>(p2) + 15. * powr<2>(p1) * powr<6>(p2) + powr<6>(l1) * (-3. * powr<2>(p1) + 3. * powr<2>(p2)) + powr<4>(l1) * (-7. * powr<4>(p1) + 61. * powr<2>(p1) * powr<2>(p2) + 25. * powr<4>(p2)) + powr<2>(l1) * (-3. * powr<6>(p1) + 61. * powr<4>(p1) * powr<2>(p2) + 126. * powr<2>(p1) * powr<4>(p2) + 15. * powr<6>(p2))))), 0.)))))))))));
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
using DiFfRG::ZA3_kernel;