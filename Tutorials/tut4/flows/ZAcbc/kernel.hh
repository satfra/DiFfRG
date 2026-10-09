#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZAcbc_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const double &l1, const double &cos1, const double &cos2, const double &p, const double &k,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA3,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZAcbc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA4,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &Zc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZA,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const double cosl1p1 = cos1;
      const double cosl1p2 = ((-1.) * (cos1) + (sqrt(3. + (-3.) * (powr<2>(cos1)))) * (cos2)) * (0.5);
      const double cosl1p3 = ((-1.) * (cos1) + (-1.) * ((sqrt(3. + (-3.) * (powr<2>(cos1)))) * (cos2))) * (0.5);
      // clang-format off
using _T = decltype((-0.5) *
        ((4.) * ((cosl1p2) * (powr<3>(l1))) + (-8.) * ((powr<3>(cosl1p2)) * (powr<3>(l1))) +
         (6.) * ((powr<2>(l1)) * (p)) +
         (-4.) * ((powr<3>(cosl1p1)) * ((cosl1p2) * ((powr<2>(l1)) * (p)))) +
         (-4.) * ((powr<2>(cosl1p2)) * ((powr<2>(l1)) * (p))) +
         (-2.) * ((cosl1p2) * ((l1) * (powr<2>(p)))) + (3.) * (powr<3>(p)) +
         (-2.) *
             ((2.) * ((cosl1p2) * (powr<2>(l1))) + (l1) * (p) +
              (6.) * ((powr<2>(cosl1p2)) * ((l1) * (p))) + (-5.) * ((cosl1p2) * (powr<2>(p)))) *
             ((powr<2>(cosl1p1)) * (l1)) +
         ((2. + (-12.) * (powr<2>(cosl1p2))) * (powr<3>(l1)) +
          (l1) * (5. + (-4.) * (powr<2>(cosl1p2))) * (powr<2>(p)) +
          (6.) * ((cosl1p2) * (powr<3>(p))) +
          ((6.) * ((cosl1p2) * (p)) + (-8.) * ((powr<3>(cosl1p2)) * (p))) * (powr<2>(l1))) *
             (cosl1p1)) *
        ((p) *
         ((RBdot(powr<2>(k), powr<2>(l1))) * (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
          (dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
           (50.) *
               ((-1.) * (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)))) *
               ((powr<6>(k)) * (powr<-1>(1. + powr<6>(k))))) *
              (RB(powr<2>(k), powr<2>(l1)))) *
         ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
          ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) *
                         (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                     (powr<2>(l1)) * (ZA(l1)))) *
           ((powr<-1>(
                (RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                    (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                    (ZA(sqrt(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
            ((ZA3((0.816496580927726) *
                  (sqrt(powr<2>(l1) + (cosl1p1) * ((l1) * (p)) + powr<2>(p))))) *
             ((ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                          (0.6666666666666666) * (cosl1p1 + (-1.) * (cosl1p2)) * ((l1) * (p)) +
                          powr<2>(p)))) *
              ((ZAcbc((0.816496580927726) *
                      (sqrt(powr<2>(l1) + (-1.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))))) *
               (powr<-1>(
                   (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                       (Zc(k)) +
                   (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                       (Zc(sqrt(powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) +
                                powr<2>(p))))))))))))) +
    (0.5) *
        ((-4.) * ((cosl1p2) * (powr<3>(l1))) + (8.) * ((powr<3>(cosl1p2)) * (powr<3>(l1))) +
         (-6.) * ((powr<2>(l1)) * (p)) + (4.) * ((powr<2>(cosl1p2)) * ((powr<2>(l1)) * (p))) +
         (powr<3>(cosl1p1)) * ((-4.) * ((cosl1p2) * (l1)) + (-14.) * (p)) * ((l1) * (p)) +
         (2.) * ((cosl1p2) * ((l1) * (powr<2>(p)))) + (-3.) * (powr<3>(p)) +
         ((4.) * ((cosl1p2) * (powr<3>(l1))) + (12.) * ((powr<2>(l1)) * (p)) +
          (-12.) * ((powr<2>(cosl1p2)) * ((powr<2>(l1)) * (p))) +
          (-18.) * ((cosl1p2) * ((l1) * (powr<2>(p)))) + (6.) * (powr<3>(p))) *
             (powr<2>(cosl1p1)) +
         ((-2. + (12.) * (powr<2>(cosl1p2))) * (powr<3>(l1)) +
          (cosl1p2) * (14. + (-8.) * (powr<2>(cosl1p2))) * ((powr<2>(l1)) * (p)) +
          (l1) * (7. + (-4.) * (powr<2>(cosl1p2))) * (powr<2>(p)) +
          (6.) * ((cosl1p2) * (powr<3>(p)))) *
             (cosl1p1)) *
        ((powr<-1>(1. + powr<6>(k))) *
         ((dtZA(pow(1. + powr<6>(k), 0.16666666666666666667))) * (1. + (1.) * (powr<6>(k))) *
              (RB(powr<2>(k), powr<2>(l1))) +
          (RBdot(powr<2>(k), powr<2>(l1))) * (1. + powr<6>(k)) *
              (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
          (powr<6>(k)) *
              ((-50.) * (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
               (50.) * (ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667))))) *
              (RB(powr<2>(k), powr<2>(l1)))) *
         ((p) *
          ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
           ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) *
                          (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                      (powr<2>(l1)) * (ZA(l1)))) *
            ((powr<-1>(
                 (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                     (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                 (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                     (ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
             ((ZA3((0.816496580927726) *
                   (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))))) *
              ((ZAcbc(
                   (0.816496580927726) *
                   (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))))) *
               ((ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                            (-0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) +
                            powr<2>(p)))) *
                (powr<-1>((RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                              powr<2>(p))) *
                              (Zc(k)) +
                          (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                           (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                              (Zc(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                       powr<2>(p)))))))))))))) +
    (-0.25) * (-1. + (2.) * ((cosl1p1) * (cosl1p2)) + (2.) * (powr<2>(cosl1p2))) *
        ((powr<-1>(1. + powr<6>(k))) *
         ((2.) * ((cosl1p1) * (l1)) + (4.) * ((cosl1p2) * (l1)) + (-3.) * (p)) *
         ((p) *
          ((dtZA(pow(1. + powr<6>(k), 0.16666666666666666667))) * (1. + (1.) * (powr<6>(k))) *
               (RB(powr<2>(k), powr<2>(l1))) +
           (RBdot(powr<2>(k), powr<2>(l1))) * (1. + powr<6>(k)) *
               (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
           (powr<6>(k)) *
               ((-50.) * (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                (50.) * (ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667))))) *
               (RB(powr<2>(k), powr<2>(l1)))) *
          ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) *
                         (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                     (powr<2>(l1)) * (ZA(l1)))) *
           ((ZAcbc((0.816496580927726) *
                   (sqrt(powr<2>(l1) + (-1.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))))) *
            ((ZAcbc(
                 (0.816496580927726) *
                 (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))))) *
             ((ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                          (-0.6666666666666666) * (cosl1p1 + (2.) * (cosl1p2)) * ((l1) * (p)) +
                          powr<2>(p)))) *
              ((powr<-1>(
                   (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                       (Zc(k)) +
                   (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                       (Zc(sqrt(powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)))))) *
               (powr<-1>((RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                             powr<2>(p))) *
                             (Zc(k)) +
                         (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                          (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                             (Zc(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                      powr<2>(p))))))))))))) +
    (3.) *
        ((0.6666666666666666) * ((cosl1p2) * (powr<3>(l1))) +
         (-1.333333333333333) * ((powr<3>(cosl1p2)) * (powr<3>(l1))) +
         (-1.) * ((powr<2>(l1)) * (p)) +
         (0.6666666666666666) * ((powr<2>(cosl1p2)) * ((powr<2>(l1)) * (p))) +
         (-0.3333333333333333) * ((cosl1p2) * ((l1) * (powr<2>(p)))) + (-0.5) * (powr<3>(p)) +
         (powr<3>(cosl1p1)) *
             ((-0.6666666666666666) * ((cosl1p2) * (l1)) + (2.333333333333333) * (p)) *
             ((l1) * (p)) +
         ((-0.6666666666666666) * ((cosl1p2) * (powr<3>(l1))) + (2.) * ((powr<2>(l1)) * (p)) +
          (-2.) * ((powr<2>(cosl1p2)) * ((powr<2>(l1)) * (p))) +
          (3.) * ((cosl1p2) * ((l1) * (powr<2>(p)))) + (1.) * (powr<3>(p))) *
             (powr<2>(cosl1p1)) +
         ((0.3333333333333333 + (-2.) * (powr<2>(cosl1p2))) * (powr<3>(l1)) +
          (cosl1p2) * (2.333333333333333 + (-1.333333333333333) * (powr<2>(cosl1p2))) *
              ((powr<2>(l1)) * (p)) +
          (l1) * (-1.166666666666667 + (0.6666666666666666) * (powr<2>(cosl1p2))) * (powr<2>(p)) +
          (1.) * ((cosl1p2) * (powr<3>(p)))) *
             (cosl1p1)) *
        ((p) *
         ((dtZc(k)) * (RB(powr<2>(k),
                          powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) +
          (RBdot(powr<2>(k),
                 powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
              (Zc(k)) +
          ((-50.) * (Zc(k)) + (50.) * (Zc((1.02) * (k)))) *
              (RB(powr<2>(k),
                  powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)))) *
         ((powr<-1>(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
          ((powr<-1>((RB(powr<2>(k), powr<2>(l1))) *
                         (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                     (powr<2>(l1)) * (ZA(l1)))) *
           ((powr<-1>(
                (RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                    (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                    (ZA(sqrt(powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
            ((ZA3((0.816496580927726) *
                  (sqrt(powr<2>(l1) + (cosl1p1) * ((l1) * (p)) + powr<2>(p))))) *
             ((ZAcbc((0.816496580927726) *
                     (sqrt(powr<2>(l1) + (l1) * (cosl1p1 + cosl1p2) * (p) + powr<2>(p))))) *
              ((ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                           (0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) +
                           powr<2>(p)))) *
               (powr<-2>((RB(powr<2>(k), powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                             powr<2>(p))) *
                             (Zc(k)) +
                         (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) +
                          (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                             (Zc(sqrt(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                      powr<2>(p))))))))))))) +
    (1.) * (-0.5 + (1.) * ((cosl1p1) * (cosl1p2)) + (1.) * (powr<2>(cosl1p2))) *
        ((p) * ((1.) * ((cosl1p1) * (l1)) + (2.) * ((cosl1p2) * (l1)) + (1.5) * (p)) *
         ((powr<-1>((RB(powr<2>(k), powr<2>(l1))) *
                        (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                    (powr<2>(l1)) * (ZA(l1)))) *
          ((dtZc(k)) * (RB(powr<2>(k),
                           powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) +
           (RBdot(powr<2>(k),
                  powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
               (Zc(k)) +
           ((-50.) * (Zc(k)) + (50.) * (Zc((1.02) * (k)))) *
               (RB(powr<2>(k),
                   powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)))) *
          ((ZAcbc((0.816496580927726) *
                  (sqrt(powr<2>(l1) + (cosl1p2) * ((l1) * (p)) + powr<2>(p))))) *
           ((ZAcbc((0.816496580927726) *
                   (sqrt(powr<2>(l1) + (l1) * (cosl1p1 + cosl1p2) * (p) + powr<2>(p))))) *
            ((ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                         (0.6666666666666666) * (cosl1p1 + (2.) * (cosl1p2)) * ((l1) * (p)) +
                         powr<2>(p)))) *
             ((powr<-1>(
                  (RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                      (Zc(k)) +
                  (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                      (Zc(sqrt(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)))))) *
              (powr<-2>((RB(powr<2>(k),
                            powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                            (Zc(k)) +
                        (powr<2>(l1) + (2.) * ((cosl1p1) * ((l1) * (p))) +
                         (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                            (Zc(sqrt(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                     powr<2>(p)))))))))))) +
    (0.25) * (-1. + (2.) * ((cosl1p1) * (cosl1p2)) + (2.) * (powr<2>(cosl1p2))) *
        ((p) * ((2.) * ((cosl1p1) * (l1)) + (4.) * ((cosl1p2) * (l1)) + (3.) * (p)) *
         ((powr<-1>((RB(powr<2>(k), powr<2>(l1))) *
                        (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                    (powr<2>(l1)) * (ZA(l1)))) *
          ((RBdot(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
               (Zc(k)) +
           (dtZc(k) + (-50.) * (Zc(k)) + (50.) * (Zc((1.02) * (k)))) *
               (RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)))) *
          ((ZAcbc((0.816496580927726) *
                  (sqrt(powr<2>(l1) + (cosl1p2) * ((l1) * (p)) + powr<2>(p))))) *
           ((ZAcbc((0.816496580927726) *
                   (sqrt(powr<2>(l1) + (l1) * (cosl1p1 + cosl1p2) * (p) + powr<2>(p))))) *
            ((ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                         (0.6666666666666666) * (cosl1p1 + (2.) * (cosl1p2)) * ((l1) * (p)) +
                         powr<2>(p)))) *
             ((powr<-2>(
                  (RB(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                      (Zc(k)) +
                  (powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                      (Zc(sqrt(powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)))))) *
              (powr<-1>((RB(powr<2>(k),
                            powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                            (Zc(k)) +
                        (powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                            (Zc(sqrt(powr<2>(l1) + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                     powr<2>(p)))))))))))));
      // clang-format on
      const auto _interp1 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp2 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp3 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp4 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = RB(powr<2>(k), fma(2., (cosl1p1) * ((l1) * (p)), powr<2>(l1) + powr<2>(p)));
      const auto _interp8 = ZA(sqrt(fma(2., (cosl1p1) * ((l1) * (p)), powr<2>(l1) + powr<2>(p))));
      const auto _interp9 = ZA3((0.816496580927726) * (sqrt(fma(cosl1p1, (l1) * (p), powr<2>(l1) + powr<2>(p)))));
      const auto _interp11 =
          ZAcbc((0.816496580927726) * (sqrt(fma(-1., (cosl1p2) * ((l1) * (p)), powr<2>(l1) + powr<2>(p)))));
      const auto _interp12 = RB(powr<2>(k), fma(-2., (cosl1p2) * ((l1) * (p)), powr<2>(l1) + powr<2>(p)));
      const auto _interp13 = Zc(k);
      const auto _interp14 = Zc(sqrt(fma(-2., (cosl1p2) * ((l1) * (p)), powr<2>(l1) + powr<2>(p))));
      const auto _interp18 =
          ZAcbc((0.816496580927726) * (sqrt(fma(-1., (l1) * (cosl1p1 + cosl1p2) * (p), powr<2>(l1) + powr<2>(p)))));
      const auto _interp20 = RB(powr<2>(k), fma(-2., (l1) * (cosl1p1 + cosl1p2) * (p), powr<2>(l1) + powr<2>(p)));
      const auto _interp21 = Zc(sqrt(fma(-2., (l1) * (cosl1p1 + cosl1p2) * (p), powr<2>(l1) + powr<2>(p))));
      const auto _interp23 =
          ZAcbc((0.816496580927726) * (sqrt(fma(cosl1p1 + cosl1p2, (l1) * (p), powr<2>(l1) + powr<2>(p)))));
      const auto _interp25 = dtZc(k);
      const auto _interp26 = RB(powr<2>(k), fma(2., (l1) * (cosl1p1 + cosl1p2) * (p), powr<2>(l1) + powr<2>(p)));
      const auto _interp27 = RBdot(powr<2>(k), fma(2., (l1) * (cosl1p1 + cosl1p2) * (p), powr<2>(l1) + powr<2>(p)));
      const auto _interp28 = Zc((1.02) * (k));
      const auto _interp29 = Zc(sqrt(fma(2., (l1) * (cosl1p1 + cosl1p2) * (p), powr<2>(l1) + powr<2>(p))));
      const auto _interp30 = ZAcbc((0.816496580927726) * (sqrt(fma(cosl1p2, (l1) * (p), powr<2>(l1) + powr<2>(p)))));
      const auto _interp31 =
          ZAcbc(sqrt(fma(0.6666666666666666, powr<2>(l1),
                         fma(0.6666666666666666, (l1) * (cosl1p1 + (2.) * (cosl1p2)) * (p), powr<2>(p)))));
      const auto _interp32 = RB(powr<2>(k), fma(2., (cosl1p2) * ((l1) * (p)), powr<2>(l1) + powr<2>(p)));
      const auto _interp33 = Zc(sqrt(fma(2., (cosl1p2) * ((l1) * (p)), powr<2>(l1) + powr<2>(p))));
      _T _acc{};
      { // subkernel 1
        const auto _interp22 =
            ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                       (-0.6666666666666666) * (cosl1p1 + (2.) * (cosl1p2)) * ((l1) * (p)) + powr<2>(p)));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = (-2.) * ((cosl1p2) * ((l1) * (p)));
        const auto _cse3 = powr<2>(p);
        const auto _cse4 = powr<6>(k);
        const auto _cse5 = powr<2>(cosl1p2);
        const auto _cse6 = (2.) * ((cosl1p2) * ((l1) * (p)));
        const auto _cse7 = (_interp2) * (_interp3);
        const auto _cse8 = (_cse1) * (_interp6);
        const auto _cse9 = _cse7 + _cse8;
        _acc += fma(
            -0.25,
            (powr<-1>(1. + _cse4)) *
                ((_interp1) * (1. + _cse4) * (_interp2) + (_interp3) * (1. + (1.) * (_cse4)) * (_interp4) +
                 (_cse4) * ((-50.) * (_interp2) + (50.) * (_interp5)) * (_interp3)) *
                ((powr<-2>(_cse9)) * (-1. + (2.) * (_cse5) + (2.) * ((cosl1p1) * (cosl1p2))) *
                 ((_interp11) * ((2.) * ((cosl1p1) * (l1)) + (4.) * ((cosl1p2) * (l1)) + (-3.) * (p)) *
                  ((powr<-1>((_interp12) * (_interp13) + (_cse1 + _cse2 + _cse3) * (_interp14))) *
                   ((_interp18) *
                    ((_interp22) * ((p) * (powr<-1>((_interp13) * (_interp20) +
                                                    (_cse1 + _cse2 + _cse3 + (-2.) * ((cosl1p1) * ((l1) * (p)))) *
                                                        (_interp21))))))))),
            fma(1.,
                (powr<-1>(_cse9)) *
                    ((_interp25) * (_interp26) + (_interp13) * (_interp27) +
                     ((-50.) * (_interp13) + (50.) * (_interp28)) * (_interp26)) *
                    ((_interp23) * (-0.5 + (1.) * (_cse5) + (1.) * ((cosl1p1) * (cosl1p2))) *
                     ((_interp30) * ((1.) * ((cosl1p1) * (l1)) + (2.) * ((cosl1p2) * (l1)) + (1.5) * (p)) *
                      ((_interp31) * ((powr<-1>((_interp13) * (_interp32) + (_cse1 + _cse3 + _cse6) * (_interp33))) *
                                      ((p) * (powr<-2>((_interp13) * (_interp26) +
                                                       (_cse1 + _cse3 + _cse6 + (2.) * ((cosl1p1) * ((l1) * (p)))) *
                                                           (_interp29)))))))),
                0.));
      }
      { // subkernel 2
        const auto _interp24 =
            ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                       (0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
        const auto _interp34 = RBdot(powr<2>(k), powr<2>(l1) + (2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = powr<2>(p);
        const auto _cse3 = powr<3>(l1);
        const auto _cse4 = powr<2>(cosl1p2);
        const auto _cse5 = powr<3>(p);
        const auto _cse6 = (2.) * ((cosl1p1) * ((l1) * (p)));
        const auto _cse7 = (_interp13) * (_interp26);
        const auto _cse8 = (-50.) * (_interp13);
        const auto _cse9 = (50.) * (_interp28);
        const auto _cse10 = (_cse1) * (_interp6);
        const auto _cse11 = (_interp2) * (_interp3);
        const auto _cse12 = _cse10 + _cse11;
        const auto _cse13 = powr<-1>(_cse12);
        const auto _cse14 = _cse1 + _cse2 + _cse6;
        _acc += fma(
            3.,
            (_cse13) * ((_cse8 + _cse9) * (_interp26) + (_interp25) * (_interp26) + (_interp13) * (_interp27)) *
                ((powr<-1>(_cse14)) *
                 ((-0.5) * (_cse5) + (0.6666666666666666) * ((_cse3) * (cosl1p2)) +
                  (-1.333333333333333) * ((_cse3) * (powr<3>(cosl1p2))) +
                  (-0.3333333333333333) * ((_cse2) * ((cosl1p2) * (l1))) + (-1.) * ((_cse1) * (p)) +
                  (0.6666666666666666) * ((_cse1) * ((_cse4) * (p))) +
                  (powr<3>(cosl1p1)) * ((-0.6666666666666666) * ((cosl1p2) * (l1)) + (2.333333333333333) * (p)) *
                      ((l1) * (p)) +
                  ((1.) * (_cse5) + (-0.6666666666666666) * ((_cse3) * (cosl1p2)) +
                   (3.) * ((_cse2) * ((cosl1p2) * (l1))) + (2.) * ((_cse1) * (p)) +
                   (-2.) * ((_cse1) * ((_cse4) * (p)))) *
                      (powr<2>(cosl1p1)) +
                  ((0.3333333333333333 + (-2.) * (_cse4)) * (_cse3) + (1.) * ((_cse5) * (cosl1p2)) +
                   (_cse2) * (-1.166666666666667 + (0.6666666666666666) * (_cse4)) * (l1) +
                   (_cse1) * (2.333333333333333 + (-1.333333333333333) * (_cse4)) * ((cosl1p2) * (p))) *
                      (cosl1p1)) *
                 ((_interp23) *
                  ((_interp24) *
                   ((powr<-1>((_interp2) * (_interp7) + (_cse14) * (_interp8))) *
                    ((_interp9) * ((p) * (powr<-2>(_cse7 + (_cse1 + _cse2 + _cse6 + (2.) * ((cosl1p2) * ((l1) * (p)))) *
                                                               (_interp29))))))))),
            fma(0.25,
                (_cse13) * ((_cse8 + _cse9 + _interp25) * (_interp32) + (_interp13) * (_interp34)) *
                    ((_interp23) * (-1. + (2.) * (_cse4) + (2.) * ((cosl1p1) * (cosl1p2))) *
                     ((_interp30) * ((2.) * ((cosl1p1) * (l1)) + (4.) * ((cosl1p2) * (l1)) + (3.) * (p)) *
                      ((_interp31) *
                       ((p) * ((powr<-2>((_interp13) * (_interp32) +
                                         (_cse1 + _cse2 + (2.) * ((cosl1p2) * ((l1) * (p)))) * (_interp33))) *
                               (powr<-1>(_cse7 + (_cse1 + _cse2 + (2.) * (cosl1p1 + cosl1p2) * ((l1) * (p))) *
                                                     (_interp29)))))))),
                0.));
      }
      { // subkernel 3
        const auto _interp10 =
            ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                       (0.6666666666666666) * (cosl1p1 + (-1.) * (cosl1p2)) * ((l1) * (p)) + powr<2>(p)));
        const auto _interp15 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p));
        const auto _interp16 = ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
        const auto _interp17 =
            ZA3((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))));
        const auto _interp19 =
            ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                       (-0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
        const auto _cse1 = powr<3>(l1);
        const auto _cse2 = powr<2>(l1);
        const auto _cse3 = powr<2>(p);
        const auto _cse4 = powr<2>(cosl1p2);
        const auto _cse5 = powr<3>(p);
        const auto _cse6 = (-2.) * ((cosl1p1) * ((l1) * (p)));
        const auto _cse7 = _cse2 + _cse3 + _cse6;
        const auto _cse8 = powr<6>(k);
        const auto _cse9 = powr<3>(cosl1p2);
        const auto _cse10 = powr<3>(cosl1p1);
        const auto _cse11 = powr<2>(cosl1p1);
        const auto _cse12 = (2.) * ((cosl1p1) * ((l1) * (p)));
        const auto _cse13 = _cse12 + _cse2 + _cse3;
        const auto _cse14 = 1. + _cse8;
        const auto _cse15 = (_interp2) * (_interp3);
        const auto _cse16 = (_cse2) * (_interp6);
        const auto _cse17 = _cse15 + _cse16;
        const auto _cse18 = powr<-2>(_cse17);
        _acc += fma(
            0.5,
            (_cse18) *
                ((_cse14) * ((_interp1) * (_interp2)) + (_interp3) * (1. + (1.) * (_cse8)) * (_interp4) +
                 (_cse8) * ((-50.) * (_interp2) + (50.) * (_interp5)) * (_interp3)) *
                ((powr<-1>(_cse7)) *
                 ((-3.) * (_cse5) + (8.) * ((_cse1) * (_cse9)) + (-4.) * ((_cse1) * (cosl1p2)) +
                  (2.) * ((_cse3) * ((cosl1p2) * (l1))) + (-6.) * ((_cse2) * (p)) + (4.) * ((_cse2) * ((_cse4) * (p))) +
                  (_cse10) * ((-4.) * ((cosl1p2) * (l1)) + (-14.) * (p)) * ((l1) * (p)) +
                  ((6.) * (_cse5) + (4.) * ((_cse1) * (cosl1p2)) + (-18.) * ((_cse3) * ((cosl1p2) * (l1))) +
                   (12.) * ((_cse2) * (p)) + (-12.) * ((_cse2) * ((_cse4) * (p)))) *
                      (_cse11) +
                  ((-2. + (12.) * (_cse4)) * (_cse1) + (6.) * ((_cse5) * (cosl1p2)) +
                   (_cse3) * (7. + (-4.) * (_cse4)) * (l1) + (_cse2) * (14. + (-8.) * (_cse4)) * ((cosl1p2) * (p))) *
                      (cosl1p1)) *
                 ((powr<-1>(1. + _cse8)) *
                  ((_interp17) *
                   ((_interp18) *
                    ((_interp19) * ((powr<-1>((_cse7) * (_interp16) + (_interp15) * (_interp2))) *
                                    ((p) * (powr<-1>((_interp13) * (_interp20) +
                                                     (_cse2 + _cse3 + _cse6 + (-2.) * ((cosl1p2) * ((l1) * (p)))) *
                                                         (_interp21)))))))))),
            fma(-0.5,
                (powr<-1>(_cse13)) *
                    ((_interp1) * (_interp2) +
                     (_interp4 + (50.) * ((-1.) * (_interp2) + _interp5) * ((powr<-1>(_cse14)) * (_cse8))) *
                         (_interp3)) *
                    ((_cse18) *
                     ((3.) * (_cse5) + (-8.) * ((_cse1) * (_cse9)) + (4.) * ((_cse1) * (cosl1p2)) +
                      (-2.) * ((_cse3) * ((cosl1p2) * (l1))) + (6.) * ((_cse2) * (p)) +
                      (-4.) * ((_cse2) * ((_cse4) * (p))) + (-4.) * ((_cse10) * ((_cse2) * ((cosl1p2) * (p)))) +
                      (-2.) *
                          ((2.) * ((_cse2) * (cosl1p2)) + (-5.) * ((_cse3) * (cosl1p2)) + (l1) * (p) +
                           (6.) * ((_cse4) * ((l1) * (p)))) *
                          ((_cse11) * (l1)) +
                      ((2. + (-12.) * (_cse4)) * (_cse1) + (6.) * ((_cse5) * (cosl1p2)) +
                       (_cse3) * (5. + (-4.) * (_cse4)) * (l1) +
                       ((-8.) * ((_cse9) * (p)) + (6.) * ((cosl1p2) * (p))) * (_cse2)) *
                          (cosl1p1)) *
                     ((_interp10) *
                      ((_interp11) *
                       ((powr<-1>((_interp2) * (_interp7) + (_cse13) * (_interp8))) *
                        ((_interp9) *
                         ((p) * (powr<-1>((_interp12) * (_interp13) +
                                          (_cse2 + _cse3 + (-2.) * ((cosl1p2) * ((l1) * (p)))) * (_interp14))))))))),
                0.));
      }
      return _acc;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto
    constant(const double &p, const double &k,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA3,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZAcbc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA4,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &Zc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZA,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA)
    {
      using namespace DiFfRG;
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