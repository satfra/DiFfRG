#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA3_kernel
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
using _T = decltype((0.0909090909090909) *
        ((-72.) * ((powr<6>(l1)) * (p)) +
         (16.) * ((powr<5>(cosl1p2)) * ((powr<5>(l1)) * (powr<2>(p)))) +
         (-372.) * ((powr<4>(l1)) * (powr<3>(p))) +
         (32.) * ((powr<6>(cosl1p1)) * ((powr<4>(l1)) * (powr<3>(p)))) +
         (-378.) * ((powr<2>(l1)) * (powr<5>(p))) + (-99.) * (powr<7>(p)) +
         (powr<5>(cosl1p1)) *
             ((-208.) * (powr<2>(l1)) + (96.) * ((cosl1p2) * ((l1) * (p))) +
              (-448.) * (powr<2>(p))) *
             ((powr<3>(l1)) * (powr<2>(p))) +
         ((-96.) * ((powr<6>(l1)) * (p)) + (-240.) * ((powr<4>(l1)) * (powr<3>(p)))) *
             (powr<4>(cosl1p2)) +
         (powr<4>(cosl1p1)) *
             ((192.) * (powr<4>(l1)) + (-520.) * ((cosl1p2) * ((powr<3>(l1)) * (p))) +
              (powr<2>(l1)) * (1244. + (40.) * (powr<2>(cosl1p2))) * (powr<2>(p)) +
              (-1120.) * ((cosl1p2) * ((l1) * (powr<3>(p)))) + (1254.) * (powr<4>(p))) *
             ((powr<2>(l1)) * (p)) +
         ((48.) * (powr<7>(l1)) + (200.) * ((powr<5>(l1)) * (powr<2>(p))) +
          (128.) * ((powr<3>(l1)) * (powr<4>(p)))) *
             (powr<3>(cosl1p2)) +
         ((24.) * ((powr<6>(l1)) * (p)) + (56.) * ((powr<4>(l1)) * (powr<3>(p))) +
          (-72.) * ((powr<2>(l1)) * (powr<5>(p)))) *
             (powr<2>(cosl1p2)) +
         (powr<3>(cosl1p1)) *
             ((-48.) * (powr<6>(l1)) + (384.) * ((cosl1p2) * ((powr<5>(l1)) * (p))) +
              (powr<4>(l1)) * (-776. + (-16.) * (powr<2>(cosl1p2))) * (powr<2>(p)) +
              (cosl1p2) * (2488. + (-80.) * (powr<2>(cosl1p2))) * ((powr<3>(l1)) * (powr<3>(p))) +
              (powr<2>(l1)) * (-1920. + (-544.) * (powr<2>(cosl1p2))) * (powr<4>(p)) +
              (2508.) * ((cosl1p2) * ((l1) * (powr<5>(p)))) + (-924.) * (powr<6>(p))) *
             (l1) +
         ((144.) * ((powr<5>(l1)) * (powr<2>(p))) + (482.) * ((powr<3>(l1)) * (powr<4>(p))) +
          (231.) * ((l1) * (powr<6>(p)))) *
             (cosl1p2) +
         ((132.) * ((powr<6>(l1)) * (p)) + (326.) * ((powr<4>(l1)) * (powr<3>(p))) +
          (-72.) * ((powr<4>(cosl1p2)) * ((powr<4>(l1)) * (powr<3>(p)))) +
          (153.) * ((powr<2>(l1)) * (powr<5>(p))) + (198.) * (powr<7>(p)) +
          ((496.) * ((powr<5>(l1)) * (powr<2>(p))) + (304.) * ((powr<3>(l1)) * (powr<4>(p)))) *
              (powr<3>(cosl1p2)) +
          ((-144.) * ((powr<6>(l1)) * (p)) + (436.) * ((powr<4>(l1)) * (powr<3>(p))) +
           (1302.) * ((powr<2>(l1)) * (powr<5>(p)))) *
              (powr<2>(cosl1p2)) +
          ((-72.) * (powr<7>(l1)) + (-1164.) * ((powr<5>(l1)) * (powr<2>(p))) +
           (-2880.) * ((powr<3>(l1)) * (powr<4>(p))) + (-1386.) * ((l1) * (powr<6>(p)))) *
              (cosl1p2)) *
             (powr<2>(cosl1p1)) +
         ((288.) * ((powr<5>(l1)) * (powr<2>(p))) +
          (-16.) * ((powr<5>(cosl1p2)) * ((powr<4>(l1)) * (powr<3>(p)))) +
          (964.) * ((powr<3>(l1)) * (powr<4>(p))) + (462.) * ((l1) * (powr<6>(p))) +
          ((232.) * ((powr<5>(l1)) * (powr<2>(p))) + (176.) * ((powr<3>(l1)) * (powr<4>(p)))) *
              (powr<4>(cosl1p2)) +
          ((-336.) * ((powr<6>(l1)) * (p)) + (-808.) * ((powr<4>(l1)) * (powr<3>(p))) +
           (48.) * ((powr<2>(l1)) * (powr<5>(p)))) *
              (powr<3>(cosl1p2)) +
          ((72.) * (powr<7>(l1)) + (12.) * ((powr<5>(l1)) * (powr<2>(p))) +
           (-704.0000000000001) * ((powr<3>(l1)) * (powr<4>(p))) +
           (-462.) * ((l1) * (powr<6>(p)))) *
              (powr<2>(cosl1p2)) +
          ((132.) * ((powr<6>(l1)) * (p)) + (326.) * ((powr<4>(l1)) * (powr<3>(p))) +
           (153.) * ((powr<2>(l1)) * (powr<5>(p))) + (198.) * (powr<7>(p))) *
              (cosl1p2)) *
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
         ((powr<-1>(p)) *
          ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
           ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                      (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
            ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) *
                           (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                       (powr<2>(l1)) * (ZA(l1)))) *
             ((powr<-1>(
                  (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                      (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                  (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                      (ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
              ((powr<-1>((RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                             powr<2>(p))) *
                             (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                         (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                          (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                             (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                      powr<2>(p)))))) *
               ((ZA3((0.816496580927726) *
                     (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))))) *
                ((ZA3((0.816496580927726) *
                      (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                            powr<2>(p))))) *
                 (ZA3(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                           (-0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) +
                           powr<2>(p))))))))))))) +
    (-0.2727272727272727) *
        ((-54. + (53.) * (powr<2>(cosl1p1)) + (110.) * ((cosl1p1) * (cosl1p2)) +
          (53.) * (powr<2>(cosl1p2))) *
             (powr<2>(l1)) +
         (-1.) *
             ((-54.) * (cosl1p1) + (53.) * (powr<3>(cosl1p1)) + (-54.) * (cosl1p2) +
              (163.) * ((powr<2>(cosl1p1)) * (cosl1p2)) +
              (163.) * ((cosl1p1) * (powr<2>(cosl1p2))) + (53.) * (powr<3>(cosl1p2))) *
             ((l1) * (p)) +
         (33.) * (-1. + powr<2>(cosl1p1) + (2.) * ((cosl1p1) * (cosl1p2)) + powr<2>(cosl1p2)) *
             (powr<2>(p))) *
        ((powr<-1>(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
         ((RBdot(powr<2>(k), powr<2>(l1))) * (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
          (dtZA(pow(1. + powr<6>(k), 0.16666666666666666667)) +
           (50.) *
               ((-1.) * (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)))) *
               ((powr<6>(k)) * (powr<-1>(1. + powr<6>(k))))) *
              (RB(powr<2>(k), powr<2>(l1)))) *
         ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) *
                        (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                    (powr<2>(l1)) * (ZA(l1)))) *
          ((powr<-1>((RB(powr<2>(k),
                         powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))) *
                         (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                     (powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)) *
                         (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                  powr<2>(p)))))) *
           ((ZA3((0.816496580927726) *
                 (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))))) *
            (ZA4((0.5) * (sqrt((2.) * (powr<2>(l1)) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                               (3.) * (powr<2>(p)))))))))) +
    (0.3636363636363637) *
        ((1.) * ((powr<3>(cosl1p1)) * (l1)) + (-1.) * ((powr<3>(cosl1p2)) * (l1)) +
         (cosl1p1) * ((-1.5) * ((cosl1p2) * (l1)) + (-2.75) * (p)) * (cosl1p2) +
         ((1.5) * ((cosl1p2) * (l1)) + (-2.75) * (p)) * (powr<2>(cosl1p1)) + (1.5) * (p) +
         (-0.5) * ((powr<2>(cosl1p2)) * (p))) *
        ((powr<2>(l1)) *
         ((dtZc(k)) * (RB(powr<2>(k), powr<2>(l1))) + (RBdot(powr<2>(k), powr<2>(l1))) * (Zc(k)) +
          ((-50.) * (Zc(k)) + (50.) * (Zc((1.02) * (k)))) * (RB(powr<2>(k), powr<2>(l1)))) *
         ((powr<-1>(p)) *
          ((ZAcbc((0.816496580927726) *
                  (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))))) *
           ((ZAcbc((0.816496580927726) *
                   (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))))) *
            ((ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                         (-0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) +
                         powr<2>(p)))) *
             ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) * (Zc(k)) + (powr<2>(l1)) * (Zc(l1)))) *
              ((powr<-1>(
                   (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                       (Zc(k)) +
                   (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                       (Zc(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
               (powr<-1>((RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                             powr<2>(p))) *
                             (Zc(k)) +
                         (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) +
                          (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                             (Zc(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                      powr<2>(p))))))))))))));
      // clang-format on
      const auto _interp1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = RB(powr<2>(k), fma(-2., (cosl1p1) * ((l1) * (p)), powr<2>(l1) + powr<2>(p)));
      const auto _interp9 = RB(powr<2>(k), fma(-2., (l1) * (cosl1p1 + cosl1p2) * (p), powr<2>(l1) + powr<2>(p)));
      const auto _interp10 = ZA(sqrt(fma(-2., (l1) * (cosl1p1 + cosl1p2) * (p), powr<2>(l1) + powr<2>(p))));
      const auto _interp12 =
          ZA3((0.816496580927726) * (sqrt(fma(-1., (l1) * (cosl1p1 + cosl1p2) * (p), powr<2>(l1) + powr<2>(p)))));
      _T _acc{};
      { // subkernel 1
        const auto _interp14 = ZA4(
            (0.5) * (sqrt((2.) * (powr<2>(l1)) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + (3.) * (powr<2>(p)))));
        const auto _interp15 =
            ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))));
        const auto _interp16 =
            ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p))));
        const auto _interp17 =
            ZAcbc(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                       (-0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
        const auto _interp18 = dtZc(k);
        const auto _interp19 = Zc(k);
        const auto _interp20 = Zc((1.02) * (k));
        const auto _interp21 = Zc(l1);
        const auto _interp22 = Zc(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
        const auto _interp23 = Zc(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = powr<2>(cosl1p1);
        const auto _cse3 = powr<2>(cosl1p2);
        const auto _cse4 = powr<2>(p);
        const auto _cse5 = cosl1p1 + cosl1p2;
        const auto _cse6 = (-2.) * ((_cse5) * ((l1) * (p)));
        const auto _cse7 = _cse1 + _cse4 + _cse6;
        const auto _cse8 = powr<6>(k);
        const auto _cse9 = powr<3>(cosl1p1);
        const auto _cse10 = powr<3>(cosl1p2);
        const auto _cse11 = (-2.75) * (p);
        const auto _cse12 = (-2.) * ((cosl1p1) * ((l1) * (p)));
        _acc += fma(
            -0.2727272727272727,
            (powr<-1>(_cse7)) *
                ((_interp3) * (_interp4) +
                 (_interp1 + (50.) * ((-1.) * (_interp4) + _interp5) * ((_cse8) * (powr<-1>(1. + _cse8)))) *
                     (_interp2)) *
                ((_interp12) *
                 ((33.) * (-1. + _cse2 + _cse3 + (2.) * ((cosl1p1) * (cosl1p2))) * (_cse4) +
                  (-54. + (53.) * (_cse2) + (53.) * (_cse3) + (110.) * ((cosl1p1) * (cosl1p2))) * (_cse1) +
                  (-1.) *
                      ((53.) * (_cse10) + (53.) * (_cse9) + (-54.) * (cosl1p1) + (163.) * ((_cse3) * (cosl1p1)) +
                       (-54.) * (cosl1p2) + (163.) * ((_cse2) * (cosl1p2))) *
                      ((l1) * (p))) *
                 ((_interp14) * ((powr<-2>((_interp2) * (_interp4) + (_cse1) * (_interp6))) *
                                 (powr<-1>((_cse7) * (_interp10) + (_interp4) * (_interp9)))))),
            fma(0.3636363636363637,
                (_cse1) *
                    ((_interp18) * (_interp2) + ((-50.) * (_interp19) + (50.) * (_interp20)) * (_interp2) +
                     (_interp19) * (_interp3)) *
                    ((_interp15) *
                     ((-1.) * ((_cse10) * (l1)) + (1.) * ((_cse9) * (l1)) +
                      (cosl1p1) * (_cse11 + (-1.5) * ((cosl1p2) * (l1))) * (cosl1p2) +
                      (_cse11 + (1.5) * ((cosl1p2) * (l1))) * (_cse2) + (1.5) * (p) + (-0.5) * ((_cse3) * (p))) *
                     ((_interp16) *
                      ((_interp17) *
                       ((powr<-2>((_interp19) * (_interp2) + (_cse1) * (_interp21))) *
                        ((powr<-1>((_cse1 + _cse12 + _cse4) * (_interp22) + (_interp19) * (_interp7))) *
                         ((powr<-1>(p)) *
                          (powr<-1>((_interp19) * (_interp9) +
                                    (_cse1 + _cse12 + _cse4 + (-2.) * ((cosl1p2) * ((l1) * (p)))) * (_interp23))))))))),
                0.));
      }
      { // subkernel 2
        const auto _interp8 = ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
        const auto _interp11 =
            ZA3((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))));
        const auto _interp13 =
            ZA3(sqrt((0.6666666666666666) * (powr<2>(l1)) +
                     (-0.6666666666666666) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) + powr<2>(p)));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = (-2.) * ((cosl1p1) * ((l1) * (p)));
        const auto _cse3 = powr<2>(p);
        const auto _cse4 = powr<4>(l1);
        const auto _cse5 = powr<3>(p);
        const auto _cse6 = powr<6>(l1);
        const auto _cse7 = powr<3>(l1);
        const auto _cse8 = powr<5>(l1);
        const auto _cse9 = powr<4>(p);
        const auto _cse10 = powr<2>(cosl1p2);
        const auto _cse11 = powr<5>(p);
        const auto _cse12 = powr<6>(p);
        const auto _cse13 = powr<4>(cosl1p2);
        const auto _cse14 = powr<7>(p);
        const auto _cse15 = powr<3>(cosl1p2);
        const auto _cse16 = powr<7>(l1);
        const auto _cse17 = powr<5>(cosl1p2);
        const auto _cse18 = (132.) * ((_cse6) * (p));
        const auto _cse19 = (326.) * ((_cse4) * (_cse5));
        const auto _cse20 = (153.) * ((_cse1) * (_cse11));
        const auto _cse21 = (198.) * (_cse14);
        const auto _cse22 = _cse1 + _cse2 + _cse3;
        const auto _cse23 = (-2.) * ((cosl1p2) * ((l1) * (p)));
        const auto _cse24 = _cse1 + _cse2 + _cse23 + _cse3;
        const auto _cse25 = powr<6>(k);
        _acc +=
            (0.0909090909090909) *
            ((powr<-1>(_cse22)) *
             ((powr<-1>(_cse24)) *
              ((powr<-1>(1. + _cse25)) *
               ((_interp11) *
                ((_interp12) *
                 ((_interp13) *
                  ((powr<-1>(p)) *
                   ((fma(-378., (_cse1) * (_cse11),
                         fma(-99., _cse14,
                             fma(-372., (_cse4) * (_cse5),
                                 fma(16., (_cse17) * ((_cse3) * (_cse8)),
                                     fma(_cse15,
                                         (48.) * (_cse16) + (200.) * ((_cse3) * (_cse8)) + (128.) * ((_cse7) * (_cse9)),
                                         fma(32., (_cse4) * ((_cse5) * (powr<6>(cosl1p1))),
                                             fma(cosl1p2,
                                                 (144.) * ((_cse3) * (_cse8)) + (482.) * ((_cse7) * (_cse9)) +
                                                     (231.) * ((_cse12) * (l1)),
                                                 fma(-72., (_cse6) * (p),
                                                     fma(_cse13,
                                                         (-240.) * ((_cse4) * (_cse5)) + (-96.) * ((_cse6) * (p)),
                                                         fma(_cse10,
                                                             (-72.) * ((_cse1) * (_cse11)) +
                                                                 (56.) * ((_cse4) * (_cse5)) + (24.) * ((_cse6) * (p)),
                                                             fma(_cse1,
                                                                 (powr<4>(cosl1p1)) *
                                                                     ((_cse1) * (1244. + (40.) * (_cse10)) * (_cse3) +
                                                                      (192.) * (_cse4) + (1254.) * (_cse9) +
                                                                      (-1120.) * ((_cse5) * ((cosl1p2) * (l1))) +
                                                                      (-520.) * ((_cse7) * ((cosl1p2) * (p)))) *
                                                                     (p),
                                                                 fma(powr<3>(cosl1p1),
                                                                     ((-924.) * (_cse12) +
                                                                      (_cse3) * (-776. + (-16.) * (_cse10)) * (_cse4) +
                                                                      (-48.) * (_cse6) +
                                                                      (_cse1) * (-1920. + (-544.) * (_cse10)) *
                                                                          (_cse9) +
                                                                      (_cse5) * (2488. + (-80.) * (_cse10)) *
                                                                          ((_cse7) * (cosl1p2)) +
                                                                      (2508.) * ((_cse11) * ((cosl1p2) * (l1))) +
                                                                      (384.) * ((_cse8) * ((cosl1p2) * (p)))) *
                                                                         (l1),
                                                                     fma(_cse3,
                                                                         (_cse7) *
                                                                             ((-208.) * (_cse1) + (-448.) * (_cse3) +
                                                                              (96.) * ((cosl1p2) * ((l1) * (p)))) *
                                                                             (powr<5>(cosl1p1)),
                                                                         fma(cosl1p1,
                                                                             (-16.) * ((_cse17) * ((_cse4) * (_cse5))) +
                                                                                 (288.) * ((_cse3) * (_cse8)) +
                                                                                 (964.) * ((_cse7) * (_cse9)) +
                                                                                 ((232.) * ((_cse3) * (_cse8)) +
                                                                                  (176.) * ((_cse7) * (_cse9))) *
                                                                                     (_cse13) +
                                                                                 (_cse18 + _cse19 + _cse20 + _cse21) *
                                                                                     (cosl1p2) +
                                                                                 (462.) * ((_cse12) * (l1)) +
                                                                                 ((72.) * (_cse16) +
                                                                                  (12.) * ((_cse3) * (_cse8)) +
                                                                                  (-704.0000000000001) *
                                                                                      ((_cse7) * (_cse9)) +
                                                                                  (-462.) * ((_cse12) * (l1))) *
                                                                                     (_cse10) +
                                                                                 ((48.) * ((_cse1) * (_cse11)) +
                                                                                  (-808.) * ((_cse4) * (_cse5)) +
                                                                                  (-336.) * ((_cse6) * (p))) *
                                                                                     (_cse15),
                                                                             fma(powr<2>(cosl1p1),
                                                                                 _cse18 + _cse19 + _cse20 + _cse21 +
                                                                                     (-72.) * ((_cse13) *
                                                                                               ((_cse4) * (_cse5))) +
                                                                                     ((496.) * ((_cse3) * (_cse8)) +
                                                                                      (304.) * ((_cse7) * (_cse9))) *
                                                                                         (_cse15) +
                                                                                     ((-72.) * (_cse16) +
                                                                                      (-1164.) * ((_cse3) * (_cse8)) +
                                                                                      (-2880.) * ((_cse7) * (_cse9)) +
                                                                                      (-1386.) * ((_cse12) * (l1))) *
                                                                                         (cosl1p2) +
                                                                                     ((1302.) * ((_cse1) * (_cse11)) +
                                                                                      (436.) * ((_cse4) * (_cse5)) +
                                                                                      (-144.) * ((_cse6) * (p))) *
                                                                                         (_cse10),
                                                                                 0.)))))))))))))))) *
                    ((powr<-1>(fma(_cse24, _interp10, fma(_interp4, _interp9, 0.)))) *
                     ((fma(1. + (1.) * (_cse25), (_interp1) * (_interp2),
                           fma(1. + _cse25, (_interp3) * (_interp4),
                               fma(_cse25, ((-50.) * (_interp4) + (50.) * (_interp5)) * (_interp2), 0.)))) *
                      ((powr<-2>(fma(_interp2, _interp4, fma(_cse1, _interp6, 0.)))) *
                       (powr<-1>(fma(_interp4, _interp7, fma(_cse22, _interp8, 0.)))))))))))))));
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
using DiFfRG::ZA3_kernel;