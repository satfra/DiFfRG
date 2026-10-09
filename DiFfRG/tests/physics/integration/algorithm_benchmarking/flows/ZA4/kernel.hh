#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA4_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const double &l1, const double &cos1, const double &cos2, const double &phi, const double &p,
           const double &k, const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA3,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZAcbc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA4,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &Zc,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZA,
           const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const double cosl1p1 = (sqrt(1. + (-1.) * (powr<2>(cos1)))) * (cos2);
      const double cosl1p2 = (-0.3333333333333333) *
                             (cos2 + (-2.) * ((sqrt(2. + (-2.) * (powr<2>(cos2)))) * (cos(phi)))) *
                             (sqrt(1. + (-1.) * (powr<2>(cos1))));
      const double cosl1p3 =
          (-0.3333333333333333) *
          (cos2 + (cos(phi) + (-1.732050807568877) * (sin(phi))) * (sqrt(2. + (-2.) * (powr<2>(cos2))))) *
          (sqrt(1. + (-1.) * (powr<2>(cos1))));
      const double cosl1p4 =
          (-0.3333333333333333) *
          (cos2 + (cos(phi) + (1.732050807568877) * (sin(phi))) * (sqrt(2. + (-2.) * (powr<2>(cos2))))) *
          (sqrt(1. + (-1.) * (powr<2>(cos1))));
      // clang-format off
using _T = decltype((-0.163265306122449) *
        ((-243.) * ((powr<2>(cosl1p3)) * (powr<10>(l1))) +
         (729.0000000000001) * ((powr<4>(cosl1p3)) * (powr<10>(l1))) +
         (324.) * ((cosl1p3) * ((powr<9>(l1)) * (p))) +
         (-1458.) * ((powr<5>(cosl1p3)) * ((powr<9>(l1)) * (p))) +
         (-432.) * ((powr<8>(l1)) * (powr<2>(p))) +
         (-756.) * ((powr<2>(cosl1p3)) * ((powr<8>(l1)) * (powr<2>(p)))) +
         (3645.) * ((powr<4>(cosl1p3)) * ((powr<8>(l1)) * (powr<2>(p)))) +
         (243.) * ((powr<6>(cosl1p3)) * ((powr<8>(l1)) * (powr<2>(p)))) +
         (972.) * ((powr<7>(cosl1p2)) * ((powr<7>(l1)) * (powr<3>(p)))) +
         (1008.) * ((cosl1p3) * ((powr<7>(l1)) * (powr<3>(p)))) +
         (-54.) * ((powr<3>(cosl1p3)) * ((powr<7>(l1)) * (powr<3>(p)))) +
         (-3726.) * ((powr<5>(cosl1p3)) * ((powr<7>(l1)) * (powr<3>(p)))) +
         (-3756.) * ((powr<6>(l1)) * (powr<4>(p))) +
         (972.) * ((powr<8>(cosl1p1)) * ((powr<6>(l1)) * (powr<4>(p)))) +
         (1764.) * ((powr<2>(cosl1p3)) * ((powr<6>(l1)) * (powr<4>(p)))) +
         (4050.) * ((powr<4>(cosl1p3)) * ((powr<6>(l1)) * (powr<4>(p)))) +
         (162.) * ((powr<6>(cosl1p3)) * ((powr<6>(l1)) * (powr<4>(p)))) +
         (3624.) * ((cosl1p3) * ((powr<5>(l1)) * (powr<5>(p)))) +
         (-1206.) * ((powr<3>(cosl1p3)) * ((powr<5>(l1)) * (powr<5>(p)))) +
         (-2052.) * ((powr<5>(cosl1p3)) * ((powr<5>(l1)) * (powr<5>(p)))) +
         (-7159.999999999999) * ((powr<4>(l1)) * (powr<6>(p))) +
         (4128.) * ((powr<2>(cosl1p3)) * ((powr<4>(l1)) * (powr<6>(p)))) +
         (1548.) * ((powr<4>(cosl1p3)) * ((powr<4>(l1)) * (powr<6>(p)))) +
         (3712.) * ((cosl1p3) * ((powr<3>(l1)) * (powr<7>(p)))) +
         (-1248.) * ((powr<3>(cosl1p3)) * ((powr<3>(l1)) * (powr<7>(p)))) +
         (-4834.) * ((powr<2>(l1)) * (powr<8>(p))) +
         (2176.) * ((powr<2>(cosl1p3)) * ((powr<2>(l1)) * (powr<8>(p)))) +
         (836.) * ((cosl1p3) * ((l1) * (powr<9>(p)))) + (-1024.) * (powr<10>(p)) +
         (powr<6>(cosl1p2)) *
             ((-6318.) * (powr<2>(l1)) + (3402.) * ((cosl1p3) * ((l1) * (p))) +
              (-9072.) * (powr<2>(p))) *
             ((powr<6>(l1)) * (powr<2>(p))) +
         (powr<7>(cosl1p1)) *
             ((-6803.999999999999) * (powr<2>(l1)) + (3402.) * ((cosl1p2) * ((l1) * (p))) +
              (4374.) * ((cosl1p3) * ((l1) * (p))) + (-5183.999999999999) * (powr<2>(p))) *
             ((powr<5>(l1)) * (powr<3>(p))) +
         (powr<5>(cosl1p2)) *
             ((5832.000000000001) * (powr<4>(l1)) +
              (-18954.) * ((cosl1p3) * ((powr<3>(l1)) * (p))) +
              (powr<2>(l1)) * (20898. + (3888.) * (powr<2>(cosl1p3))) * (powr<2>(p)) +
              (-27216.) * ((cosl1p3) * ((l1) * (powr<3>(p)))) + (19926.) * (powr<4>(p))) *
             ((powr<5>(l1)) * (p)) +
         (powr<4>(cosl1p2)) *
             ((-1458.) * (powr<6>(l1)) + (14580.) * ((cosl1p3) * ((powr<5>(l1)) * (p))) +
              (powr<4>(l1)) * (-9234. + (-16281.) * (powr<2>(cosl1p3))) * (powr<2>(p)) +
              (cosl1p3) * (50625. + (1215.) * (powr<2>(cosl1p3))) * ((powr<3>(l1)) * (powr<3>(p))) +
              (powr<2>(l1)) * (-25353. + (-22113.) * (powr<2>(cosl1p3))) * (powr<4>(p)) +
              (49868.99999999999) * ((cosl1p3) * ((l1) * (powr<5>(p)))) +
              (-17802.) * (powr<6>(p))) *
             (powr<4>(l1)) +
         (powr<2>(cosl1p2)) *
             ((486. + (-729.0000000000001) * (powr<2>(cosl1p3))) * (powr<8>(l1)) +
              (cosl1p3) * (-1215. + (-1458.) * (powr<2>(cosl1p3))) * ((powr<7>(l1)) * (p)) +
              (powr<6>(l1)) *
                  (1755. + (-4131.) * (powr<2>(cosl1p3)) + (6075.) * (powr<4>(cosl1p3))) *
                  (powr<2>(p)) +
              (cosl1p3) * (8586. + (-10449.) * (powr<2>(cosl1p3)) + (-972.) * (powr<4>(cosl1p3))) *
                  ((powr<5>(l1)) * (powr<3>(p))) +
              (powr<4>(l1)) *
                  (-3528. + (-22275.) * (powr<2>(cosl1p3)) + (9153.) * (powr<4>(cosl1p3))) *
                  (powr<4>(p)) +
              (cosl1p3) * (17712. + (-1404.) * (powr<2>(cosl1p3))) *
                  ((powr<3>(l1)) * (powr<5>(p))) +
              (powr<2>(l1)) * (-17742. + (-15012.) * (powr<2>(cosl1p3))) * (powr<6>(p)) +
              (10824.) * ((cosl1p3) * ((l1) * (powr<7>(p)))) + (-11368.) * (powr<8>(p))) *
             (powr<2>(l1)) +
         ((-243. + (-729.0000000000001) * (powr<2>(cosl1p2)) + (1215.) * ((cosl1p2) * (cosl1p3)) +
           (1458.) * (powr<2>(cosl1p3))) *
              (powr<10>(l1)) +
          (powr<9>(l1)) *
              ((20412.) * (powr<3>(cosl1p2)) + (4779.) * (cosl1p3) +
               (11178.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-16038.) * (powr<3>(cosl1p3)) +
               (-3321. + (-15552.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
              (p) +
          (powr<8>(l1)) *
              (1053. + (-68768.99999999999) * (powr<4>(cosl1p2)) +
               (-97848.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (-8586.) * (powr<2>(cosl1p3)) +
               (27702.) * (powr<4>(cosl1p3)) +
               (14013. + (1944.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
               ((7209.) * (cosl1p3) + (61560.00000000001) * (powr<3>(cosl1p3))) * (cosl1p2)) *
              (powr<2>(p)) +
          (powr<7>(l1)) *
              ((55404.) * (powr<5>(cosl1p2)) + (-14148.) * (cosl1p3) +
               (120123.) * ((powr<4>(cosl1p2)) * (cosl1p3)) + (-28323.) * (powr<3>(cosl1p3)) +
               (-9234.) * (powr<5>(cosl1p3)) +
               (54648. + (55080.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
               ((33291.) * (cosl1p3) + (-39042.) * (powr<3>(cosl1p3))) * (powr<2>(cosl1p2)) +
               (-26838. + (-21870.) * (powr<2>(cosl1p3)) +
                (-43983.00000000001) * (powr<4>(cosl1p3))) *
                   (cosl1p2)) *
              (powr<3>(p)) +
          (powr<6>(l1)) *
              (4644. + (-5832.000000000001) * (powr<6>(cosl1p2)) +
               (-16686.) * ((powr<5>(cosl1p2)) * (cosl1p3)) + (10881.) * (powr<2>(cosl1p3)) +
               (35721.) * (powr<4>(cosl1p3)) + (486.) * (powr<6>(cosl1p3)) +
               (-121689. + (-14256.) * (powr<2>(cosl1p3))) * (powr<4>(cosl1p2)) +
               ((-188892.) * (cosl1p3) + (-324.) * (powr<3>(cosl1p3))) * (powr<3>(cosl1p2)) +
               (48708.00000000001 + (-21816.) * (powr<2>(cosl1p3)) + (6237.) * (powr<4>(cosl1p3))) *
                   (powr<2>(cosl1p2)) +
               ((90351.) * (cosl1p3) + (77085.00000000001) * (powr<3>(cosl1p3)) +
                (3402.) * (powr<5>(cosl1p3))) *
                   (cosl1p2)) *
              (powr<4>(p)) +
          (powr<5>(l1)) *
              ((21762.) * (powr<5>(cosl1p2)) + (-90102.) * (cosl1p3) +
               (51111.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
               (-3473.999999999999) * (powr<3>(cosl1p3)) + (-3240.) * (powr<5>(cosl1p3)) +
               (42561. + (25974.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
               ((-6372.000000000001) * (cosl1p3) + (-11988.) * (powr<3>(cosl1p3))) *
                   (powr<2>(cosl1p2)) +
               (-70008.00000000001 + (-28809.) * (powr<2>(cosl1p3)) +
                (-12825.) * (powr<4>(cosl1p3))) *
                   (cosl1p2)) *
              (powr<5>(p)) +
          (powr<4>(l1)) *
              (-5424. + (-16830.) * (powr<4>(cosl1p2)) +
               (-24525.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (69840.) * (powr<2>(cosl1p3)) +
               (-702.) * (powr<4>(cosl1p3)) +
               (110952. + (-11736.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
               ((225468.) * (cosl1p3) + (-5958.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
              (powr<6>(p)) +
          (powr<3>(l1)) *
              ((-51858.) * (powr<3>(cosl1p2)) + (-94120.) * (cosl1p3) +
               (-106866.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (2700.) * (powr<3>(cosl1p3)) +
               (-65252. + (-48816.00000000001) * (powr<2>(cosl1p3))) * (cosl1p2)) *
              (powr<7>(p)) +
          (powr<2>(l1)) *
              (-1594. + (82452.) * (powr<2>(cosl1p2)) + (115242.) * ((cosl1p2) * (cosl1p3)) +
               (33312.) * (powr<2>(cosl1p3))) *
              (powr<8>(p)) +
          (l1) * ((-34188.) * (cosl1p2) + (-27480.) * (cosl1p3)) * (powr<9>(p)) +
          (3072.) * (powr<10>(p))) *
             (powr<2>(cosl1p1)) +
         (powr<6>(cosl1p1)) *
             ((8991.) * (powr<4>(l1)) +
              (powr<2>(l1)) *
                  (23328. + (3402.) * (powr<2>(cosl1p2)) + (12555.) * ((cosl1p2) * (cosl1p3)) +
                   (6803.999999999999) * (powr<2>(cosl1p3))) *
                  (powr<2>(p)) +
              (l1) * ((-22086.) * (cosl1p2) + (-14202.) * (cosl1p3)) * (powr<3>(p)) +
              (21960.) * (powr<4>(p)) +
              ((-20169.) * ((cosl1p2) * (p)) + (-27459.) * ((cosl1p3) * (p))) * (powr<3>(l1))) *
             ((powr<4>(l1)) * (powr<2>(p))) +
         (powr<4>(cosl1p1)) *
             ((729.0000000000001) * (powr<8>(l1)) +
              (powr<6>(l1)) *
                  (5589. + (6075.) * (powr<2>(cosl1p2)) + (59940.) * ((cosl1p2) * (cosl1p3)) +
                   (36450.) * (powr<2>(cosl1p3))) *
                  (powr<2>(p)) +
              (powr<5>(l1)) *
                  ((33048.) * (powr<3>(cosl1p2)) + (-46602.) * (cosl1p3) +
                   (-21708.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-36693.) * (powr<3>(cosl1p3)) +
                   (-61128. + (-72657.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (powr<3>(p)) +
              (powr<4>(l1)) *
                  (49581. + (-13122.) * (powr<4>(cosl1p2)) +
                   (-14580.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (36855.) * (powr<2>(cosl1p3)) +
                   (6318.) * (powr<4>(cosl1p3)) +
                   (71955. + (7695.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
                   ((141075.) * (cosl1p3) + (16605.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                  (powr<4>(p)) +
              (powr<3>(l1)) *
                  ((-6318.) * (powr<3>(cosl1p2)) + (-135891.) * (cosl1p3) +
                   (-31563.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-2322.) * (powr<3>(cosl1p3)) +
                   (-183429. + (-20547.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (powr<5>(p)) +
              (powr<2>(l1)) *
                  (100098. + (103734.) * (powr<2>(cosl1p2)) + (132255.) * ((cosl1p2) * (cosl1p3)) +
                   (19944.) * (powr<2>(cosl1p3))) *
                  (powr<6>(p)) +
              (l1) * ((-151758.) * (cosl1p2) + (-98622.) * (cosl1p3)) * (powr<7>(p)) +
              (48972.) * (powr<8>(p)) +
              ((-7290.000000000001) * ((cosl1p2) * (p)) + (-14580.) * ((cosl1p3) * (p))) *
                  (powr<7>(l1))) *
             (powr<2>(l1)) +
         (powr<5>(cosl1p1)) *
             ((-4374.) * (powr<6>(l1)) +
              (powr<4>(l1)) *
                  (-21546. + (-13608.) * (powr<2>(cosl1p2)) + (-65772.) * ((cosl1p2) * (cosl1p3)) +
                   (-35478.) * (powr<2>(cosl1p3))) *
                  (powr<2>(p)) +
              (powr<3>(l1)) *
                  ((-3888.) * (powr<3>(cosl1p2)) + (62964.) * (cosl1p3) +
                   (8100.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                   (7290.000000000001) * (powr<3>(cosl1p3)) +
                   (77003.99999999999 + (16200.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (powr<3>(p)) +
              (powr<2>(l1)) *
                  (-63864. + (-31698.) * (powr<2>(cosl1p2)) + (-44982.) * ((cosl1p2) * (cosl1p3)) +
                   (-8046.) * (powr<2>(cosl1p3))) *
                  (powr<4>(p)) +
              (l1) * ((82638.) * (cosl1p2) + (49122.) * (cosl1p3)) * (powr<5>(p)) +
              (-50076.00000000001) * (powr<6>(p)) +
              ((20898.) * ((cosl1p2) * (p)) + (33048.) * ((cosl1p3) * (p))) * (powr<5>(l1))) *
             ((powr<3>(l1)) * (p)) +
         (powr<3>(cosl1p2)) *
             ((-486.) * ((powr<6>(l1)) * (p)) + (4644.) * ((powr<4>(l1)) * (powr<3>(p))) +
              (-972.) * ((powr<4>(cosl1p3)) * ((powr<4>(l1)) * (powr<3>(p)))) +
              (14526.) * ((powr<2>(l1)) * (powr<5>(p))) + (12480.) * (powr<7>(p)) +
              ((-972.) * ((powr<5>(l1)) * (powr<2>(p))) +
               (1620.) * ((powr<3>(l1)) * (powr<4>(p)))) *
                  (powr<3>(cosl1p3)) +
              ((8748.) * ((powr<6>(l1)) * (p)) + (27378.) * ((powr<4>(l1)) * (powr<3>(p))) +
               (33507.) * ((powr<2>(l1)) * (powr<5>(p)))) *
                  (powr<2>(cosl1p3)) +
              ((-2916.) * (powr<7>(l1)) + (-16524.) * ((powr<5>(l1)) * (powr<2>(p))) +
               (-50031.00000000001) * ((powr<3>(l1)) * (powr<4>(p))) +
               (-34020.) * ((l1) * (powr<6>(p)))) *
                  (cosl1p3)) *
             (powr<3>(l1)) +
         (cosl1p2) *
             ((-324.) * ((powr<8>(l1)) * (p)) + (1584.) * ((powr<6>(l1)) * (powr<3>(p))) +
              (-243.) * ((powr<6>(cosl1p3)) * ((powr<6>(l1)) * (powr<3>(p)))) +
              (13344.) * ((powr<4>(l1)) * (powr<5>(p))) +
              (17428.) * ((powr<2>(l1)) * (powr<7>(p))) + (6016.) * (powr<9>(p)) +
              ((3402.) * ((powr<7>(l1)) * (powr<2>(p))) +
               (3726.) * ((powr<5>(l1)) * (powr<4>(p)))) *
                  (powr<5>(cosl1p3)) +
              ((-4374.) * ((powr<8>(l1)) * (p)) + (-14742.) * ((powr<6>(l1)) * (powr<3>(p))) +
               (-7992.) * ((powr<4>(l1)) * (powr<5>(p)))) *
                  (powr<4>(cosl1p3)) +
              ((729.0000000000001) * (powr<9>(l1)) + (5103.) * ((powr<7>(l1)) * (powr<2>(p))) +
               (4104.) * ((powr<5>(l1)) * (powr<4>(p))) +
               (1728.) * ((powr<3>(l1)) * (powr<6>(p)))) *
                  (powr<3>(cosl1p3)) +
              ((243.) * ((powr<8>(l1)) * (p)) + (4482.) * ((powr<6>(l1)) * (powr<3>(p))) +
               (1530.) * ((powr<4>(l1)) * (powr<5>(p))) +
               (-2364.) * ((powr<2>(l1)) * (powr<7>(p)))) *
                  (powr<2>(cosl1p3)) +
              ((486.) * (powr<9>(l1)) + (-270.) * ((powr<7>(l1)) * (powr<2>(p))) +
               (-2088.) * ((powr<5>(l1)) * (powr<4>(p))) +
               (-6054.) * ((powr<3>(l1)) * (powr<6>(p))) + (-3198.) * ((l1) * (powr<8>(p)))) *
                  (cosl1p3)) *
             (l1) +
         (powr<3>(cosl1p1)) *
             ((486.) * ((powr<8>(l1)) * (p)) + (-13662.) * ((powr<6>(l1)) * (powr<3>(p))) +
              (-13122.) * ((powr<5>(cosl1p2)) * ((powr<5>(l1)) * (powr<4>(p)))) +
              (2916.) * ((powr<5>(cosl1p3)) * ((powr<5>(l1)) * (powr<4>(p)))) +
              (-53370.00000000001) * ((powr<4>(l1)) * (powr<5>(p))) +
              (-53124.) * ((powr<2>(l1)) * (powr<7>(p))) + (-20556.) * (powr<9>(p)) +
              (powr<4>(cosl1p2)) *
                  ((73628.99999999999) * (powr<2>(l1)) +
                   (-27783.00000000001) * ((cosl1p3) * ((l1) * (p))) + (25056.) * (powr<2>(p))) *
                  ((powr<4>(l1)) * (powr<3>(p))) +
              (powr<3>(cosl1p2)) *
                  ((-44712.) * (powr<4>(l1)) + (93312.) * ((cosl1p3) * ((powr<3>(l1)) * (p))) +
                   (powr<2>(l1)) * (-48600. + (-12312.) * (powr<2>(cosl1p3))) * (powr<2>(p)) +
                   (34938.) * ((cosl1p3) * ((l1) * (powr<3>(p)))) + (36432.) * (powr<4>(p))) *
                  ((powr<3>(l1)) * (powr<2>(p))) +
              ((-29403.) * ((powr<6>(l1)) * (powr<3>(p))) +
               (-6372.000000000001) * ((powr<4>(l1)) * (powr<5>(p)))) *
                  (powr<4>(cosl1p3)) +
              (powr<2>(cosl1p2)) *
                  ((1458.) * (powr<6>(l1)) + (-3564.) * ((cosl1p3) * ((powr<5>(l1)) * (p))) +
                   (powr<4>(l1)) * (-41661. + (-19926.) * (powr<2>(cosl1p3))) * (powr<2>(p)) +
                   (cosl1p3) * (16551. + (10692.) * (powr<2>(cosl1p3))) *
                       ((powr<3>(l1)) * (powr<3>(p))) +
                   (powr<2>(l1)) * (-140949. + (-378.) * (powr<2>(cosl1p3))) * (powr<4>(p)) +
                   (85905.) * ((cosl1p3) * ((l1) * (powr<5>(p)))) + (-152982.) * (powr<6>(p))) *
                  ((powr<2>(l1)) * (p)) +
              ((39852.) * ((powr<7>(l1)) * (powr<2>(p))) +
               (32940.) * ((powr<5>(l1)) * (powr<4>(p))) +
               (-8676.) * ((powr<3>(l1)) * (powr<6>(p)))) *
                  (powr<3>(cosl1p3)) +
              ((-13122.) * ((powr<8>(l1)) * (p)) + (-12609.) * ((powr<6>(l1)) * (powr<3>(p))) +
               (-45873.00000000001) * ((powr<4>(l1)) * (powr<5>(p))) +
               (-46710.) * ((powr<2>(l1)) * (powr<7>(p)))) *
                  (powr<2>(cosl1p3)) +
              (cosl1p2) *
                  ((729.0000000000001) * (powr<8>(l1)) +
                   (-17496.) * ((cosl1p3) * ((powr<7>(l1)) * (p))) +
                   (powr<6>(l1)) * (18711. + (58643.99999999999) * (powr<2>(cosl1p3))) *
                       (powr<2>(p)) +
                   (cosl1p3) * (-76788. + (-75168.00000000001) * (powr<2>(cosl1p3))) *
                       ((powr<5>(l1)) * (powr<3>(p))) +
                   (powr<4>(l1)) *
                       (111771. + (65528.99999999999) * (powr<2>(cosl1p3)) +
                        (11745.) * (powr<4>(cosl1p3))) *
                       (powr<4>(p)) +
                   (cosl1p3) * (-237636. + (-13878.) * (powr<2>(cosl1p3))) *
                       ((powr<3>(l1)) * (powr<5>(p))) +
                   (powr<2>(l1)) * (213900. + (31671.) * (powr<2>(cosl1p3))) * (powr<6>(p)) +
                   (-202428.) * ((cosl1p3) * ((l1) * (powr<7>(p)))) + (114324.) * (powr<8>(p))) *
                  (l1) +
              ((2187.) * (powr<9>(l1)) + (3645.) * ((powr<7>(l1)) * (powr<2>(p))) +
               (86553.) * ((powr<5>(l1)) * (powr<4>(p))) +
               (186492.) * ((powr<3>(l1)) * (powr<6>(p))) + (81564.) * ((l1) * (powr<8>(p)))) *
                  (cosl1p3)) *
             (l1) +
         ((2592.) * ((powr<7>(l1)) * (powr<3>(p))) +
          (-972.) * ((powr<7>(cosl1p2)) * ((powr<6>(l1)) * (powr<4>(p)))) +
          (16968.) * ((powr<5>(l1)) * (powr<5>(p))) + (21140.) * ((powr<3>(l1)) * (powr<7>(p))) +
          (6852.000000000001) * ((l1) * (powr<9>(p))) +
          (powr<6>(cosl1p2)) *
              ((16038.) * (powr<2>(l1)) + (-3402.) * ((cosl1p3) * ((l1) * (p))) +
               (5183.999999999999) * (powr<2>(p))) *
              ((powr<5>(l1)) * (powr<3>(p))) +
          (powr<5>(cosl1p2)) *
              ((-36450.) * (powr<4>(l1)) + (47304.) * ((cosl1p3) * ((powr<3>(l1)) * (p))) +
               (powr<2>(l1)) * (-64044. + (-3888.) * (powr<2>(cosl1p3))) * (powr<2>(p)) +
               (15552.) * ((cosl1p3) * ((l1) * (powr<3>(p)))) + (-10206.) * (powr<4>(p))) *
              ((powr<4>(l1)) * (powr<2>(p))) +
          ((-729.0000000000001) * ((powr<7>(l1)) * (powr<3>(p))) +
           (-162.) * ((powr<5>(l1)) * (powr<5>(p)))) *
              (powr<6>(cosl1p3)) +
          (powr<4>(cosl1p2)) *
              ((20412.) * (powr<6>(l1)) + (-82944.) * ((cosl1p3) * ((powr<5>(l1)) * (p))) +
               (powr<4>(l1)) * (73899. + (40257.) * (powr<2>(cosl1p3))) * (powr<2>(p)) +
               (cosl1p3) * (-149040. + (-1215.) * (powr<2>(cosl1p3))) *
                   ((powr<3>(l1)) * (powr<3>(p))) +
               (powr<2>(l1)) * (81819. + (12393.) * (powr<2>(cosl1p3))) * (powr<4>(p)) +
               (-26541.) * ((cosl1p3) * ((l1) * (powr<5>(p)))) + (-558.) * (powr<6>(p))) *
              ((powr<3>(l1)) * (p)) +
          ((6803.999999999999) * ((powr<8>(l1)) * (powr<2>(p))) +
           (9234.) * ((powr<6>(l1)) * (powr<4>(p))) + (756.) * ((powr<4>(l1)) * (powr<6>(p)))) *
              (powr<5>(cosl1p3)) +
          ((-8748.) * ((powr<9>(l1)) * (p)) + (-23598.) * ((powr<7>(l1)) * (powr<3>(p))) +
           (-11754.) * ((powr<5>(l1)) * (powr<5>(p))) + (864.) * ((powr<3>(l1)) * (powr<7>(p)))) *
              (powr<4>(cosl1p3)) +
          (powr<3>(cosl1p2)) *
              ((-2916.) * (powr<8>(l1)) + (33048.) * ((cosl1p3) * ((powr<7>(l1)) * (p))) +
               (powr<6>(l1)) * (-11988. + (-42444.) * (powr<2>(cosl1p3))) * (powr<2>(p)) +
               (cosl1p3) * (122958. + (1296.) * (powr<2>(cosl1p3))) *
                   ((powr<5>(l1)) * (powr<3>(p))) +
               (powr<4>(l1)) *
                   (-42885.00000000001 + (-74952.) * (powr<2>(cosl1p3)) +
                    (972.) * (powr<4>(cosl1p3))) *
                   (powr<4>(p)) +
               (cosl1p3) * (149004. + (-1620.) * (powr<2>(cosl1p3))) *
                   ((powr<3>(l1)) * (powr<5>(p))) +
               (powr<2>(l1)) * (-22200. + (-20979.) * (powr<2>(cosl1p3))) * (powr<6>(p)) +
               (-3924.) * ((cosl1p3) * ((l1) * (powr<7>(p)))) + (17100.) * (powr<8>(p))) *
              (powr<2>(l1)) +
          ((2187.) * (powr<10>(l1)) + (6237.) * ((powr<8>(l1)) * (powr<2>(p))) +
           (3312.) * ((powr<6>(l1)) * (powr<4>(p))) + (2796.) * ((powr<4>(l1)) * (powr<6>(p))) +
           (720.) * ((powr<2>(l1)) * (powr<8>(p)))) *
              (powr<3>(cosl1p3)) +
          ((3807.) * ((powr<9>(l1)) * (p)) + (4104.) * ((powr<7>(l1)) * (powr<3>(p))) +
           (-23412.) * ((powr<5>(l1)) * (powr<5>(p))) + (-29764.) * ((powr<3>(l1)) * (powr<7>(p))) +
           (-6924.) * ((l1) * (powr<9>(p)))) *
              (powr<2>(cosl1p3)) +
          ((-972.) * (powr<10>(l1)) + (-1458.) * ((powr<8>(l1)) * (powr<2>(p))) +
           (9936.) * ((powr<6>(l1)) * (powr<4>(p))) + (16446.) * ((powr<4>(l1)) * (powr<6>(p))) +
           (11950.) * ((powr<2>(l1)) * (powr<8>(p))) + (3072.) * (powr<10>(p))) *
              (cosl1p3) +
          ((486. + (1215.) * (powr<2>(cosl1p3))) * (powr<10>(l1)) +
           (-15552.) * ((powr<3>(cosl1p3)) * ((powr<9>(l1)) * (p))) +
           (powr<8>(l1)) * (3564. + (-1863.) * (powr<2>(cosl1p3)) + (27864.) * (powr<4>(cosl1p3))) *
               (powr<2>(p)) +
           (cosl1p3) *
               (-5183.999999999999 + (-44226.00000000001) * (powr<2>(cosl1p3)) +
                (-8748.) * (powr<4>(cosl1p3))) *
               ((powr<7>(l1)) * (powr<3>(p))) +
           (powr<6>(l1)) *
               (-648. + (-10170.) * (powr<2>(cosl1p3)) + (42525.) * (powr<4>(cosl1p3)) +
                (243.) * (powr<6>(cosl1p3))) *
               (powr<4>(p)) +
           (cosl1p3) * (-47448. + (-14544.) * (powr<2>(cosl1p3)) + (-1782.) * (powr<4>(cosl1p3))) *
               ((powr<5>(l1)) * (powr<5>(p))) +
           (powr<4>(l1)) *
               (-27294. + (25116.) * (powr<2>(cosl1p3)) + (1782.) * (powr<4>(cosl1p3))) *
               (powr<6>(p)) +
           (cosl1p3) * (-55896. + (2412.) * (powr<2>(cosl1p3))) * ((powr<3>(l1)) * (powr<7>(p))) +
           (powr<2>(l1)) * (-15138. + (18018.) * (powr<2>(cosl1p3))) * (powr<8>(p)) +
           (-20556.) * ((cosl1p3) * ((l1) * (powr<9>(p)))) + (3072.) * (powr<10>(p))) *
              (cosl1p2) +
          (powr<2>(cosl1p2)) *
              ((-4293.) * ((powr<8>(l1)) * (p)) + (-8586.) * ((powr<6>(l1)) * (powr<3>(p))) +
               (972.) * ((powr<5>(cosl1p3)) * ((powr<5>(l1)) * (powr<4>(p)))) +
               (-3318.) * ((powr<4>(l1)) * (powr<5>(p))) +
               (-896.) * ((powr<2>(l1)) * (powr<7>(p))) + (-13632.) * (powr<9>(p)) +
               ((-16200.) * ((powr<6>(l1)) * (powr<3>(p))) +
                (-5265.) * ((powr<4>(l1)) * (powr<5>(p)))) *
                   (powr<4>(cosl1p3)) +
               ((19764.) * ((powr<7>(l1)) * (powr<2>(p))) +
                (37503.) * ((powr<5>(l1)) * (powr<4>(p))) +
                (-3618.) * ((powr<3>(l1)) * (powr<6>(p)))) *
                   (powr<3>(cosl1p3)) +
               ((4374.) * ((powr<8>(l1)) * (p)) + (23571.) * ((powr<6>(l1)) * (powr<3>(p))) +
                (60237.) * ((powr<4>(l1)) * (powr<5>(p))) +
                (-1818.) * ((powr<2>(l1)) * (powr<7>(p)))) *
                   (powr<2>(cosl1p3)) +
               ((-2430.) * (powr<9>(l1)) + (-11826.) * ((powr<7>(l1)) * (powr<2>(p))) +
                (-47025.) * ((powr<5>(l1)) * (powr<4>(p))) +
                (6023.999999999999) * ((powr<3>(l1)) * (powr<6>(p))) +
                (34398.) * ((l1) * (powr<8>(p)))) *
                   (cosl1p3)) *
              (l1)) *
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
         ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
          ((powr<-1>(powr<2>(l1) +
                     (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p) +
                     powr<2>(p))) *
           ((powr<-1>((3.) * (powr<2>(l1)) + (-6.) * ((cosl1p1) * ((l1) * (p))) +
                      (-6.) * ((cosl1p2) * ((l1) * (p))) + (4.) * (powr<2>(p)))) *
            ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) *
                           (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                       (powr<2>(l1)) * (ZA(l1)))) *
             ((powr<-1>(
                  (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                      (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                  (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                      (ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
              ((powr<-1>(
                   (RB(powr<2>(k), powr<2>(l1) +
                                       (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                       powr<2>(p))) *
                       (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                   (powr<2>(l1) +
                    (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p) +
                    powr<2>(p)) *
                       (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                powr<2>(p)))))) *
               ((powr<-1>((3.) * ((RB(powr<2>(k), powr<2>(l1) +
                                                      (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                                      (1.333333333333333) * (powr<2>(p)))) *
                                  (ZA(pow(1. + powr<6>(k), 0.16666666666666666667)))) +
                          ((3.) * (powr<2>(l1)) + (-6.) * ((cosl1p1) * ((l1) * (p))) +
                           (-6.) * ((cosl1p2) * ((l1) * (p))) + (4.) * (powr<2>(p))) *
                              (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                       (1.333333333333333) * (powr<2>(p))))))) *
                ((ZA3((0.816496580927726) *
                      (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))))) *
                 ((ZA3((0.816496580927726) *
                       (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                             powr<2>(p))))) *
                  ((ZA3((0.3333333333333333) *
                        (sqrt((6.) * (powr<2>(l1)) +
                              (-6.) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) +
                              (10.) * (powr<2>(p)))))) *
                   (ZA3((0.3333333333333333) *
                        (sqrt((6.) * (powr<2>(l1)) +
                              (-6.) * ((2.) * (cosl1p1) + (2.) * (cosl1p2) + cosl1p3) *
                                  ((l1) * (p)) +
                              (10.) * (powr<2>(p))))))))))))))))) +
    (0.02040816326530612) *
        ((-120. + (-27.) * (powr<4>(cosl1p1)) + (54.) * (powr<4>(cosl1p2)) +
          ((-27.) * (cosl1p2) + (-81.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
          (108.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (63.) * (powr<2>(cosl1p3)) +
          (-27.) * (powr<4>(cosl1p3)) +
          (63. + (27.) * (powr<2>(cosl1p2)) + (-45.) * ((cosl1p2) * (cosl1p3)) +
           (-54.) * (powr<2>(cosl1p3))) *
              (powr<2>(cosl1p1)) +
          (210. + (27.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
          ((210.) * (cosl1p3) + (-27.) * (powr<3>(cosl1p3))) * (cosl1p2) +
          ((108.) * (powr<3>(cosl1p2)) + (324.) * (cosl1p3) +
           (90.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-81.) * (powr<3>(cosl1p3)) +
           (210. + (-45.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
              (cosl1p1)) *
             (powr<6>(l1)) +
         (powr<5>(l1)) *
             ((27.) * (powr<5>(cosl1p1)) + (-108.) * (powr<5>(cosl1p2)) + (312.) * (cosl1p3) +
              (-270.) * ((powr<4>(cosl1p2)) * (cosl1p3)) + (-18.) * (powr<3>(cosl1p3)) +
              (27.) * (powr<5>(cosl1p3)) +
              ((81.) * (cosl1p2) + (108.) * (cosl1p3)) * (powr<4>(cosl1p1)) +
              (-1776. + (-162.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
              (-18. + (27.) * (powr<2>(cosl1p2)) + (234.) * ((cosl1p2) * (cosl1p3)) +
               (135.) * (powr<2>(cosl1p3))) *
                  (powr<3>(cosl1p1)) +
              ((-2664.) * (cosl1p3) + (27.) * (powr<3>(cosl1p3))) * (powr<2>(cosl1p2)) +
              (624. + (-924.) * (powr<2>(cosl1p3)) + (81.) * (powr<4>(cosl1p3))) * (cosl1p2) +
              ((-162.) * (powr<3>(cosl1p2)) + (-126.) * (cosl1p3) +
               (-27.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (135.) * (powr<3>(cosl1p3)) +
               (-924. + (198.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (powr<2>(cosl1p1)) +
              (312. + (-270.) * (powr<4>(cosl1p2)) + (-396.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
               (-126.) * (powr<2>(cosl1p3)) + (108.) * (powr<4>(cosl1p3)) +
               (-2664. + (-27.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
               ((-1992.) * (cosl1p3) + (234.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                  (cosl1p1)) *
             (p) +
         (powr<4>(l1)) *
             (-1072. + (-27.) * ((powr<5>(cosl1p1)) * (cosl1p2)) + (54.) * (powr<6>(cosl1p2)) +
              (6372.000000000001) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
              (162.) * ((powr<5>(cosl1p2)) * (cosl1p3)) + (402.) * (powr<2>(cosl1p3)) +
              (-99.) * (powr<4>(cosl1p3)) +
              (-99. + (-54.) * (powr<2>(cosl1p2)) + (-108.) * ((cosl1p2) * (cosl1p3))) *
                  (powr<4>(cosl1p1)) +
              (3186. + (135.) * (powr<2>(cosl1p3))) * (powr<4>(cosl1p2)) +
              (1576. + (3579.) * (powr<2>(cosl1p3)) + (-54.) * (powr<4>(cosl1p3))) *
                  (powr<2>(cosl1p2)) +
              ((1576.) * (cosl1p3) + (393.) * (powr<3>(cosl1p3)) + (-27.) * (powr<5>(cosl1p3))) *
                  (cosl1p2) +
              ((-630.) * (cosl1p3) + (-153.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
               (393. + (-135.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (powr<3>(cosl1p1)) +
              (402. + (135.) * (powr<4>(cosl1p2)) + (72.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
               (-1062.) * (powr<2>(cosl1p3)) +
               (3579. + (-144.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
               ((609.) * (cosl1p3) + (-135.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                  (powr<2>(cosl1p1)) +
              ((162.) * (powr<5>(cosl1p2)) + (876.) * (cosl1p3) +
               (306.) * ((powr<4>(cosl1p2)) * (cosl1p3)) + (-630.) * (powr<3>(cosl1p3)) +
               (6372.000000000001 + (72.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
               ((6587.999999999999) * (cosl1p3) + (-153.) * (powr<3>(cosl1p3))) *
                   (powr<2>(cosl1p2)) +
               (1576. + (609.) * (powr<2>(cosl1p3)) + (-108.) * (powr<4>(cosl1p3))) * (cosl1p2)) *
                  (cosl1p1)) *
             (powr<2>(p)) +
         (powr<3>(l1)) *
             ((99.) * ((powr<4>(cosl1p1)) * (cosl1p2)) + (-1164.) * (powr<5>(cosl1p2)) +
              (1404.) * (cosl1p3) + (-2910.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
              (-294.) * (powr<3>(cosl1p3)) +
              (-294. + (-285.) * (powr<2>(cosl1p2)) + (630.) * ((cosl1p2) * (cosl1p3))) *
                  (powr<3>(cosl1p1)) +
              (-7447.999999999999 + (-2130.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
              ((-11172.) * (cosl1p3) + (-285.) * (powr<3>(cosl1p3))) * (powr<2>(cosl1p2)) +
              (2808. + (-4311.999999999999) * (powr<2>(cosl1p3)) + (99.) * (powr<4>(cosl1p3))) *
                  (cosl1p2) +
              ((-2130.) * (powr<3>(cosl1p2)) + (-798.) * (cosl1p3) +
               (-153.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
               (-4311.999999999999 + (1062.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (powr<2>(cosl1p1)) +
              (1404. + (-2910.) * (powr<4>(cosl1p2)) + (-3792.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
               (-798.) * (powr<2>(cosl1p3)) +
               (-11172. + (-153.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
               ((-8456.) * (cosl1p3) + (630.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                  (cosl1p1)) *
             (powr<3>(p)) +
         (powr<2>(l1)) *
             (-1364. + (204.) * ((powr<3>(cosl1p1)) * (cosl1p2)) + (5028.) * (powr<4>(cosl1p2)) +
              (10056.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (406.) * (powr<2>(cosl1p3)) +
              (406. + (5232.000000000001) * (powr<2>(cosl1p2)) + (468.) * ((cosl1p2) * (cosl1p3))) *
                  (powr<2>(cosl1p1)) +
              (1816. + (5232.000000000001) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
              ((1816.) * (cosl1p3) + (204.) * (powr<3>(cosl1p3))) * (cosl1p2) +
              ((10056.) * (powr<3>(cosl1p2)) + (764.0000000000001) * (cosl1p3) +
               (10320.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
               (1816. + (468.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (cosl1p1)) *
             (powr<4>(p)) +
         (l1) *
             ((-2316.) * ((powr<2>(cosl1p1)) * (cosl1p2)) + (-4632.) * (powr<3>(cosl1p2)) +
              (772.0000000000001) * (cosl1p3) +
              (-6947.999999999999) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
              (772.0000000000001 + (-6947.999999999999) * (powr<2>(cosl1p2)) +
               (-4632.) * ((cosl1p2) * (cosl1p3))) *
                  (cosl1p1) +
              (1544. + (-2316.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
             (powr<5>(p)) +
         (-392. + (1176.) * ((cosl1p1) * (cosl1p2)) + (1176.) * (powr<2>(cosl1p2)) +
          (1176.) * ((cosl1p2) * (cosl1p3))) *
             (powr<6>(p))) *
        ((powr<-1>(1. + powr<6>(k))) *
         ((dtZA(pow(1. + powr<6>(k), 0.16666666666666666667))) * (1. + (1.) * (powr<6>(k))) *
              (RB(powr<2>(k), powr<2>(l1))) +
          (RBdot(powr<2>(k), powr<2>(l1))) * (1. + powr<6>(k)) *
              (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
          (powr<6>(k)) *
              ((-50.) * (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
               (50.) * (ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667))))) *
              (RB(powr<2>(k), powr<2>(l1)))) *
         ((powr<-1>(powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
          ((powr<-1>(powr<2>(l1) +
                     (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p) +
                     powr<2>(p))) *
           ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) *
                          (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                      (powr<2>(l1)) * (ZA(l1)))) *
            ((powr<-1>(
                 (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))) *
                     (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                 (powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)) *
                     (ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)))))) *
             ((powr<-1>(
                  (RB(powr<2>(k), powr<2>(l1) +
                                      (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                      powr<2>(p))) *
                      (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                  (powr<2>(l1) +
                   (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p) +
                   powr<2>(p)) *
                      (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                               powr<2>(p)))))) *
              ((ZA3((0.816496580927726) *
                    (sqrt(powr<2>(l1) + (-1.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))))) *
               ((ZA3((0.816496580927726) *
                     (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                           powr<2>(p))))) *
                (ZA4((0.7071067811865475) *
                     (sqrt(powr<2>(l1) +
                           (-1.) * (cosl1p1 + (2.) * (cosl1p2) + cosl1p3) * ((l1) * (p)) +
                           (2.) * (powr<2>(p)))))))))))))) +
    (-0.04081632653061224) *
        ((1080. + (243.) * (powr<4>(cosl1p1)) + (243.) * (powr<4>(cosl1p2)) +
          (243.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (-1890.) * (powr<2>(cosl1p3)) +
          (-486.) * (powr<4>(cosl1p3)) +
          ((729.0000000000001) * (cosl1p2) + (243.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
          (-567. + (-243.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
          (-567. + (486.) * (powr<2>(cosl1p2)) + (405.) * ((cosl1p2) * (cosl1p3)) +
           (-243.) * (powr<2>(cosl1p3))) *
              (powr<2>(cosl1p1)) +
          ((-1890.) * (cosl1p3) + (-972.) * (powr<3>(cosl1p3))) * (cosl1p2) +
          ((729.0000000000001) * (powr<3>(cosl1p2)) + (-1890.) * (cosl1p3) +
           (405.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-972.) * (powr<3>(cosl1p3)) +
           (-2916. + (-810.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
              (cosl1p1)) *
             (powr<6>(l1)) +
         (powr<5>(l1)) *
             ((-729.0000000000001) * (powr<5>(cosl1p1)) +
              (-729.0000000000001) * (powr<5>(cosl1p2)) +
              ((-2916.) * (cosl1p2) + (-1215.) * (cosl1p3)) * (powr<4>(cosl1p1)) +
              (-504.) * (cosl1p3) + (-1215.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
              (-3348.) * (powr<3>(cosl1p3)) + (972.) * (powr<5>(cosl1p3)) +
              (2592. + (243.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
              (2592. + (-3645.) * (powr<2>(cosl1p2)) + (-3402.) * ((cosl1p2) * (cosl1p3)) +
               (243.) * (powr<2>(cosl1p3))) *
                  (powr<3>(cosl1p1)) +
              ((6210.) * (cosl1p3) + (3402.) * (powr<3>(cosl1p3))) * (powr<2>(cosl1p2)) +
              (-3851.999999999999 + (108.) * (powr<2>(cosl1p3)) + (3402.) * (powr<4>(cosl1p3))) *
                  (cosl1p2) +
              ((-3645.) * (powr<3>(cosl1p2)) + (6210.) * (cosl1p3) +
               (-3402.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (3402.) * (powr<3>(cosl1p3)) +
               (18576. + (2349.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (powr<2>(cosl1p1)) +
              (-3851.999999999999 + (-2916.) * (powr<4>(cosl1p2)) +
               (-3402.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (108.) * (powr<2>(cosl1p3)) +
               (3402.) * (powr<4>(cosl1p3)) +
               (18576. + (2349.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
               ((22248.) * (cosl1p3) + (7452.000000000001) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                  (cosl1p1)) *
             (p) +
         (powr<4>(l1)) *
             (9144. + (486.) * (powr<6>(cosl1p1)) + (486.) * (powr<6>(cosl1p2)) +
              (1215.) * ((powr<5>(cosl1p2)) * (cosl1p3)) + (-11898.) * (powr<2>(cosl1p3)) +
              (9666.) * (powr<4>(cosl1p3)) + (-486.) * (powr<6>(cosl1p3)) +
              ((2430.) * (cosl1p2) + (1215.) * (cosl1p3)) * (powr<5>(cosl1p1)) +
              (-3159. + (486.) * (powr<2>(cosl1p3))) * (powr<4>(cosl1p2)) +
              (-3159. + (4374.) * (powr<2>(cosl1p2)) + (4698.) * ((cosl1p2) * (cosl1p3)) +
               (486.) * (powr<2>(cosl1p3))) *
                  (powr<4>(cosl1p1)) +
              ((-3807.) * (cosl1p3) + (-2430.) * (powr<3>(cosl1p3))) * (powr<3>(cosl1p2)) +
              (-4257. + (18171.) * (powr<2>(cosl1p3)) + (-4131.) * (powr<4>(cosl1p3))) *
                  (powr<2>(cosl1p2)) +
              ((-18450.) * (cosl1p3) + (27108.) * (powr<3>(cosl1p3)) +
               (-2430.) * (powr<5>(cosl1p3))) *
                  (cosl1p2) +
              ((4860.) * (powr<3>(cosl1p2)) + (-3807.) * (cosl1p3) +
               (6561.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-2430.) * (powr<3>(cosl1p3)) +
               (-31752. + (81.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (powr<3>(cosl1p1)) +
              (-4257. + (4374.) * (powr<4>(cosl1p2)) + (6561.) * ((powr<3>(cosl1p2)) * (cosl1p3)) +
               (18171.) * (powr<2>(cosl1p3)) + (-4131.) * (powr<4>(cosl1p3)) +
               (-57510. + (-1296.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
               ((-44307.) * (cosl1p3) + (-8586.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                  (powr<2>(cosl1p1)) +
              ((2430.) * (powr<5>(cosl1p2)) + (-18450.) * (cosl1p3) +
               (4698.) * ((powr<4>(cosl1p2)) * (cosl1p3)) + (27108.) * (powr<3>(cosl1p3)) +
               (-2430.) * (powr<5>(cosl1p3)) +
               (-31752. + (81.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
               ((-44307.) * (cosl1p3) + (-8586.) * (powr<3>(cosl1p3))) * (powr<2>(cosl1p2)) +
               (-13968. + (24192.) * (powr<2>(cosl1p3)) + (-8586.) * (powr<4>(cosl1p3))) *
                   (cosl1p2)) *
                  (cosl1p1)) *
             (powr<2>(p)) +
         (powr<3>(l1)) *
             ((648.) * (powr<5>(cosl1p1)) + (648.) * (powr<5>(cosl1p2)) +
              ((12366.) * (cosl1p2) + (-2349.) * (cosl1p3)) * (powr<4>(cosl1p1)) +
              (-5760.) * (cosl1p3) + (-2349.) * ((powr<4>(cosl1p2)) * (cosl1p3)) +
              (-4824.) * (powr<3>(cosl1p3)) + (-324.) * (powr<5>(cosl1p3)) +
              (23949. + (-18549.) * (powr<2>(cosl1p3))) * (powr<3>(cosl1p2)) +
              (23949. + (34182.) * (powr<2>(cosl1p2)) + (14148.) * ((cosl1p2) * (cosl1p3)) +
               (-18549.) * (powr<2>(cosl1p3))) *
                  (powr<3>(cosl1p1)) +
              ((57618.00000000001) * (cosl1p3) + (-26406.) * (powr<3>(cosl1p3))) *
                  (powr<2>(cosl1p2)) +
              (-25200. + (25146.) * (powr<2>(cosl1p3)) + (-11178.) * (powr<4>(cosl1p3))) *
                  (cosl1p2) +
              ((34182.) * (powr<3>(cosl1p2)) + (57618.00000000001) * (cosl1p3) +
               (33318.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-26406.) * (powr<3>(cosl1p3)) +
               (88785. + (-37557.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (powr<2>(cosl1p1)) +
              (-25200. + (12366.) * (powr<4>(cosl1p2)) +
               (14148.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (25146.) * (powr<2>(cosl1p3)) +
               (-11178.) * (powr<4>(cosl1p3)) +
               (88785. + (-37557.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
               ((128772.) * (cosl1p3) + (-49140.) * (powr<3>(cosl1p3))) * (cosl1p2)) *
                  (cosl1p1)) *
             (powr<3>(p)) +
         (powr<2>(l1)) *
             (13716. + (-17172.) * (powr<4>(cosl1p1)) + (-17172.) * (powr<4>(cosl1p2)) +
              ((-78840.) * (cosl1p2) + (-34677.) * (cosl1p3)) * (powr<3>(cosl1p1)) +
              (-34677.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (-10236.) * (powr<2>(cosl1p3)) +
              (4734.) * (powr<4>(cosl1p3)) +
              (1464. + (-8595.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
              (1464. + (-123336.) * (powr<2>(cosl1p2)) + (-121473.) * ((cosl1p2) * (cosl1p3)) +
               (-8595.) * (powr<2>(cosl1p3))) *
                  (powr<2>(cosl1p1)) +
              ((-22596.) * (cosl1p3) + (13644.) * (powr<3>(cosl1p3))) * (cosl1p2) +
              ((-78840.) * (powr<3>(cosl1p2)) + (-22596.) * (cosl1p3) +
               (-121473.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (13644.) * (powr<3>(cosl1p3)) +
               (-132. + (-24480.) * (powr<2>(cosl1p3))) * (cosl1p2)) *
                  (cosl1p1)) *
             (powr<4>(p)) +
         (l1) *
             ((21738.) * (powr<3>(cosl1p1)) + (21738.) * (powr<3>(cosl1p2)) + (-3112.) * (cosl1p3) +
              (36030.) * ((powr<2>(cosl1p2)) * (cosl1p3)) + (-588.) * (powr<3>(cosl1p3)) +
              ((67914.) * (cosl1p2) + (36030.) * (cosl1p3)) * (powr<2>(cosl1p1)) +
              (-18052. + (13704.) * (powr<2>(cosl1p3))) * (cosl1p2) +
              (-18052. + (67914.) * (powr<2>(cosl1p2)) +
               (74760.00000000001) * ((cosl1p2) * (cosl1p3)) + (13704.) * (powr<2>(cosl1p3))) *
                  (cosl1p1)) *
             (powr<5>(p)) +
         (4672. + (-6744.) * (powr<2>(cosl1p1)) + (-13488.) * ((cosl1p1) * (cosl1p2)) +
          (-6744.) * (powr<2>(cosl1p2)) + (-7272.) * ((cosl1p1) * (cosl1p3)) +
          (-7272.) * ((cosl1p2) * (cosl1p3)) + (-527.9999999999999) * (powr<2>(cosl1p3))) *
             (powr<6>(p))) *
        ((powr<-1>(1. + powr<6>(k))) *
         ((dtZA(pow(1. + powr<6>(k), 0.16666666666666666667))) * (1. + (1.) * (powr<6>(k))) *
              (RB(powr<2>(k), powr<2>(l1))) +
          (RBdot(powr<2>(k), powr<2>(l1))) * (1. + powr<6>(k)) *
              (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
          (powr<6>(k)) *
              ((-50.) * (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
               (50.) * (ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667))))) *
              (RB(powr<2>(k), powr<2>(l1)))) *
         ((powr<-1>(powr<2>(l1) +
                    (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p) +
                    powr<2>(p))) *
          ((powr<-1>((3.) * (powr<2>(l1)) + (-6.) * ((cosl1p1) * ((l1) * (p))) +
                     (-6.) * ((cosl1p2) * ((l1) * (p))) + (4.) * (powr<2>(p)))) *
           ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) *
                          (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                      (powr<2>(l1)) * (ZA(l1)))) *
            ((powr<-1>(
                 (RB(powr<2>(k), powr<2>(l1) +
                                     (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                     powr<2>(p))) *
                     (ZA(pow(1. + powr<6>(k), 0.16666666666666666667))) +
                 (powr<2>(l1) +
                  (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p) +
                  powr<2>(p)) *
                     (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                              powr<2>(p)))))) *
             ((powr<-1>(
                  (3.) * ((RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                              (1.333333333333333) * (powr<2>(p)))) *
                          (ZA(pow(1. + powr<6>(k), 0.16666666666666666667)))) +
                  ((3.) * (powr<2>(l1)) + (-6.) * ((cosl1p1) * ((l1) * (p))) +
                   (-6.) * ((cosl1p2) * ((l1) * (p))) + (4.) * (powr<2>(p))) *
                      (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                               (1.333333333333333) * (powr<2>(p))))))) *
              ((ZA3((0.816496580927726) *
                    (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                          powr<2>(p))))) *
               ((ZA3((0.3333333333333333) *
                     (sqrt((6.) * (powr<2>(l1)) +
                           (-6.) * ((2.) * (cosl1p1) + (2.) * (cosl1p2) + cosl1p3) * ((l1) * (p)) +
                           (10.) * (powr<2>(p)))))) *
                (ZA4((0.408248290463863) *
                     (sqrt((3.) * (powr<2>(l1)) + (-3.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                           (5.) * (powr<2>(p)))))))))))))) +
    (0.01530612244897959) *
        ((-3.) *
             (-1304. + (54.) * (powr<4>(cosl1p1)) + (-27.) * (powr<4>(cosl1p2)) +
              (-81.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + (-399.) * (powr<2>(cosl1p3)) +
              (-27.) * (powr<4>(cosl1p3)) + (108.) * (cosl1p2 + cosl1p3) * (powr<3>(cosl1p1)) +
              (3.) *
                  (-262. + (9.) * (powr<2>(cosl1p2)) + (30.) * ((cosl1p2) * (cosl1p3)) +
                   (9.) * (powr<2>(cosl1p3))) *
                  (powr<2>(cosl1p1)) +
              (-3.) * (133. + (18.) * (powr<2>(cosl1p3))) * (powr<2>(cosl1p2)) +
              (-3.) * (4. + (27.) * (powr<2>(cosl1p3))) * ((cosl1p2) * (cosl1p3)) +
              (-3.) *
                  ((262.) * (cosl1p2) + (9.) * (powr<3>(cosl1p2)) + (262.) * (cosl1p3) +
                   (15.) * ((powr<2>(cosl1p2)) * (cosl1p3)) +
                   (15.) * ((cosl1p2) * (powr<2>(cosl1p3))) + (9.) * (powr<3>(cosl1p3))) *
                  (cosl1p1)) *
             (powr<2>(l1)) +
         (-9.) * (cosl1p2 + cosl1p3) *
             ((l1) *
              (872. + (282.) * (powr<2>(cosl1p1)) + (151.) * (powr<2>(cosl1p2)) +
               (20.) * ((cosl1p2) * (cosl1p3)) + (151.) * (powr<2>(cosl1p3)) +
               (282.) * (cosl1p2 + cosl1p3) * (cosl1p1)) *
              (p)) +
         (16.) *
             (382. + (57.) * (powr<2>(cosl1p1)) + (-9.) * (powr<2>(cosl1p2)) +
              (-75.) * ((cosl1p2) * (cosl1p3)) + (-9.) * (powr<2>(cosl1p3)) +
              (57.) * (cosl1p2 + cosl1p3) * (cosl1p1)) *
             (powr<2>(p))) *
        ((powr<-1>((3.) * (powr<2>(l1)) + (-6.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                   (4.) * (powr<2>(p)))) *
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
          ((powr<-1>((3.) *
                         ((RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                              (1.333333333333333) * (powr<2>(p)))) *
                          (ZA(pow(1. + powr<6>(k), 0.16666666666666666667)))) +
                     ((3.) * (powr<2>(l1)) + (-6.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                      (4.) * (powr<2>(p))) *
                         (ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                  (1.333333333333333) * (powr<2>(p))))))) *
           (powr<2>(ZA4((0.408248290463863) *
                        (sqrt((3.) * (powr<2>(l1)) + (-3.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                              (5.) * (powr<2>(p)))))))))) +
    (1.653061224489796) *
        ((1.) * ((powr<4>(cosl1p1)) * (powr<2>(l1))) +
         (-2.) * ((powr<4>(cosl1p2)) * (powr<2>(l1))) +
         (-0.3333333333333333) * ((powr<2>(cosl1p3)) * (powr<2>(l1))) +
         (1.) * ((powr<4>(cosl1p3)) * (powr<2>(l1))) +
         (powr<3>(cosl1p1)) *
             ((1.) * ((cosl1p2) * (l1)) + (3.) * ((cosl1p3) * (l1)) + (-1.333333333333333) * (p)) *
             (l1) +
         (0.4444444444444445) * ((cosl1p3) * ((l1) * (p))) +
         (-0.6666666666666666) * ((powr<3>(cosl1p3)) * ((l1) * (p))) +
         (-0.5925925925925926) * (powr<2>(p)) +
         (powr<3>(cosl1p2)) * ((-4.) * ((cosl1p3) * (l1)) + (2.) * (p)) * (l1) +
         ((0.6666666666666666 + (-1.) * (powr<2>(cosl1p3))) * (powr<2>(l1)) +
          (2.333333333333333) * ((cosl1p3) * ((l1) * (p))) + (0.3333333333333333) * (powr<2>(p))) *
             (powr<2>(cosl1p2)) +
         ((0.6666666666666666) * ((cosl1p3) * (powr<2>(l1))) +
          (1.) * ((powr<3>(cosl1p3)) * (powr<2>(l1))) + (-0.4444444444444445) * ((l1) * (p)) +
          (0.3333333333333333) * ((powr<2>(cosl1p3)) * ((l1) * (p))) +
          (0.2222222222222222) * ((cosl1p3) * (powr<2>(p)))) *
             (cosl1p2) +
         ((-0.3333333333333333 + (-1.) * (powr<2>(cosl1p2)) +
           (1.666666666666667) * ((cosl1p2) * (cosl1p3)) + (2.) * (powr<2>(cosl1p3))) *
              (powr<2>(l1)) +
          (1.592592592592593) * (powr<2>(p)) +
          ((-1.888888888888889) * ((cosl1p2) * (p)) + (-2.111111111111111) * ((cosl1p3) * (p))) *
              (l1)) *
             (powr<2>(cosl1p1)) +
         ((-4.) * ((powr<3>(cosl1p2)) * (powr<2>(l1))) +
          (powr<2>(cosl1p2)) *
              ((-3.333333333333333) * ((cosl1p3) * (l1)) + (0.7777777777777778) * (p)) * (l1) +
          ((-1.333333333333333 + (3.) * (powr<2>(cosl1p3))) * (powr<2>(l1)) +
           (0.5555555555555555) * ((cosl1p3) * ((l1) * (p))) + (1.259259259259259) * (powr<2>(p))) *
              (cosl1p3) +
          ((0.6666666666666666 + (1.666666666666667) * (powr<2>(cosl1p3))) * (powr<2>(l1)) +
           (1.925925925925926) * (powr<2>(p))) *
              (cosl1p2)) *
             (cosl1p1)) *
        ((powr<2>(l1)) *
         ((dtZc(k)) * (RB(powr<2>(k), powr<2>(l1))) + (RBdot(powr<2>(k), powr<2>(l1))) * (Zc(k)) +
          ((-50.) * (Zc(k)) + (50.) * (Zc((1.02) * (k)))) * (RB(powr<2>(k), powr<2>(l1)))) *
         ((ZAcbc((0.816496580927726) *
                 (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))))) *
          ((ZAcbc((0.816496580927726) *
                  (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                        powr<2>(p))))) *
           ((ZAcbc(
                (0.3333333333333333) *
                (sqrt((6.) * (powr<2>(l1)) + (-6.) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) +
                      (10.) * (powr<2>(p)))))) *
            ((ZAcbc((0.3333333333333333) *
                    (sqrt((6.) * (powr<2>(l1)) +
                          (-6.) * ((2.) * (cosl1p1) + (2.) * (cosl1p2) + cosl1p3) * ((l1) * (p)) +
                          (10.) * (powr<2>(p)))))) *
             ((powr<-2>((RB(powr<2>(k), powr<2>(l1))) * (Zc(k)) + (powr<2>(l1)) * (Zc(l1)))) *
              ((powr<-1>(
                   (RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))) *
                       (Zc(k)) +
                   (powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)) *
                       (Zc(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)))))) *
               ((powr<-1>(
                    (RB(powr<2>(k), powr<2>(l1) +
                                        (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                        powr<2>(p))) *
                        (Zc(k)) +
                    (powr<2>(l1) +
                     (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p) +
                     powr<2>(p)) *
                        (Zc(sqrt(powr<2>(l1) +
                                 (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                 powr<2>(p)))))) *
                (powr<-1>((3.) * ((RB(powr<2>(k), powr<2>(l1) +
                                                      (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                                      (1.333333333333333) * (powr<2>(p)))) *
                                  (Zc(k))) +
                          ((3.) * (powr<2>(l1)) + (-6.) * ((cosl1p1) * ((l1) * (p))) +
                           (-6.) * ((cosl1p2) * ((l1) * (p))) + (4.) * (powr<2>(p))) *
                              (Zc(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) +
                                       (1.333333333333333) * (powr<2>(p))))))))))))))));
      // clang-format on
      const auto _interp1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = RB(powr<2>(k), fma(-2., (cosl1p1) * ((l1) * (p)), powr<2>(l1) + powr<2>(p)));
      const auto _interp9 =
          RB(powr<2>(k), fma(-2., (l1) * (cosl1p1 + cosl1p2 + cosl1p3) * (p), powr<2>(l1) + powr<2>(p)));
      const auto _interp10 = ZA(sqrt(fma(-2., (l1) * (cosl1p1 + cosl1p2 + cosl1p3) * (p), powr<2>(l1) + powr<2>(p))));
      const auto _interp11 =
          RB(powr<2>(k), fma(-2., (l1) * (cosl1p1 + cosl1p2) * (p), fma(1.333333333333333, powr<2>(p), powr<2>(l1))));
      const auto _interp12 =
          ZA(sqrt(fma(-2., (l1) * (cosl1p1 + cosl1p2) * (p), fma(1.333333333333333, powr<2>(p), powr<2>(l1)))));
      const auto _interp14 = ZA3(
          (0.816496580927726) * (sqrt(fma(-1., (l1) * (cosl1p1 + cosl1p2 + cosl1p3) * (p), powr<2>(l1) + powr<2>(p)))));
      const auto _interp16 =
          ZA3((0.3333333333333333) *
              (sqrt(fma(6., powr<2>(l1),
                        fma(-6., (l1) * ((cosl1p1 + cosl1p2) * (2.) + cosl1p3) * (p), fma(10., powr<2>(p), 0.))))));
      _T _acc{};
      { // subkernel 1
        const auto _interp17 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p));
        const auto _interp18 = ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p)));
        const auto _interp19 =
            ZA3((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p2) * ((l1) * (p))) + powr<2>(p))));
        const auto _interp20 = ZA4(
            (0.7071067811865475) *
            (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + (2.) * (cosl1p2) + cosl1p3) * ((l1) * (p)) + (2.) * (powr<2>(p)))));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = powr<2>(p);
        const auto _cse3 = powr<2>(cosl1p3);
        const auto _cse4 = powr<2>(cosl1p2);
        const auto _cse5 = powr<3>(cosl1p2);
        const auto _cse6 = powr<3>(cosl1p3);
        const auto _cse7 = powr<4>(cosl1p2);
        const auto _cse8 = powr<4>(cosl1p1);
        const auto _cse9 = powr<3>(cosl1p1);
        const auto _cse10 = (27.) * (_cse4);
        const auto _cse11 = powr<4>(cosl1p3);
        const auto _cse12 = powr<2>(cosl1p1);
        const auto _cse13 = powr<5>(cosl1p1);
        const auto _cse14 = powr<5>(cosl1p2);
        const auto _cse15 = (135.) * (_cse3);
        _acc +=
            (0.02040816326530612) *
            ((_interp14) *
             ((_interp19) *
              ((_interp20) *
               ((powr<-1>(1. + powr<6>(k))) *
                ((powr<-1>(fma(-2., (cosl1p2) * ((l1) * (p)), _cse1 + _cse2))) *
                 ((fma(_cse2,
                       (-1072. + (-99.) * (_cse11) + (402.) * (_cse3) +
                        (1576. + (-54.) * (_cse11) + (3579.) * (_cse3)) * (_cse4) + (3186. + _cse15) * (_cse7) +
                        (-27.) * ((_cse13) * (cosl1p2)) + (54.) * (powr<6>(cosl1p2)) + (162.) * ((_cse14) * (cosl1p3)) +
                        (6372.000000000001) * ((_cse5) * (cosl1p3)) +
                        ((393. + (-135.) * (_cse3)) * (cosl1p2) + (-630.) * (cosl1p3) +
                         (-153.) * ((_cse4) * (cosl1p3))) *
                            (_cse9) +
                        (-99. + (-54.) * (_cse4) + (-108.) * ((cosl1p2) * (cosl1p3))) * (_cse8) +
                        ((393.) * (_cse6) + (1576.) * (cosl1p3) + (-27.) * (powr<5>(cosl1p3))) * (cosl1p2) +
                        (402. + (-1062.) * (_cse3) + (3579. + (-144.) * (_cse3)) * (_cse4) + (135.) * (_cse7) +
                         (72.) * ((_cse5) * (cosl1p3)) + ((-135.) * (_cse6) + (609.) * (cosl1p3)) * (cosl1p2)) *
                            (_cse12) +
                        ((162.) * (_cse14) + (6372.000000000001 + (72.) * (_cse3)) * (_cse5) + (-630.) * (_cse6) +
                         (1576. + (-108.) * (_cse11) + (609.) * (_cse3)) * (cosl1p2) + (876.) * (cosl1p3) +
                         (306.) * ((_cse7) * (cosl1p3)) +
                         ((-153.) * (_cse6) + (6587.999999999999) * (cosl1p3)) * (_cse4)) *
                            (cosl1p1)) *
                           (powr<4>(l1)),
                       fma(-120. + (-27.) * (_cse11) + (63.) * (_cse3) + (210. + (27.) * (_cse3)) * (_cse4) +
                               (54.) * (_cse7) + (-27.) * (_cse8) +
                               ((-27.) * (cosl1p2) + (-81.) * (cosl1p3)) * (_cse9) + (108.) * ((_cse5) * (cosl1p3)) +
                               ((-27.) * (_cse6) + (210.) * (cosl1p3)) * (cosl1p2) +
                               ((108.) * (_cse5) + (-81.) * (_cse6) + (210. + (-45.) * (_cse3)) * (cosl1p2) +
                                (324.) * (cosl1p3) + (90.) * ((_cse4) * (cosl1p3))) *
                                   (cosl1p1) +
                               (63. + _cse10 + (-54.) * (_cse3) + (-45.) * ((cosl1p2) * (cosl1p3))) * (_cse12),
                           powr<6>(l1),
                           fma((27.) * (_cse13) + (-108.) * (_cse14) + (-1776. + (-162.) * (_cse3)) * (_cse5) +
                                   (-18.) * (_cse6) + (624. + (81.) * (_cse11) + (-924.) * (_cse3)) * (cosl1p2) +
                                   ((27.) * (_cse6) + (-2664.) * (cosl1p3)) * (_cse4) + (312.) * (cosl1p3) +
                                   (-270.) * ((_cse7) * (cosl1p3)) + (27.) * (powr<5>(cosl1p3)) +
                                   ((81.) * (cosl1p2) + (108.) * (cosl1p3)) * (_cse8) +
                                   ((-162.) * (_cse5) + (135.) * (_cse6) + (-924. + (198.) * (_cse3)) * (cosl1p2) +
                                    (-126.) * (cosl1p3) + (-27.) * ((_cse4) * (cosl1p3))) *
                                       (_cse12) +
                                   (312. + (108.) * (_cse11) + (-126.) * (_cse3) +
                                    (-2664. + (-27.) * (_cse3)) * (_cse4) + (-270.) * (_cse7) +
                                    ((234.) * (_cse6) + (-1992.) * (cosl1p3)) * (cosl1p2) +
                                    (-396.) * ((_cse5) * (cosl1p3))) *
                                       (cosl1p1) +
                                   (-18. + _cse10 + _cse15 + (234.) * ((cosl1p2) * (cosl1p3))) * (_cse9),
                               (powr<5>(l1)) * (p),
                               fma((-1164.) * (_cse14) + (-7447.999999999999 + (-2130.) * (_cse3)) * (_cse5) +
                                       (-294.) * (_cse6) +
                                       (2808. + (99.) * (_cse11) + (-4311.999999999999) * (_cse3)) * (cosl1p2) +
                                       (99.) * ((_cse8) * (cosl1p2)) +
                                       ((-285.) * (_cse6) + (-11172.) * (cosl1p3)) * (_cse4) + (1404.) * (cosl1p3) +
                                       (-2910.) * ((_cse7) * (cosl1p3)) +
                                       ((-2130.) * (_cse5) + (-4311.999999999999 + (1062.) * (_cse3)) * (cosl1p2) +
                                        (-798.) * (cosl1p3) + (-153.) * ((_cse4) * (cosl1p3))) *
                                           (_cse12) +
                                       (1404. + (-798.) * (_cse3) + (-11172. + (-153.) * (_cse3)) * (_cse4) +
                                        (-2910.) * (_cse7) + ((630.) * (_cse6) + (-8456.) * (cosl1p3)) * (cosl1p2) +
                                        (-3792.) * ((_cse5) * (cosl1p3))) *
                                           (cosl1p1) +
                                       (-294. + (-285.) * (_cse4) + (630.) * ((cosl1p2) * (cosl1p3))) * (_cse9),
                                   (powr<3>(l1)) * (powr<3>(p)),
                                   fma(_cse1,
                                       (-1364. + (406.) * (_cse3) + (1816. + (5232.000000000001) * (_cse3)) * (_cse4) +
                                        (5028.) * (_cse7) + (204.) * ((_cse9) * (cosl1p2)) +
                                        (10056.) * ((_cse5) * (cosl1p3)) +
                                        ((204.) * (_cse6) + (1816.) * (cosl1p3)) * (cosl1p2) +
                                        ((10056.) * (_cse5) + (1816. + (468.) * (_cse3)) * (cosl1p2) +
                                         (764.0000000000001) * (cosl1p3) + (10320.) * ((_cse4) * (cosl1p3))) *
                                            (cosl1p1) +
                                        (406. + (5232.000000000001) * (_cse4) + (468.) * ((cosl1p2) * (cosl1p3))) *
                                            (_cse12)) *
                                           (powr<4>(p)),
                                       fma((-4632.) * (_cse5) + (-2316.) * ((_cse12) * (cosl1p2)) +
                                               (1544. + (-2316.) * (_cse3)) * (cosl1p2) +
                                               (772.0000000000001) * (cosl1p3) +
                                               (-6947.999999999999) * ((_cse4) * (cosl1p3)) +
                                               (772.0000000000001 + (-6947.999999999999) * (_cse4) +
                                                (-4632.) * ((cosl1p2) * (cosl1p3))) *
                                                   (cosl1p1),
                                           (l1) * (powr<5>(p)),
                                           fma(-392. + (1176.) * (_cse4) + (1176.) * ((cosl1p1) * (cosl1p2)) +
                                                   (1176.) * ((cosl1p2) * (cosl1p3)),
                                               powr<6>(p), 0.)))))))) *
                  ((powr<-1>(fma(_interp17, _interp4,
                                 fma(_interp18, _cse1 + _cse2 + (-2.) * ((cosl1p2) * ((l1) * (p))), 0.)))) *
                   ((powr<-2>(fma(_interp2, _interp4, fma(_cse1, _interp6, 0.)))) *
                    ((fma(_interp2, ((-50.) * (_interp4) + (50.) * (_interp5)) * (powr<6>(k)),
                          fma(_interp3, (1. + powr<6>(k)) * (_interp4),
                              fma(_interp1, (1. + (1.) * (powr<6>(k))) * (_interp2), 0.)))) *
                     ((powr<-1>(fma(_interp4, _interp9,
                                    fma(_interp10,
                                        _cse1 + _cse2 +
                                            (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p),
                                        0.)))) *
                      (powr<-1>(fma((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3), (l1) * (p),
                                    _cse1 + _cse2)))))))))))));
      }
      { // subkernel 2
        const auto _interp22 = RB(powr<2>(k), powr<2>(l1) + (-2.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) +
                                                  (1.333333333333333) * (powr<2>(p)));
        const auto _interp23 =
            ZA(sqrt(powr<2>(l1) + (-2.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) + (1.333333333333333) * (powr<2>(p))));
        const auto _interp24 =
            ZA4((0.408248290463863) *
                (sqrt((3.) * (powr<2>(l1)) + (-3.) * (cosl1p2 + cosl1p3) * ((l1) * (p)) + (5.) * (powr<2>(p)))));
        const auto _cse1 = cosl1p2 + cosl1p3;
        const auto _cse2 = powr<2>(cosl1p3);
        const auto _cse3 = powr<2>(cosl1p2);
        const auto _cse4 = powr<3>(cosl1p2);
        const auto _cse5 = powr<2>(l1);
        const auto _cse6 = powr<2>(cosl1p1);
        const auto _cse7 = powr<2>(p);
        const auto _cse8 = powr<6>(k);
        const auto _cse9 = (3.) * (_cse5);
        const auto _cse10 = (-6.) * ((_cse1) * ((l1) * (p)));
        const auto _cse11 = (4.) * (_cse7);
        const auto _cse12 = _cse10 + _cse11 + _cse9;
        _acc +=
            (0.01530612244897959) *
            ((powr<-1>(_cse12)) *
             ((powr<2>(_interp24)) *
              ((fma(16.,
                    (382. + (-9.) * (_cse2) + (-9.) * (_cse3) + (57.) * (_cse6) + (57.) * ((_cse1) * (cosl1p1)) +
                     (-75.) * ((cosl1p2) * (cosl1p3))) *
                        (_cse7),
                    fma(-3.,
                        (-1304. + (-399.) * (_cse2) + (-3.) * (133. + (18.) * (_cse2)) * (_cse3) +
                         (108.) * ((_cse1) * (powr<3>(cosl1p1))) + (54.) * (powr<4>(cosl1p1)) +
                         (-27.) * (powr<4>(cosl1p2)) + (-81.) * ((_cse4) * (cosl1p3)) +
                         (-3.) * (4. + (27.) * (_cse2)) * ((cosl1p2) * (cosl1p3)) + (-27.) * (powr<4>(cosl1p3)) +
                         (3.) * (-262. + (9.) * (_cse2) + (9.) * (_cse3) + (30.) * ((cosl1p2) * (cosl1p3))) * (_cse6) +
                         (-3.) *
                             ((9.) * (_cse4) + (262.) * (cosl1p2) + (15.) * ((_cse2) * (cosl1p2)) + (262.) * (cosl1p3) +
                              (15.) * ((_cse3) * (cosl1p3)) + (9.) * (powr<3>(cosl1p3))) *
                             (cosl1p1)) *
                            (_cse5),
                        fma(-9.,
                            (_cse1) *
                                (872. + (151.) * (_cse2) + (151.) * (_cse3) + (282.) * (_cse6) +
                                 (282.) * ((_cse1) * (cosl1p1)) + (20.) * ((cosl1p2) * (cosl1p3))) *
                                ((l1) * (p)),
                            0.)))) *
               ((powr<-1>(fma(_cse12, _interp23, fma(3., (_interp22) * (_interp4), 0.)))) *
                ((powr<-2>(fma(_interp2, _interp4, fma(_cse5, _interp6, 0.)))) *
                 (fma(_interp3, _interp4,
                      fma(_interp2,
                          _interp1 + (50.) * ((-1.) * (_interp4) + _interp5) * ((_cse8) * (powr<-1>(1. + _cse8))),
                          0.))))))));
      }
      { // subkernel 3
        const auto _interp21 =
            ZA4((0.408248290463863) *
                (sqrt((3.) * (powr<2>(l1)) + (-3.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + (5.) * (powr<2>(p)))));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = powr<2>(p);
        const auto _cse3 = powr<2>(cosl1p3);
        const auto _cse4 = powr<2>(cosl1p2);
        const auto _cse5 = (-243.) * (_cse3);
        const auto _cse6 = powr<3>(cosl1p2);
        const auto _cse7 = (-1890.) * (cosl1p3);
        const auto _cse8 = powr<3>(cosl1p3);
        const auto _cse9 = (-972.) * (_cse8);
        const auto _cse10 = powr<4>(cosl1p1);
        const auto _cse11 = powr<4>(cosl1p2);
        const auto _cse12 = powr<3>(cosl1p1);
        const auto _cse13 = (243.) * (_cse3);
        const auto _cse14 = powr<4>(cosl1p3);
        const auto _cse15 = powr<2>(cosl1p1);
        const auto _cse16 = (6210.) * (cosl1p3);
        const auto _cse17 = (3402.) * (_cse8);
        const auto _cse18 = (108.) * (_cse3);
        _acc +=
            (-0.04081632653061224) *
            ((_interp14) *
             ((_interp16) *
              ((_interp21) *
               ((powr<-1>(1. + powr<6>(k))) *
                ((powr<-1>(
                     fma(3., _cse1,
                         fma(4., _cse2, fma(-6., (cosl1p1) * ((l1) * (p)), fma(-6., (cosl1p2) * ((l1) * (p)), 0.)))))) *
                 ((powr<-1>(fma(3., (_interp11) * (_interp4),
                                fma(_interp12,
                                    (3.) * (_cse1) + (4.) * (_cse2) + (-6.) * ((cosl1p1) * ((l1) * (p))) +
                                        (-6.) * ((cosl1p2) * ((l1) * (p))),
                                    0.)))) *
                  ((fma(
                       _cse2,
                       (9144. + (9666.) * (_cse14) + (-11898.) * (_cse3) + (-3159. + (486.) * (_cse3)) * (_cse11) +
                        (-4257. + (-4131.) * (_cse14) + (18171.) * (_cse3)) * (_cse4) + (486.) * (powr<6>(cosl1p1)) +
                        (486.) * (powr<6>(cosl1p2)) + ((-2430.) * (_cse8) + (-3807.) * (cosl1p3)) * (_cse6) +
                        (1215.) * ((powr<5>(cosl1p2)) * (cosl1p3)) + (-486.) * (powr<6>(cosl1p3)) +
                        ((2430.) * (cosl1p2) + (1215.) * (cosl1p3)) * (powr<5>(cosl1p1)) +
                        ((4860.) * (_cse6) + (-2430.) * (_cse8) + (-31752. + (81.) * (_cse3)) * (cosl1p2) +
                         (-3807.) * (cosl1p3) + (6561.) * ((_cse4) * (cosl1p3))) *
                            (_cse12) +
                        (-4257. + (4374.) * (_cse11) + (-4131.) * (_cse14) + (18171.) * (_cse3) +
                         (-57510. + (-1296.) * (_cse3)) * (_cse4) +
                         ((-8586.) * (_cse8) + (-44307.) * (cosl1p3)) * (cosl1p2) + (6561.) * ((_cse6) * (cosl1p3))) *
                            (_cse15) +
                        (-3159. + (486.) * (_cse3) + (4374.) * (_cse4) + (4698.) * ((cosl1p2) * (cosl1p3))) * (_cse10) +
                        ((27108.) * (_cse8) + (-18450.) * (cosl1p3) + (-2430.) * (powr<5>(cosl1p3))) * (cosl1p2) +
                        ((-31752. + (81.) * (_cse3)) * (_cse6) + (27108.) * (_cse8) +
                         (-13968. + (-8586.) * (_cse14) + (24192.) * (_cse3)) * (cosl1p2) +
                         (2430.) * (powr<5>(cosl1p2)) + ((-8586.) * (_cse8) + (-44307.) * (cosl1p3)) * (_cse4) +
                         (-18450.) * (cosl1p3) + (4698.) * ((_cse11) * (cosl1p3)) + (-2430.) * (powr<5>(cosl1p3))) *
                            (cosl1p1)) *
                           (powr<4>(l1)),
                       fma(1080. + (243.) * (_cse10) + (243.) * (_cse11) + (-486.) * (_cse14) + (-1890.) * (_cse3) +
                               (-567. + _cse5) * (_cse4) + (_cse7 + _cse9) * (cosl1p2) +
                               (243.) * ((_cse6) * (cosl1p3)) +
                               ((729.0000000000001) * (cosl1p2) + (243.) * (cosl1p3)) * (_cse12) +
                               ((729.0000000000001) * (_cse6) + _cse7 + _cse9 +
                                (-2916. + (-810.) * (_cse3)) * (cosl1p2) + (405.) * ((_cse4) * (cosl1p3))) *
                                   (cosl1p1) +
                               (-567. + (486.) * (_cse4) + _cse5 + (405.) * ((cosl1p2) * (cosl1p3))) * (_cse15),
                           powr<6>(l1),
                           fma((_cse16 + _cse17) * (_cse4) + (2592. + _cse13) * (_cse6) + (-3348.) * (_cse8) +
                                   (-729.0000000000001) * (powr<5>(cosl1p1)) +
                                   (-3851.999999999999 + (3402.) * (_cse14) + _cse18) * (cosl1p2) +
                                   (-729.0000000000001) * (powr<5>(cosl1p2)) +
                                   ((-2916.) * (cosl1p2) + (-1215.) * (cosl1p3)) * (_cse10) + (-504.) * (cosl1p3) +
                                   (-1215.) * ((_cse11) * (cosl1p3)) + (972.) * (powr<5>(cosl1p3)) +
                                   (_cse16 + _cse17 + (-3645.) * (_cse6) + (18576. + (2349.) * (_cse3)) * (cosl1p2) +
                                    (-3402.) * ((_cse4) * (cosl1p3))) *
                                       (_cse15) +
                                   (2592. + _cse13 + (-3645.) * (_cse4) + (-3402.) * ((cosl1p2) * (cosl1p3))) *
                                       (_cse12) +
                                   (-3851.999999999999 + (-2916.) * (_cse11) + (3402.) * (_cse14) + _cse18 +
                                    (18576. + (2349.) * (_cse3)) * (_cse4) + (-3402.) * ((_cse6) * (cosl1p3)) +
                                    ((7452.000000000001) * (_cse8) + (22248.) * (cosl1p3)) * (cosl1p2)) *
                                       (cosl1p1),
                               (powr<5>(l1)) * (p),
                               fma((23949. + (-18549.) * (_cse3)) * (_cse6) + (-4824.) * (_cse8) +
                                       (648.) * (powr<5>(cosl1p1)) +
                                       (-25200. + (-11178.) * (_cse14) + (25146.) * (_cse3)) * (cosl1p2) +
                                       (648.) * (powr<5>(cosl1p2)) +
                                       ((12366.) * (cosl1p2) + (-2349.) * (cosl1p3)) * (_cse10) + (-5760.) * (cosl1p3) +
                                       (-2349.) * ((_cse11) * (cosl1p3)) + (-324.) * (powr<5>(cosl1p3)) +
                                       ((-26406.) * (_cse8) + (57618.00000000001) * (cosl1p3)) * (_cse4) +
                                       ((34182.) * (_cse6) + (-26406.) * (_cse8) +
                                        (88785. + (-37557.) * (_cse3)) * (cosl1p2) + (57618.00000000001) * (cosl1p3) +
                                        (33318.) * ((_cse4) * (cosl1p3))) *
                                           (_cse15) +
                                       (23949. + (-18549.) * (_cse3) + (34182.) * (_cse4) +
                                        (14148.) * ((cosl1p2) * (cosl1p3))) *
                                           (_cse12) +
                                       (-25200. + (12366.) * (_cse11) + (-11178.) * (_cse14) + (25146.) * (_cse3) +
                                        (88785. + (-37557.) * (_cse3)) * (_cse4) + (14148.) * ((_cse6) * (cosl1p3)) +
                                        ((-49140.) * (_cse8) + (128772.) * (cosl1p3)) * (cosl1p2)) *
                                           (cosl1p1),
                                   (powr<3>(l1)) * (powr<3>(p)),
                                   fma(_cse1,
                                       (13716. + (-17172.) * (_cse10) + (-17172.) * (_cse11) + (4734.) * (_cse14) +
                                        (-10236.) * (_cse3) + (1464. + (-8595.) * (_cse3)) * (_cse4) +
                                        ((-78840.) * (cosl1p2) + (-34677.) * (cosl1p3)) * (_cse12) +
                                        ((13644.) * (_cse8) + (-22596.) * (cosl1p3)) * (cosl1p2) +
                                        (-34677.) * ((_cse6) * (cosl1p3)) +
                                        ((-78840.) * (_cse6) + (13644.) * (_cse8) +
                                         (-132. + (-24480.) * (_cse3)) * (cosl1p2) + (-22596.) * (cosl1p3) +
                                         (-121473.) * ((_cse4) * (cosl1p3))) *
                                            (cosl1p1) +
                                        (1464. + (-8595.) * (_cse3) + (-123336.) * (_cse4) +
                                         (-121473.) * ((cosl1p2) * (cosl1p3))) *
                                            (_cse15)) *
                                           (powr<4>(p)),
                                       fma((21738.) * (_cse12) + (21738.) * (_cse6) + (-588.) * (_cse8) +
                                               (-18052. + (13704.) * (_cse3)) * (cosl1p2) + (-3112.) * (cosl1p3) +
                                               (36030.) * ((_cse4) * (cosl1p3)) +
                                               ((67914.) * (cosl1p2) + (36030.) * (cosl1p3)) * (_cse15) +
                                               (-18052. + (13704.) * (_cse3) + (67914.) * (_cse4) +
                                                (74760.00000000001) * ((cosl1p2) * (cosl1p3))) *
                                                   (cosl1p1),
                                           (l1) * (powr<5>(p)),
                                           fma(4672. + (-6744.) * (_cse15) + (-527.9999999999999) * (_cse3) +
                                                   (-6744.) * (_cse4) + (-13488.) * ((cosl1p1) * (cosl1p2)) +
                                                   (-7272.) * ((cosl1p1) * (cosl1p3)) +
                                                   (-7272.) * ((cosl1p2) * (cosl1p3)),
                                               powr<6>(p), 0.)))))))) *
                   ((powr<-2>(fma(_interp2, _interp4, fma(_cse1, _interp6, 0.)))) *
                    ((fma(_interp2, ((-50.) * (_interp4) + (50.) * (_interp5)) * (powr<6>(k)),
                          fma(_interp3, (1. + powr<6>(k)) * (_interp4),
                              fma(_interp1, (1. + (1.) * (powr<6>(k))) * (_interp2), 0.)))) *
                     ((powr<-1>(fma(_interp4, _interp9,
                                    fma(_interp10,
                                        _cse1 + _cse2 +
                                            (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p),
                                        0.)))) *
                      (powr<-1>(fma((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3), (l1) * (p),
                                    _cse1 + _cse2)))))))))))));
      }
      { // subkernel 4
        const auto _interp25 =
            ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))));
        const auto _interp26 =
            ZAcbc((0.816496580927726) *
                  (sqrt(powr<2>(l1) + (-1.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p))));
        const auto _interp27 = ZAcbc(
            (0.3333333333333333) *
            (sqrt((6.) * (powr<2>(l1)) + (-6.) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) + (10.) * (powr<2>(p)))));
        const auto _interp28 =
            ZAcbc((0.3333333333333333) *
                  (sqrt((6.) * (powr<2>(l1)) + (-6.) * ((cosl1p1 + cosl1p2) * (2.) + cosl1p3) * ((l1) * (p)) +
                        (10.) * (powr<2>(p)))));
        const auto _interp29 = dtZc(k);
        const auto _interp30 = Zc(k);
        const auto _interp31 = Zc((1.02) * (k));
        const auto _interp32 = Zc(l1);
        const auto _interp33 = Zc(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
        const auto _interp34 =
            Zc(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2 + cosl1p3) * ((l1) * (p)) + powr<2>(p)));
        const auto _interp35 =
            Zc(sqrt(powr<2>(l1) + (-2.) * (cosl1p1 + cosl1p2) * ((l1) * (p)) + (1.333333333333333) * (powr<2>(p))));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = powr<2>(cosl1p3);
        const auto _cse3 = powr<2>(p);
        const auto _cse4 = powr<3>(cosl1p3);
        const auto _cse5 = powr<2>(cosl1p2);
        const auto _cse6 = powr<3>(cosl1p2);
        _acc +=
            (1.653061224489796) *
            ((_cse1) *
             ((_interp25) *
              ((_interp26) *
               ((_interp27) *
                ((_interp28) *
                 ((fma(-0.3333333333333333, (_cse1) * (_cse2),
                       fma(-0.5925925925925926, _cse3,
                           fma(1., (_cse1) * (powr<4>(cosl1p1)),
                               fma(-2., (_cse1) * (powr<4>(cosl1p2)),
                                   fma(1., (_cse1) * (powr<4>(cosl1p3)),
                                       fma(powr<3>(cosl1p1),
                                           ((1.) * ((cosl1p2) * (l1)) + (3.) * ((cosl1p3) * (l1)) +
                                            (-1.333333333333333) * (p)) *
                                               (l1),
                                           fma(-0.6666666666666666, (_cse4) * ((l1) * (p)),
                                               fma(0.4444444444444445, (cosl1p3) * ((l1) * (p)),
                                                   fma(_cse6, ((-4.) * ((cosl1p3) * (l1)) + (2.) * (p)) * (l1),
                                                       fma(cosl1p2,
                                                           (1.) * ((_cse1) * (_cse4)) +
                                                               (0.6666666666666666) * ((_cse1) * (cosl1p3)) +
                                                               (0.2222222222222222) * ((_cse3) * (cosl1p3)) +
                                                               (-0.4444444444444445) * ((l1) * (p)) +
                                                               (0.3333333333333333) * ((_cse2) * ((l1) * (p))),
                                                           fma(_cse5,
                                                               (0.6666666666666666 + (-1.) * (_cse2)) * (_cse1) +
                                                                   (0.3333333333333333) * (_cse3) +
                                                                   (2.333333333333333) * ((cosl1p3) * ((l1) * (p))),
                                                               fma(powr<2>(cosl1p1),
                                                                   (1.592592592592593) * (_cse3) +
                                                                       (-0.3333333333333333 + (2.) * (_cse2) +
                                                                        (-1.) * (_cse5) +
                                                                        (1.666666666666667) * ((cosl1p2) * (cosl1p3))) *
                                                                           (_cse1) +
                                                                       ((-1.888888888888889) * ((cosl1p2) * (p)) +
                                                                        (-2.111111111111111) * ((cosl1p3) * (p))) *
                                                                           (l1),
                                                                   fma(cosl1p1,
                                                                       (-4.) * ((_cse1) * (_cse6)) +
                                                                           ((0.6666666666666666 +
                                                                             (1.666666666666667) * (_cse2)) *
                                                                                (_cse1) +
                                                                            (1.925925925925926) * (_cse3)) *
                                                                               (cosl1p2) +
                                                                           (_cse5) *
                                                                               ((-3.333333333333333) *
                                                                                    ((cosl1p3) * (l1)) +
                                                                                (0.7777777777777778) * (p)) *
                                                                               (l1) +
                                                                           ((-1.333333333333333 + (3.) * (_cse2)) *
                                                                                (_cse1) +
                                                                            (1.259259259259259) * (_cse3) +
                                                                            (0.5555555555555555) *
                                                                                ((cosl1p3) * ((l1) * (p)))) *
                                                                               (cosl1p3),
                                                                       0.)))))))))))))) *
                  ((powr<-1>(fma(3., (_interp11) * (_interp30),
                                 fma(_interp35,
                                     (3.) * (_cse1) + (4.) * (_cse3) + (-6.) * ((cosl1p1) * ((l1) * (p))) +
                                         (-6.) * ((cosl1p2) * ((l1) * (p))),
                                     0.)))) *
                   ((fma(_interp2, _interp29,
                         fma(_interp3, _interp30, fma(_interp2, (-50.) * (_interp30) + (50.) * (_interp31), 0.)))) *
                    ((powr<-2>(fma(_interp2, _interp30, fma(_cse1, _interp32, 0.)))) *
                     ((powr<-1>(fma(_interp30, _interp7,
                                    fma(_interp33, _cse1 + _cse3 + (-2.) * ((cosl1p1) * ((l1) * (p))), 0.)))) *
                      (powr<-1>(fma(
                          _interp30, _interp9,
                          fma(_interp34,
                              _cse1 + _cse3 + (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p),
                              0.))))))))))))));
      }
      { // subkernel 5
        const auto _interp8 = ZA(sqrt(powr<2>(l1) + (-2.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p)));
        const auto _interp13 =
            ZA3((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cosl1p1) * ((l1) * (p))) + powr<2>(p))));
        const auto _interp15 = ZA3(
            (0.3333333333333333) *
            (sqrt((6.) * (powr<2>(l1)) + (-6.) * ((2.) * (cosl1p1) + cosl1p2) * ((l1) * (p)) + (10.) * (powr<2>(p)))));
        const auto _cse1 = powr<2>(l1);
        const auto _cse2 = powr<2>(p);
        const auto _cse3 = powr<10>(l1);
        const auto _cse4 = powr<9>(l1);
        const auto _cse5 = powr<2>(cosl1p3);
        const auto _cse6 = powr<8>(l1);
        const auto _cse7 = powr<4>(cosl1p3);
        const auto _cse8 = powr<7>(l1);
        const auto _cse9 = powr<3>(p);
        const auto _cse10 = powr<5>(cosl1p3);
        const auto _cse11 = powr<6>(l1);
        const auto _cse12 = powr<4>(p);
        const auto _cse13 = powr<6>(cosl1p3);
        const auto _cse14 = powr<3>(cosl1p3);
        const auto _cse15 = powr<5>(l1);
        const auto _cse16 = powr<5>(p);_acc += (-0.163265306122449) * ((_interp13) * ((_interp14) * ((_interp15) * ((_interp16) * ((powr<-1>(1. + powr<6>(k))) * ((fma(-3756., (_cse11) * (_cse12), fma(162., (_cse11) * ((_cse12) * (_cse13)), fma(-2052., (_cse10) * ((_cse15) * (_cse16)), fma(-1206., (_cse14) * ((_cse15) * (_cse16)), fma(1764., (_cse11) * ((_cse12) * (_cse5)), fma(-243., (_cse3) * (_cse5), fma(-432., (_cse2) * (_cse6), fma(243., (_cse13) * ((_cse2) * (_cse6)), fma(-756., (_cse2) * ((_cse5) * (_cse6)), fma(4050., (_cse11) * ((_cse12) * (_cse7)), fma(729.0000000000001, (_cse3) * (_cse7), fma(3645., (_cse2) * ((_cse6) * (_cse7)), fma(-3726., (_cse10) * ((_cse8) * (_cse9)), fma(-54., (_cse14) * ((_cse8) * (_cse9)), fma(972., (_cse11) * ((_cse12) * (powr<8>(cosl1p1))), fma(972., (_cse8) * ((_cse9) * (powr<7>(cosl1p2))), fma(3624., (_cse15) * ((_cse16) * (cosl1p3)), fma(1008., (_cse8) * ((_cse9) * (cosl1p3)), fma(-1458., (_cse10) * ((_cse4) * (p)), fma(324., (_cse4) * ((cosl1p3) * (p)), fma(-7159.999999999999, (powr<4>(l1)) * (powr<6>(p)), fma(4128., (_cse5) * ((powr<4>(l1)) * (powr<6>(p))), fma(1548., (_cse7) * ((powr<4>(l1)) * (powr<6>(p))), fma(-1248., (_cse14) * ((powr<3>(l1)) * (powr<7>(p))), fma(3712., (cosl1p3) * ((powr<3>(l1)) * (powr<7>(p))), fma(-4834., (_cse1) * (powr<8>(p)), fma(2176., (_cse1) * ((_cse5) * (powr<8>(p))), fma(836., (cosl1p3) * ((l1) * (powr<9>(p))), fma(-1024., powr<10>(p), fma(_cse11, (_cse2) * ((-6318.) * (_cse1) + (-9072.) * (_cse2) + (3402.) * ((cosl1p3) * ((l1) * (p)))) * (powr<6>(cosl1p2)), fma(_cse15, (_cse9) * ((-6803.999999999999) * (_cse1) + (-5183.999999999999) * (_cse2) + (3402.) * ((cosl1p2) * ((l1) * (p))) + (4374.) * ((cosl1p3) * ((l1) * (p)))) * (powr<7>(cosl1p1)), fma(_cse15, (powr<5>(cosl1p2)) * ((19926.) * (_cse12) + (_cse1) * (20898. + (3888.) * (_cse5)) * (_cse2) + (-27216.) * ((_cse9) * ((cosl1p3) * (l1))) + (5832.000000000001) * (powr<4>(l1)) + (-18954.) * ((cosl1p3) * ((powr<3>(l1)) * (p)))) * (p), fma(powr<4>(cosl1p2), ((-1458.) * (_cse11) + (_cse1) * (-25353. + (-22113.) * (_cse5)) * (_cse12) + (49868.99999999999) * ((_cse16) * ((cosl1p3) * (l1))) + (_cse9) * (50625. + (1215.) * (_cse5)) * ((cosl1p3) * (powr<3>(l1))) + (_cse2) * (-9234. + (-16281.) * (_cse5)) * (powr<4>(l1)) + (14580.) * ((_cse15) * ((cosl1p3) * (p))) + (-17802.) * (powr<6>(p))) * (powr<4>(l1)), fma(_cse1, ((486. + (-729.0000000000001) * (_cse5)) * (_cse6) + (_cse11) * (1755. + (-4131.) * (_cse5) + (6075.) * (_cse7)) * (_cse2) + (_cse15) * (8586. + (-10449.) * (_cse5) + (-972.) * (_cse7)) * ((_cse9) * (cosl1p3)) + (_cse16) * (17712. + (-1404.) * (_cse5)) * ((cosl1p3) * (powr<3>(l1))) + (_cse12) * (-3528. + (-22275.) * (_cse5) + (9153.) * (_cse7)) * (powr<4>(l1)) + (_cse8) * (-1215. + (-1458.) * (_cse5)) * ((cosl1p3) * (p)) + (_cse1) * (-17742. + (-15012.) * (_cse5)) * (powr<6>(p)) + (10824.) * ((cosl1p3) * ((l1) * (powr<7>(p)))) + (-11368.) * (powr<8>(p))) * (powr<2>(cosl1p2)), fma(powr<2>(cosl1p1), (-243. + (1458.) * (_cse5) + (-729.0000000000001) * (powr<2>(cosl1p2)) + (1215.) * ((cosl1p2) * (cosl1p3))) * (_cse3) + (_cse15) * ((-3240.) * (_cse10) + (-3473.999999999999) * (_cse14) + (-70008.00000000001 + (-28809.) * (_cse5) + (-12825.) * (_cse7)) * (cosl1p2) + (42561. + (25974.) * (_cse5)) * (powr<3>(cosl1p2)) + (21762.) * (powr<5>(cosl1p2)) + ((-11988.) * (_cse14) + (-6372.000000000001) * (cosl1p3)) * (powr<2>(cosl1p2)) + (-90102.) * (cosl1p3) + (51111.) * ((powr<4>(cosl1p2)) * (cosl1p3))) * (_cse16) + (_cse2) * (1053. + (-8586.) * (_cse5) + (27702.) * (_cse7) + (14013. + (1944.) * (_cse5)) * (powr<2>(cosl1p2)) + (-68768.99999999999) * (powr<4>(cosl1p2)) + (-97848.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + ((61560.00000000001) * (_cse14) + (7209.) * (cosl1p3)) * (cosl1p2)) * (_cse6) + (_cse8) * ((-9234.) * (_cse10) + (-28323.) * (_cse14) + (-26838. + (-21870.) * (_cse5) + (-43983.00000000001) * (_cse7)) * (cosl1p2) + (54648. + (55080.) * (_cse5)) * (powr<3>(cosl1p2)) + (55404.) * (powr<5>(cosl1p2)) + (-14148.) * (cosl1p3) + (120123.) * ((powr<4>(cosl1p2)) * (cosl1p3)) + ((-39042.) * (_cse14) + (33291.) * (cosl1p3)) * (powr<2>(cosl1p2))) * (_cse9) + (_cse11) * (4644. + (486.) * (_cse13) + (10881.) * (_cse5) + (35721.) * (_cse7) + (48708.00000000001 + (-21816.) * (_cse5) + (6237.) * (_cse7)) * (powr<2>(cosl1p2)) + (-121689. + (-14256.) * (_cse5)) * (powr<4>(cosl1p2)) + (-5832.000000000001) * (powr<6>(cosl1p2)) + ((-324.) * (_cse14) + (-188892.) * (cosl1p3)) * (powr<3>(cosl1p2)) + (-16686.) * ((powr<5>(cosl1p2)) * (cosl1p3)) + ((3402.) * (_cse10) + (77085.00000000001) * (_cse14) + (90351.) * (cosl1p3)) * (cosl1p2)) * (_cse12) + (_cse4) * ((-16038.) * (_cse14) + (-3321. + (-15552.) * (_cse5)) * (cosl1p2) + (20412.) * (powr<3>(cosl1p2)) + (4779.) * (cosl1p3) + (11178.) * ((powr<2>(cosl1p2)) * (cosl1p3))) * (p) + (powr<4>(l1)) * (-5424. + (69840.) * (_cse5) + (-702.) * (_cse7) + (110952. + (-11736.) * (_cse5)) * (powr<2>(cosl1p2)) + (-16830.) * (powr<4>(cosl1p2)) + (-24525.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + ((-5958.) * (_cse14) + (225468.) * (cosl1p3)) * (cosl1p2)) * (powr<6>(p)) + (powr<3>(l1)) * ((2700.) * (_cse14) + (-65252. + (-48816.00000000001) * (_cse5)) * (cosl1p2) + (-51858.) * (powr<3>(cosl1p2)) + (-94120.) * (cosl1p3) + (-106866.) * ((powr<2>(cosl1p2)) * (cosl1p3))) * (powr<7>(p)) + (_cse1) * (-1594. + (33312.) * (_cse5) + (82452.) * (powr<2>(cosl1p2)) + (115242.) * ((cosl1p2) * (cosl1p3))) * (powr<8>(p)) + (l1) * ((-34188.) * (cosl1p2) + (-27480.) * (cosl1p3)) * (powr<9>(p)) + (3072.) * (powr<10>(p)), fma(_cse2, (powr<6>(cosl1p1)) * ((21960.) * (_cse12) + (_cse1) * (23328. + (6803.999999999999) * (_cse5) + (3402.) * (powr<2>(cosl1p2)) + (12555.) * ((cosl1p2) * (cosl1p3))) * (_cse2) + (_cse9) * ((-22086.) * (cosl1p2) + (-14202.) * (cosl1p3)) * (l1) + (8991.) * (powr<4>(l1)) + ((-20169.) * ((cosl1p2) * (p)) + (-27459.) * ((cosl1p3) * (p))) * (powr<3>(l1))) * (powr<4>(l1)), fma(_cse1, ((729.0000000000001) * (_cse6) + (_cse11) * (5589. + (36450.) * (_cse5) + (6075.) * (powr<2>(cosl1p2)) + (59940.) * ((cosl1p2) * (cosl1p3))) * (_cse2) + (_cse15) * ((-36693.) * (_cse14) + (-61128. + (-72657.) * (_cse5)) * (cosl1p2) + (33048.) * (powr<3>(cosl1p2)) + (-46602.) * (cosl1p3) + (-21708.) * ((powr<2>(cosl1p2)) * (cosl1p3))) * (_cse9) + (_cse16) * ((-2322.) * (_cse14) + (-183429. + (-20547.) * (_cse5)) * (cosl1p2) + (-6318.) * (powr<3>(cosl1p2)) + (-135891.) * (cosl1p3) + (-31563.) * ((powr<2>(cosl1p2)) * (cosl1p3))) * (powr<3>(l1)) + (_cse12) * (49581. + (36855.) * (_cse5) + (6318.) * (_cse7) + (71955. + (7695.) * (_cse5)) * (powr<2>(cosl1p2)) + (-13122.) * (powr<4>(cosl1p2)) + (-14580.) * ((powr<3>(cosl1p2)) * (cosl1p3)) + ((16605.) * (_cse14) + (141075.) * (cosl1p3)) * (cosl1p2)) * (powr<4>(l1)) + (_cse1) * (100098. + (19944.) * (_cse5) + (103734.) * (powr<2>(cosl1p2)) + (132255.) * ((cosl1p2) * (cosl1p3))) * (powr<6>(p)) + (l1) * ((-151758.) * (cosl1p2) + (-98622.) * (cosl1p3)) * (powr<7>(p)) + (48972.) * (powr<8>(p)) + ((-7290.000000000001) * ((cosl1p2) * (p)) + (-14580.) * ((cosl1p3) * (p))) * (_cse8)) * (powr<4>(cosl1p1)), fma(powr<5>(cosl1p1), (powr<3>(l1)) * ((-4374.) * (_cse11) + (_cse1) * (-63864. + (-8046.) * (_cse5) + (-31698.) * (powr<2>(cosl1p2)) + (-44982.) * ((cosl1p2) * (cosl1p3))) * (_cse12) + (_cse16) * ((82638.) * (cosl1p2) + (49122.) * (cosl1p3)) * (l1) + (_cse9) * ((7290.000000000001) * (_cse14) + (77003.99999999999 + (16200.) * (_cse5)) * (cosl1p2) + (-3888.) * (powr<3>(cosl1p2)) + (62964.) * (cosl1p3) + (8100.) * ((powr<2>(cosl1p2)) * (cosl1p3))) * (powr<3>(l1)) + (_cse2) * (-21546. + (-35478.) * (_cse5) + (-13608.) * (powr<2>(cosl1p2)) + (-65772.) * ((cosl1p2) * (cosl1p3))) * (powr<4>(l1)) + (-50076.00000000001) * (powr<6>(p)) + ((20898.) * ((cosl1p2) * (p)) + (33048.) * ((cosl1p3) * (p))) * (_cse15)) * (p), fma(powr<3>(cosl1p2), ((14526.) * ((_cse1) * (_cse16)) + (4644.) * ((_cse9) * (powr<4>(l1))) + (-972.) * ((_cse7) * ((_cse9) * (powr<4>(l1)))) + ((-972.) * ((_cse15) * (_cse2)) + (1620.) * ((_cse12) * (powr<3>(l1)))) * (_cse14) + (-486.) * ((_cse11) * (p)) + (12480.) * (powr<7>(p)) + ((33507.) * ((_cse1) * (_cse16)) + (27378.) * ((_cse9) * (powr<4>(l1))) + (8748.) * ((_cse11) * (p))) * (_cse5) + ((-16524.) * ((_cse15) * (_cse2)) + (-2916.) * (_cse8) + (-50031.00000000001) * ((_cse12) * (powr<3>(l1))) + (-34020.) * ((l1) * (powr<6>(p)))) * (cosl1p3)) * (powr<3>(l1)), fma(cosl1p2, (((3726.) * ((_cse12) * (_cse15)) + (3402.) * ((_cse2) * (_cse8))) * (_cse10) + (1584.) * ((_cse11) * (_cse9)) + (-243.) * ((_cse11) * ((_cse13) * (_cse9))) + (13344.) * ((_cse16) * (powr<4>(l1))) + (-324.) * ((_cse6) * (p)) + (17428.) * ((_cse1) * (powr<7>(p))) + (6016.) * (powr<9>(p)) + ((-14742.) * ((_cse11) * (_cse9)) + (-7992.) * ((_cse16) * (powr<4>(l1))) + (-4374.) * ((_cse6) * (p))) * (_cse7) + ((4104.) * ((_cse12) * (_cse15)) + (729.0000000000001) * (_cse4) + (5103.) * ((_cse2) * (_cse8)) + (1728.) * ((powr<3>(l1)) * (powr<6>(p)))) * (_cse14) + ((4482.) * ((_cse11) * (_cse9)) + (1530.) * ((_cse16) * (powr<4>(l1))) + (243.) * ((_cse6) * (p)) + (-2364.) * ((_cse1) * (powr<7>(p)))) * (_cse5) + ((-2088.) * ((_cse12) * (_cse15)) + (486.) * (_cse4) + (-270.) * ((_cse2) * (_cse8)) + (-6054.) * ((powr<3>(l1)) * (powr<6>(p))) + (-3198.) * ((l1) * (powr<8>(p)))) * (cosl1p3)) * (l1), fma(powr<3>(cosl1p1), ((2916.) * ((_cse10) * ((_cse12) * (_cse15))) + (-13662.) * ((_cse11) * (_cse9)) + (-13122.) * ((_cse12) * ((_cse15) * (powr<5>(cosl1p2)))) + (-53370.00000000001) * ((_cse16) * (powr<4>(l1))) + ((-29403.) * ((_cse11) * (_cse9)) + (-6372.000000000001) * ((_cse16) * (powr<4>(l1)))) * (_cse7) + (486.) * ((_cse6) * (p)) + (-53124.) * ((_cse1) * (powr<7>(p))) + (-20556.) * (powr<9>(p)) + (_cse9) * ((73628.99999999999) * (_cse1) + (25056.) * (_cse2) + (-27783.00000000001) * ((cosl1p3) * ((l1) * (p)))) * ((powr<4>(cosl1p2)) * (powr<4>(l1))) + (_cse2) * ((36432.) * (_cse12) + (_cse1) * (-48600. + (-12312.) * (_cse5)) * (_cse2) + (34938.) * ((_cse9) * ((cosl1p3) * (l1))) + (-44712.) * (powr<4>(l1)) + (93312.) * ((cosl1p3) * ((powr<3>(l1)) * (p)))) * ((powr<3>(cosl1p2)) * (powr<3>(l1))) + (_cse1) * ((1458.) * (_cse11) + (_cse1) * (-140949. + (-378.) * (_cse5)) * (_cse12) + (85905.) * ((_cse16) * ((cosl1p3) * (l1))) + (_cse9) * (16551. + (10692.) * (_cse5)) * ((cosl1p3) * (powr<3>(l1))) + (_cse2) * (-41661. + (-19926.) * (_cse5)) * (powr<4>(l1)) + (-3564.) * ((_cse15) * ((cosl1p3) * (p))) + (-152982.) * (powr<6>(p))) * ((powr<2>(cosl1p2)) * (p)) + ((32940.) * ((_cse12) * (_cse15)) + (39852.) * ((_cse2) * (_cse8)) + (-8676.) * ((powr<3>(l1)) * (powr<6>(p)))) * (_cse14) + ((-12609.) * ((_cse11) * (_cse9)) + (-45873.00000000001) * ((_cse16) * (powr<4>(l1))) + (-13122.) * ((_cse6) * (p)) + (-46710.) * ((_cse1) * (powr<7>(p)))) * (_cse5) + (cosl1p2) * ((_cse11) * (18711. + (58643.99999999999) * (_cse5)) * (_cse2) + (729.0000000000001) * (_cse6) + (_cse15) * (-76788. + (-75168.00000000001) * (_cse5)) * ((_cse9) * (cosl1p3)) + (_cse16) * (-237636. + (-13878.) * (_cse5)) * ((cosl1p3) * (powr<3>(l1))) + (_cse12) * (111771. + (65528.99999999999) * (_cse5) + (11745.) * (_cse7)) * (powr<4>(l1)) + (-17496.) * ((_cse8) * ((cosl1p3) * (p))) + (_cse1) * (213900. + (31671.) * (_cse5)) * (powr<6>(p)) + (-202428.) * ((cosl1p3) * ((l1) * (powr<7>(p)))) + (114324.) * (powr<8>(p))) * (l1) + ((86553.) * ((_cse12) * (_cse15)) + (2187.) * (_cse4) + (3645.) * ((_cse2) * (_cse8)) + (186492.) * ((powr<3>(l1)) * (powr<6>(p))) + (81564.) * ((l1) * (powr<8>(p)))) * (cosl1p3)) * (l1), fma(cosl1p1, (16968.) * ((_cse15) * (_cse16)) + (2592.) * ((_cse8) * (_cse9)) + ((-162.) * ((_cse15) * (_cse16)) + (-729.0000000000001) * ((_cse8) * (_cse9))) * (_cse13) + (-972.) * ((_cse11) * ((_cse12) * (powr<7>(cosl1p2)))) + (21140.) * ((powr<3>(l1)) * (powr<7>(p))) + (6852.000000000001) * ((l1) * (powr<9>(p))) + (_cse15) * ((16038.) * (_cse1) + (5183.999999999999) * (_cse2) + (-3402.) * ((cosl1p3) * ((l1) * (p)))) * ((_cse9) * (powr<6>(cosl1p2))) + (_cse2) * ((-10206.) * (_cse12) + (_cse1) * (-64044. + (-3888.) * (_cse5)) * (_cse2) + (15552.) * ((_cse9) * ((cosl1p3) * (l1))) + (-36450.) * (powr<4>(l1)) + (47304.) * ((cosl1p3) * ((powr<3>(l1)) * (p)))) * ((powr<5>(cosl1p2)) * (powr<4>(l1))) + (powr<4>(cosl1p2)) * ((20412.) * (_cse11) + (_cse1) * (81819. + (12393.) * (_cse5)) * (_cse12) + (-26541.) * ((_cse16) * ((cosl1p3) * (l1))) + (_cse9) * (-149040. + (-1215.) * (_cse5)) * ((cosl1p3) * (powr<3>(l1))) + (_cse2) * (73899. + (40257.) * (_cse5)) * (powr<4>(l1)) + (-82944.) * ((_cse15) * ((cosl1p3) * (p))) + (-558.) * (powr<6>(p))) * ((powr<3>(l1)) * (p)) + ((9234.) * ((_cse11) * (_cse12)) + (6803.999999999999) * ((_cse2) * (_cse6)) + (756.) * ((powr<4>(l1)) * (powr<6>(p)))) * (_cse10) + ((-11754.) * ((_cse15) * (_cse16)) + (-23598.) * ((_cse8) * (_cse9)) + (-8748.) * ((_cse4) * (p)) + (864.) * ((powr<3>(l1)) * (powr<7>(p)))) * (_cse7) + (_cse1) * ((_cse11) * (-11988. + (-42444.) * (_cse5)) * (_cse2) + (-2916.) * (_cse6) + (_cse15) * (122958. + (1296.) * (_cse5)) * ((_cse9) * (cosl1p3)) + (_cse16) * (149004. + (-1620.) * (_cse5)) * ((cosl1p3) * (powr<3>(l1))) + (_cse12) * (-42885.00000000001 + (-74952.) * (_cse5) + (972.) * (_cse7)) * (powr<4>(l1)) + (33048.) * ((_cse8) * ((cosl1p3) * (p))) + (_cse1) * (-22200. + (-20979.) * (_cse5)) * (powr<6>(p)) + (-3924.) * ((cosl1p3) * ((l1) * (powr<7>(p)))) + (17100.) * (powr<8>(p))) * (powr<3>(cosl1p2)) + ((3312.) * ((_cse11) * (_cse12)) + (2187.) * (_cse3) + (6237.) * ((_cse2) * (_cse6)) + (2796.) * ((powr<4>(l1)) * (powr<6>(p))) + (720.) * ((_cse1) * (powr<8>(p)))) * (_cse14) + ((-23412.) * ((_cse15) * (_cse16)) + (4104.) * ((_cse8) * (_cse9)) + (3807.) * ((_cse4) * (p)) + (-29764.) * ((powr<3>(l1)) * (powr<7>(p))) + (-6924.) * ((l1) * (powr<9>(p)))) * (_cse5) + ((9936.) * ((_cse11) * (_cse12)) + (-972.) * (_cse3) + (-1458.) * ((_cse2) * (_cse6)) + (16446.) * ((powr<4>(l1)) * (powr<6>(p))) + (11950.) * ((_cse1) * (powr<8>(p))) + (3072.) * (powr<10>(p))) * (cosl1p3) + ((486. + (1215.) * (_cse5)) * (_cse3) + (_cse2) * (3564. + (-1863.) * (_cse5) + (27864.) * (_cse7)) * (_cse6) + (_cse11) * (-648. + (243.) * (_cse13) + (-10170.) * (_cse5) + (42525.) * (_cse7)) * (_cse12) + (_cse15) * (-47448. + (-14544.) * (_cse5) + (-1782.) * (_cse7)) * ((_cse16) * (cosl1p3)) + (_cse8) * (-5183.999999999999 + (-44226.00000000001) * (_cse5) + (-8748.) * (_cse7)) * ((_cse9) * (cosl1p3)) + (-15552.) * ((_cse14) * ((_cse4) * (p))) + (powr<4>(l1)) * (-27294. + (25116.) * (_cse5) + (1782.) * (_cse7)) * (powr<6>(p)) + (cosl1p3) * (-55896. + (2412.) * (_cse5)) * ((powr<3>(l1)) * (powr<7>(p))) + (_cse1) * (-15138. + (18018.) * (_cse5)) * (powr<8>(p)) + (-20556.) * ((cosl1p3) * ((l1) * (powr<9>(p)))) + (3072.) * (powr<10>(p))) * (cosl1p2) + (powr<2>(cosl1p2)) * ((972.) * ((_cse10) * ((_cse12) * (_cse15))) + (-8586.) * ((_cse11) * (_cse9)) + (-3318.) * ((_cse16) * (powr<4>(l1))) + ((-16200.) * ((_cse11) * (_cse9)) + (-5265.) * ((_cse16) * (powr<4>(l1)))) * (_cse7) + (-4293.) * ((_cse6) * (p)) + (-896.) * ((_cse1) * (powr<7>(p))) + (-13632.) * (powr<9>(p)) + ((37503.) * ((_cse12) * (_cse15)) + (19764.) * ((_cse2) * (_cse8)) + (-3618.) * ((powr<3>(l1)) * (powr<6>(p)))) * (_cse14) + ((23571.) * ((_cse11) * (_cse9)) + (60237.) * ((_cse16) * (powr<4>(l1))) + (4374.) * ((_cse6) * (p)) + (-1818.) * ((_cse1) * (powr<7>(p)))) * (_cse5) + ((-47025.) * ((_cse12) * (_cse15)) + (-2430.) * (_cse4) + (-11826.) * ((_cse2) * (_cse8)) + (6023.999999999999) * ((powr<3>(l1)) * (powr<6>(p))) + (34398.) * ((l1) * (powr<8>(p)))) * (cosl1p3)) * (l1), 0.))))))))))))))))))))))))))))))))))))))))))) * ((powr<-1>(fma(-2., (cosl1p1) * ((l1) * (p)), _cse1 + _cse2))) * ((powr<-1>(fma(3., _cse1, fma(4., _cse2, fma(-6., (cosl1p1) * ((l1) * (p)), fma(-6., (cosl1p2) * ((l1) * (p)), 0.)))))) * ((powr<-1>(fma(3., (_interp11) * (_interp4), fma(_interp12, (3.) * (_cse1) + (4.) * (_cse2) + (-6.) * ((cosl1p1) * ((l1) * (p))) + (-6.) * ((cosl1p2) * ((l1) * (p))), 0.)))) * ((powr<-2>(fma(_interp2, _interp4, fma(_cse1, _interp6, 0.)))) * ((fma(_interp2, ((-50.) * (_interp4) + (50.) * (_interp5)) * (powr<6>(k)), fma(_interp3, (1. + powr<6>(k)) * (_interp4), fma(_interp1, (1. + (1.) * (powr<6>(k))) * (_interp2), 0.)))) * ((powr<-1>(fma(_interp4, _interp7, fma(_interp8, _cse1 + _cse2 + (-2.) * ((cosl1p1) * ((l1) * (p))), 0.)))) * ((powr<-1>(fma(_interp4, _interp9, fma(_interp10, _cse1 + _cse2 + (l1) * ((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3)) * (p), 0.)))) * (powr<-1>(fma((-2.) * (cosl1p1) + (-2.) * (cosl1p2) + (-2.) * (cosl1p3), (l1) * (p), _cse1 + _cse2))))))))))))))));
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
using DiFfRG::ZA4_kernel;