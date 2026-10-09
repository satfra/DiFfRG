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
    kernel(const double &l1, const double &cos1, const double &cos2, const double &p, const double &k,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA3,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZAcbc,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA4,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZc,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &Zc,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZA,
           const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      const double cosl1p1 = cos1;
      const double cosl1p2 = 0.5 * (-cos1 + sqrt(3. - 3. * powr<2>(cos1)) * cos2);
      const double cosl1p3 = 0.5 * (-cos1 - sqrt(3. - 3. * powr<2>(cos1)) * cos2);
      const auto _interp1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = RB(powr<2>(k), fma(-2., cosl1p1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp8 = ZA(sqrt(fma(-2., cosl1p1 * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp9 = RB(powr<2>(k), fma(2., cosl1p2 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp11 = ZA3(0.816496580927726 * sqrt(fma(-1., cosl1p1 * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp14 = RB(powr<2>(k), fma(-2., (cosl1p1 + cosl1p2) * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp15 = ZA(sqrt(fma(-2., (cosl1p1 + cosl1p2) * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp16 =
          ZA3(0.816496580927726 * sqrt(fma(-1., (cosl1p1 + cosl1p2) * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp18 = RB(powr<2>(k), fma(-2., cosl1p2 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp19 = ZA(sqrt(fma(-2., cosl1p2 * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp20 = ZA3(0.816496580927726 * sqrt(fma(-1., cosl1p2 * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp28 = dtZc(k);
      const auto _interp29 = Zc(k);
      const auto _interp30 = Zc(1.02 * k);
      const auto _interp31 = Zc(l1);
      const auto _interp34 =
          ZAcbc(0.816496580927726 * sqrt(fma(-1., (cosl1p1 + cosl1p2) * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp36 = Zc(sqrt(fma(-2., (cosl1p1 + cosl1p2) * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _den1 = powr<-1>(1. + powr<6>(k));
      const auto _den2 = powr<-1>(1. + powr<6>(k));
      const auto _den3 = powr<-2>(fma(_interp2, _interp29, fma(_interp31, powr<2>(l1), 0.)));
      const auto _den4 = powr<-2>(fma(_interp2, _interp4, fma(_interp6, powr<2>(l1), 0.)));
      const auto _den6 = powr<-1>(fma(-2., cosl1p1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _den14 =
          powr<-1>(fma(_interp4, _interp7, fma(_interp8, powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p), 0.)));
      const auto _den22 =
          powr<-1>(fma(_interp14, _interp29,
                       fma(_interp36, powr<2>(l1) + (-2. * cosl1p1 - 2. * cosl1p2) * l1 * p + powr<2>(p), 0.)));
      const auto _interp25 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p + powr<2>(p)));
      const auto _interp26 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p2 * l1 * p + powr<2>(p)));
      const auto _interp27 = ZAcbc(
          sqrt(0.6666666666666666 * powr<2>(l1) + 0.6666666666666666 * (-cosl1p1 + cosl1p2) * l1 * p + powr<2>(p)));
      const auto _interp32 = Zc(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p)));
      const auto _interp33 = Zc(sqrt(powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p)));
      const auto _interp35 = ZAcbc(
          sqrt(0.6666666666666666 * powr<2>(l1) - 0.6666666666666666 * (2. * cosl1p1 + cosl1p2) * l1 * p + powr<2>(p)));
      const auto _interp22 = ZA4(0.5 * sqrt(2. * powr<2>(l1) - 2. * cosl1p1 * l1 * p + 3. * powr<2>(p)));
      const auto _interp23 = ZA4(0.5 * sqrt(2. * powr<2>(l1) - 2. * cosl1p2 * l1 * p + 3. * powr<2>(p)));
      const auto _interp24 = ZA4(0.5 * sqrt(2. * powr<2>(l1) - 2. * (cosl1p1 + cosl1p2) * l1 * p + 3. * powr<2>(p)));
      const auto _interp37 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p2 * l1 * p + powr<2>(p)));
      const auto _interp38 = ZAcbc(
          sqrt(0.6666666666666666 * powr<2>(l1) - 0.6666666666666666 * (cosl1p1 + 2. * cosl1p2) * l1 * p + powr<2>(p)));
      const auto _interp39 = Zc(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p)));
      const auto _interp17 = ZA3(
          sqrt(0.6666666666666666 * powr<2>(l1) - 0.6666666666666666 * (2. * cosl1p1 + cosl1p2) * l1 * p + powr<2>(p)));
      const auto _interp21 = ZA3(
          sqrt(0.6666666666666666 * powr<2>(l1) - 0.6666666666666666 * (cosl1p1 + 2. * cosl1p2) * l1 * p + powr<2>(p)));
      const auto _interp10 = ZA(sqrt(powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p)));
      const auto _interp12 = ZA3(0.816496580927726 * sqrt(powr<2>(l1) + cosl1p2 * l1 * p + powr<2>(p)));
      const auto _interp13 =
          ZA3(sqrt(0.6666666666666666 * powr<2>(l1) + 0.6666666666666666 * (-cosl1p1 + cosl1p2) * l1 * p + powr<2>(p)));
      // clang-format off
      using _T = decltype(_den1 + _den14 + _den2 + _den22 + _den3 + _den4 + _den6 + _interp1 + _interp10 + _interp11 + _interp12 + _interp13 + _interp14 + _interp15 + _interp16 + _interp17 + _interp18 + _interp19 + _interp2 + _interp20 + _interp21 + _interp22 + _interp23 + _interp24 + _interp25 + _interp26 + _interp27 + _interp28 + _interp29 + _interp3 + _interp30 + _interp31 + _interp32 + _interp33 + _interp34 + _interp35 + _interp36 + _interp37 + _interp38 + _interp39 + _interp4 + _interp5 + _interp6 + _interp7 + _interp8 + _interp9 + cosl1p1 + cosl1p2 + k + l1 + p);
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _den13 =
            powr<-1>(_interp29 * _interp7 + _interp32 * (powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p)));
        const auto _den19 =
            powr<-1>(_interp29 * _interp9 + _interp33 * (powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p)));
        const auto _cse1_k1 = powr<-1>(p);
        const auto _cse2_k1 = 1. * powr<3>(cosl1p1) * l1;
        const auto _cse3_k1 = -powr<3>(cosl1p2) * l1;
        const auto _cse4_k1 = -1.5 * cosl1p2 * l1;
        const auto _cse5_k1 = 1.5 * cosl1p2 * l1;
        const auto _cse6_k1 = -2.75 * p;
        const auto _cse7_k1 = _interp2 * _interp28;
        const auto _cse8_k1 = _interp29 * _interp3;
        const auto _cse9_k1 = -50. * _interp29;
        const auto _cse10_k1 = 50. * _interp30;
        const auto _cse11_k1 = _cse10_k1 + _cse9_k1;
        const auto _cse12_k1 = _cse11_k1 * _interp2;
        const auto _cse13_k1 = _cse12_k1 + _cse7_k1 + _cse8_k1;
        _acc +=
            fma(0.1212121212121212,
                _cse13_k1 * _cse1_k1 * _den13 * _den22 * _den3 * _interp25 * _interp34 * _interp35 * powr<2>(l1) *
                    (_cse2_k1 + _cse3_k1 + (_cse5_k1 + _cse6_k1) * powr<2>(cosl1p1) +
                     (_cse4_k1 + _cse6_k1) * cosl1p1 * cosl1p2 + 1.5 * p - 0.5 * powr<2>(cosl1p2) * p),
                fma(-0.1212121212121212,
                    _cse13_k1 * _cse1_k1 * _den13 * _den19 * _den3 * _interp25 * _interp26 * _interp27 * powr<2>(l1) *
                        (_cse2_k1 + _cse3_k1 + cosl1p1 * cosl1p2 * (_cse4_k1 - 1.75 * p) +
                         powr<2>(cosl1p1) * (_cse5_k1 + 0.5 * p) - 1.5 * p + 0.5 * powr<2>(cosl1p2) * p),
                    0.));
      }
      { // subkernel 2
        const auto _den5 = powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p));
        const auto _den7 = powr<-1>(powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p));
        const auto _den12 =
            powr<-1>(_interp4 * _interp7 + _interp8 * (powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p)));
        const auto _den15 =
            powr<-1>(_interp18 * _interp4 + _interp19 * (powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p)));
        const auto _cse1_k2 = 4. * powr<2>(cosl1p1);
        const auto _cse2_k2 = 4. * cosl1p1 * cosl1p2;
        const auto _cse3_k2 = -53. * powr<2>(cosl1p2);
        const auto _cse4_k2 = 54. + _cse1_k2 + _cse2_k2 + _cse3_k2;
        const auto _cse5_k2 = _interp3 * _interp4;
        const auto _cse6_k2 = -_interp4;
        const auto _cse7_k2 = _cse6_k2 + _interp5;
        const auto _cse8_k2 = 50. * _cse7_k2 * _den1 * powr<6>(k);
        const auto _cse9_k2 = _cse8_k2 + _interp1;
        const auto _cse10_k2 = _cse9_k2 * _interp2;
        const auto _cse11_k2 = _cse10_k2 + _cse5_k2;
        // clang-format off
        _acc += 0.0909090909090909 * fma(54., _cse11_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * powr<2>(l1), fma(_cse11_k2, _cse4_k2 * _den15 * _den4 * _den7 * _interp20 * _interp23 * powr<2>(l1), fma(-53., _cse11_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * powr<2>(cosl1p1) * powr<2>(l1), fma(4., _cse11_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * cosl1p1 * cosl1p2 * powr<2>(l1), fma(4., _cse11_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * powr<2>(cosl1p2) * powr<2>(l1), fma(-54., _cse11_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * cosl1p1 * l1 * p, fma(-1., _cse11_k2 * _cse2_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * cosl1p1 * l1 * p, fma(53., _cse11_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * powr<3>(cosl1p1) * l1 * p, fma(-1., _cse11_k2 * _cse4_k2 * _den15 * _den4 * _den7 * _interp20 * _interp23 * cosl1p2 * l1 * p, fma(-4., _cse11_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * cosl1p1 * powr<2>(cosl1p2) * l1 * p, fma(33., _cse11_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * powr<2>(p), fma(33., _cse11_k2 * _den15 * _den4 * _den7 * _interp20 * _interp23 * powr<2>(p), fma(-33., _cse11_k2 * _den12 * _den4 * _den5 * _interp11 * _interp22 * powr<2>(cosl1p1) * powr<2>(p), fma(-33., _cse11_k2 * _den15 * _den4 * _den7 * _interp20 * _interp23 * powr<2>(cosl1p2) * powr<2>(p), 0.))))))))))))));
        // clang-format on
      }
      { // subkernel 3
        const auto _den10 = powr<-1>(powr<2>(l1) - 2. * (cosl1p1 + cosl1p2) * l1 * p + powr<2>(p));
        const auto _den17 =
            powr<-1>(_interp18 * _interp29 + _interp39 * (powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p)));
        const auto _den20 =
            powr<-1>(_interp14 * _interp4 + _interp15 * (powr<2>(l1) - 2. * (cosl1p1 + cosl1p2) * l1 * p + powr<2>(p)));
        _acc += fma(
            -0.0909090909090909,
            _den10 * _den20 * _den4 * _interp16 * _interp24 *
                (_interp3 * _interp4 + _interp2 * (_interp1 + 50. * _den1 * (-_interp4 + _interp5) * powr<6>(k))) *
                ((-54. + 53. * powr<2>(cosl1p1) + 110. * cosl1p1 * cosl1p2 + 53. * powr<2>(cosl1p2)) * powr<2>(l1) -
                 (-54. * cosl1p1 + 53. * powr<3>(cosl1p1) - 54. * cosl1p2 + 163. * powr<2>(cosl1p1) * cosl1p2 +
                  163. * cosl1p1 * powr<2>(cosl1p2) + 53. * powr<3>(cosl1p2)) *
                     l1 * p +
                 33. * (-1. + powr<2>(cosl1p1) + 2. * cosl1p1 * cosl1p2 + powr<2>(cosl1p2)) * powr<2>(p)),
            fma(-0.1212121212121212,
                _den17 * _den22 * _den3 *
                    (_interp2 * _interp28 + _interp29 * _interp3 + _interp2 * (-50. * _interp29 + 50. * _interp30)) *
                    _interp34 * _interp37 * _interp38 * powr<2>(l1) * powr<-1>(p) *
                    (1. * powr<3>(cosl1p1) * l1 - powr<3>(cosl1p2) * l1 +
                     powr<2>(cosl1p1) * (1.5 * cosl1p2 * l1 + 0.5 * p) - 1.5 * p + 2.75 * powr<2>(cosl1p2) * p +
                     cosl1p1 * cosl1p2 * (-1.5 * cosl1p2 * l1 + 2.75 * p)),
                0.));
      }
      { // subkernel 4
        const auto _den8 = powr<-1>(powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p));
        const auto _den11 = powr<-1>(powr<2>(l1) + (-2. * cosl1p1 - 2. * cosl1p2) * l1 * p + powr<2>(p));
        const auto _den16 =
            powr<-1>(_interp18 * _interp4 + _interp19 * (powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p)));
        const auto _den21 = powr<-1>(_interp14 * _interp4 +
                                     _interp15 * (powr<2>(l1) + (-2. * cosl1p1 - 2. * cosl1p2) * l1 * p + powr<2>(p)));
        const auto _cse1_k4 = 132. * powr<6>(l1) * p;
        const auto _cse2_k4 = 326. * powr<4>(l1) * powr<3>(p);
        const auto _cse3_k4 = 153. * powr<2>(l1) * powr<5>(p);
        const auto _cse4_k4 = 198. * powr<7>(p);
        const auto _cse5_k4 = powr<-1>(p);
        const auto _cse6_k4 = -72. * powr<6>(l1) * p;
        const auto _cse7_k4 = -372. * powr<4>(l1) * powr<3>(p);
        const auto _cse8_k4 = -378. * powr<2>(l1) * powr<5>(p);
        const auto _cse9_k4 = -99. * powr<7>(p);
        const auto _cse10_k4 = -80. * powr<2>(cosl1p2);
        const auto _cse11_k4 = 288. * powr<5>(l1) * powr<2>(p);
        const auto _cse12_k4 = 964. * powr<3>(l1) * powr<4>(p);
        const auto _cse13_k4 = 462. * l1 * powr<6>(p);
        const auto _cse14_k4 = _cse1_k4 + _cse2_k4 + _cse3_k4 + _cse4_k4;
        const auto _cse15_k4 = 144. * powr<5>(l1) * powr<2>(p);
        const auto _cse16_k4 = 482. * powr<3>(l1) * powr<4>(p);
        const auto _cse17_k4 = 231. * l1 * powr<6>(p);
        const auto _cse18_k4 = -72. * powr<7>(l1);
        const auto _cse19_k4 = -1164. * powr<5>(l1) * powr<2>(p);
        const auto _cse20_k4 = -2880. * powr<3>(l1) * powr<4>(p);
        const auto _cse21_k4 = -1386. * l1 * powr<6>(p);
        const auto _cse22_k4 = _cse18_k4 + _cse19_k4 + _cse20_k4 + _cse21_k4;
        const auto _cse23_k4 = _cse14_k4 * cosl1p2;
        const auto _cse24_k4 = -50. * _interp4;
        const auto _cse25_k4 = 50. * _interp5;
        const auto _cse26_k4 = _cse24_k4 + _cse25_k4;
        const auto _cse27_k4 = _cse26_k4 * _interp2 * powr<6>(k);
        const auto _cse28_k4 = 1. + powr<6>(k);
        const auto _cse29_k4 = _cse28_k4 * _interp3 * _interp4;
        // clang-format off
        _acc += fma(0.0303030303030303, _cse5_k4 * _den11 * _den16 * _den2 * _den21 * _den4 * _den8 * _interp16 * _interp20 * _interp21 * (_cse27_k4 + _cse29_k4 + _interp1 * _interp2 * (1. + 1. * powr<6>(k))) * (_cse6_k4 + _cse7_k4 + _cse8_k4 + _cse9_k4 + (_cse11_k4 + _cse12_k4 + _cse13_k4) * cosl1p2 + _cse14_k4 * powr<2>(cosl1p2) + 32. * powr<6>(cosl1p2) * powr<4>(l1) * powr<3>(p) + powr<5>(cosl1p1) * powr<4>(l1) * powr<2>(p) * (16. * l1 - 16. * cosl1p2 * p) + powr<4>(cosl1p1) * powr<3>(l1) * p * (-96. * powr<3>(l1) + 232. * cosl1p2 * powr<2>(l1) * p + (-240. - 72. * powr<2>(cosl1p2)) * l1 * powr<2>(p) + 176. * cosl1p2 * powr<3>(p)) + powr<5>(cosl1p2) * (-208. * powr<5>(l1) * powr<2>(p) - 448. * powr<3>(l1) * powr<4>(p)) + powr<3>(cosl1p1) * powr<2>(l1) * (48. * powr<5>(l1) - 336. * cosl1p2 * powr<4>(l1) * p + (200. + 496. * powr<2>(cosl1p2)) * powr<3>(l1) * powr<2>(p) + (-808. + _cse10_k4) * cosl1p2 * powr<2>(l1) * powr<3>(p) + (128. + 304. * powr<2>(cosl1p2)) * l1 * powr<4>(p) + 48. * cosl1p2 * powr<5>(p)) + powr<4>(cosl1p2) * (192. * powr<6>(l1) * p + 1244. * powr<4>(l1) * powr<3>(p) + 1254. * powr<2>(l1) * powr<5>(p)) + powr<3>(cosl1p2) * (-48. * powr<7>(l1) - 776. * powr<5>(l1) * powr<2>(p) - 1920. * powr<3>(l1) * powr<4>(p) - 924. * l1 * powr<6>(p)) + cosl1p1 * (_cse15_k4 + _cse16_k4 + _cse17_k4 + _cse23_k4 + _cse22_k4 * powr<2>(cosl1p2) + 96. * powr<5>(cosl1p2) * powr<4>(l1) * powr<3>(p) + powr<4>(cosl1p2) * (-520. * powr<5>(l1) * powr<2>(p) - 1120. * powr<3>(l1) * powr<4>(p)) + powr<3>(cosl1p2) * (384. * powr<6>(l1) * p + 2488. * powr<4>(l1) * powr<3>(p) + 2508. * powr<2>(l1) * powr<5>(p))) + powr<2>(cosl1p1) * l1 * (24. * powr<5>(l1) * p + 56. * powr<3>(l1) * powr<3>(p) + 40. * powr<4>(cosl1p2) * powr<3>(l1) * powr<3>(p) - 72. * l1 * powr<5>(p) + powr<3>(cosl1p2) * (-16. * powr<4>(l1) * powr<2>(p) - 544. * powr<2>(l1) * powr<4>(p)) + powr<2>(cosl1p2) * (-144. * powr<5>(l1) * p + 436. * powr<3>(l1) * powr<3>(p) + 1302. * l1 * powr<5>(p)) + cosl1p2 * (72. * powr<6>(l1) + 12. * powr<4>(l1) * powr<2>(p) - 704.0000000000001 * powr<2>(l1) * powr<4>(p) - 462. * powr<6>(p)))), fma(0.0303030303030303, _cse5_k4 * _den11 * _den14 * _den2 * _den21 * _den4 * _den6 * _interp11 * _interp16 * _interp17 * (_cse27_k4 + _cse29_k4 + _interp1 * _interp2 * (1. + 1. * powr<6>(k))) * (_cse6_k4 + _cse7_k4 + _cse8_k4 + _cse9_k4 + (_cse15_k4 + _cse16_k4 + _cse17_k4) * cosl1p2 + 16. * powr<5>(cosl1p2) * powr<5>(l1) * powr<2>(p) + 32. * powr<6>(cosl1p1) * powr<4>(l1) * powr<3>(p) + powr<5>(cosl1p1) * powr<3>(l1) * powr<2>(p) * (-208. * powr<2>(l1) + 96. * cosl1p2 * l1 * p - 448. * powr<2>(p)) + powr<4>(cosl1p2) * (-96. * powr<6>(l1) * p - 240. * powr<4>(l1) * powr<3>(p)) + powr<4>(cosl1p1) * powr<2>(l1) * p * (192. * powr<4>(l1) - 520. * cosl1p2 * powr<3>(l1) * p + (1244. + 40. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<2>(p) - 1120. * cosl1p2 * l1 * powr<3>(p) + 1254. * powr<4>(p)) + powr<3>(cosl1p2) * (48. * powr<7>(l1) + 200. * powr<5>(l1) * powr<2>(p) + 128. * powr<3>(l1) * powr<4>(p)) + powr<2>(cosl1p2) * (24. * powr<6>(l1) * p + 56. * powr<4>(l1) * powr<3>(p) - 72. * powr<2>(l1) * powr<5>(p)) + powr<3>(cosl1p1) * l1 * (-48. * powr<6>(l1) + 384. * cosl1p2 * powr<5>(l1) * p + (-776. - 16. * powr<2>(cosl1p2)) * powr<4>(l1) * powr<2>(p) + (2488. + _cse10_k4) * cosl1p2 * powr<3>(l1) * powr<3>(p) + (-1920. - 544. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<4>(p) + 2508. * cosl1p2 * l1 * powr<5>(p) - 924. * powr<6>(p)) + powr<2>(cosl1p1) * (_cse1_k4 + _cse2_k4 + _cse3_k4 + _cse4_k4 + _cse22_k4 * cosl1p2 - 72. * powr<4>(cosl1p2) * powr<4>(l1) * powr<3>(p) + powr<3>(cosl1p2) * (496. * powr<5>(l1) * powr<2>(p) + 304. * powr<3>(l1) * powr<4>(p)) + powr<2>(cosl1p2) * (-144. * powr<6>(l1) * p + 436. * powr<4>(l1) * powr<3>(p) + 1302. * powr<2>(l1) * powr<5>(p))) + cosl1p1 * (_cse11_k4 + _cse12_k4 + _cse13_k4 + _cse23_k4 - 16. * powr<5>(cosl1p2) * powr<4>(l1) * powr<3>(p) + powr<4>(cosl1p2) * (232. * powr<5>(l1) * powr<2>(p) + 176. * powr<3>(l1) * powr<4>(p)) + powr<3>(cosl1p2) * (-336. * powr<6>(l1) * p - 808. * powr<4>(l1) * powr<3>(p) + 48. * powr<2>(l1) * powr<5>(p)) + powr<2>(cosl1p2) * (72. * powr<7>(l1) + 12. * powr<5>(l1) * powr<2>(p) - 704.0000000000001 * powr<3>(l1) * powr<4>(p) - 462. * l1 * powr<6>(p)))), 0.));
        // clang-format on
      }
      { // subkernel 5
        const auto _den9 = powr<-1>(powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p));
        const auto _den18 =
            powr<-1>(_interp4 * _interp9 + _interp10 * (powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p)));
        // clang-format off
        _acc += -0.0303030303030303 * _den14 * _den18 * _den2 * _den4 * _den6 * _den9 * _interp11 * _interp12 * _interp13 * powr<-1>(p) * fma(72., powr<6>(l1) * p, fma(16., powr<5>(cosl1p2) * powr<5>(l1) * powr<2>(p), fma(372., powr<4>(l1) * powr<3>(p), fma(378., powr<2>(l1) * powr<5>(p), fma(99., powr<7>(p), fma(powr<5>(cosl1p1), powr<4>(l1) * powr<2>(p) * (-16. * l1 - 16. * cosl1p2 * p), fma(powr<4>(cosl1p1), powr<3>(l1) * p * (96. * powr<3>(l1) + 152. * cosl1p2 * powr<2>(l1) * p + (240. - 8. * powr<2>(cosl1p2)) * l1 * powr<2>(p) + 176. * cosl1p2 * powr<3>(p)), fma(powr<4>(cosl1p2), 96. * powr<6>(l1) * p + 240. * powr<4>(l1) * powr<3>(p), fma(powr<3>(cosl1p2), 48. * powr<7>(l1) + 200. * powr<5>(l1) * powr<2>(p) + 128. * powr<3>(l1) * powr<4>(p), fma(powr<3>(cosl1p1), powr<2>(l1) * (-48. * powr<5>(l1) + 48. * cosl1p2 * powr<4>(l1) * p + (-200. + 272. * powr<2>(cosl1p2)) * powr<3>(l1) * powr<2>(p) + cosl1p2 * (152. + 48. * powr<2>(cosl1p2)) * powr<2>(l1) * powr<3>(p) + (-128. + 400. * powr<2>(cosl1p2)) * l1 * powr<4>(p) + 48. * cosl1p2 * powr<5>(p)), fma(powr<2>(cosl1p2), -24. * powr<6>(l1) * p - 56. * powr<4>(l1) * powr<3>(p) + 72. * powr<2>(l1) * powr<5>(p), fma(cosl1p2, 144. * powr<5>(l1) * powr<2>(p) + 482. * powr<3>(l1) * powr<4>(p) + 231. * l1 * powr<6>(p), fma(powr<2>(cosl1p1), l1 * (-24. * powr<5>(l1) * p - 56. * powr<3>(l1) * powr<3>(p) - 8. * powr<4>(cosl1p2) * powr<3>(l1) * powr<3>(p) + 72. * l1 * powr<5>(p) + powr<3>(cosl1p2) * (-272. * powr<4>(l1) * powr<2>(p) - 400. * powr<2>(l1) * powr<4>(p)) + powr<2>(cosl1p2) * (-288. * powr<5>(l1) * p - 1420. * powr<3>(l1) * powr<3>(p) - 1158. * l1 * powr<5>(p)) + cosl1p2 * (-72. * powr<6>(l1) - 588. * powr<4>(l1) * powr<2>(p) - 1088. * powr<2>(l1) * powr<4>(p) - 462. * powr<6>(p))), fma(cosl1p1, -144. * powr<5>(l1) * powr<2>(p) - 16. * powr<5>(cosl1p2) * powr<4>(l1) * powr<3>(p) - 482. * powr<3>(l1) * powr<4>(p) - 231. * l1 * powr<6>(p) + powr<4>(cosl1p2) * (-152. * powr<5>(l1) * powr<2>(p) - 176. * powr<3>(l1) * powr<4>(p)) + powr<3>(cosl1p2) * (48. * powr<6>(l1) * p + 152. * powr<4>(l1) * powr<3>(p) + 48. * powr<2>(l1) * powr<5>(p)) + powr<2>(cosl1p2) * (72. * powr<7>(l1) + 588. * powr<5>(l1) * powr<2>(p) + 1088. * powr<3>(l1) * powr<4>(p) + 462. * l1 * powr<6>(p)) + cosl1p2 * (84. * powr<6>(l1) * p + 214. * powr<4>(l1) * powr<3>(p) + 297. * powr<2>(l1) * powr<5>(p) + 198. * powr<7>(p)), 0.)))))))))))))) * fma(_interp2, (-50. * _interp4 + 50. * _interp5) * powr<6>(k), fma(_interp3, _interp4 * (1. + powr<6>(k)), fma(_interp1, _interp2 * (1. + 1. * powr<6>(k)), 0.)));
        // clang-format on
      }
      return _acc;
    }

    static KOKKOS_INLINE_FUNCTION auto
    constant(const double &p, const double &k, const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA3,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZAcbc,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA4,
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