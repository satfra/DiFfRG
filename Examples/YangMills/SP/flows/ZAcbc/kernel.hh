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
      const auto _interp1 = RB(powr<2>(k), fma(-2., cosl1p2 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp2 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp4 = RB(powr<2>(k), fma(-2., (cosl1p1 + cosl1p2) * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp5 = ZA(sqrt(fma(-2., (cosl1p1 + cosl1p2) * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp7 = ZAcbc(0.816496580927726 * sqrt(fma(-1., cosl1p2 * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp8 =
          ZAcbc(0.816496580927726 * sqrt(fma(-1., (cosl1p1 + cosl1p2) * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp9 = dtZc(k);
      const auto _interp10 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp11 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp12 = Zc(k);
      const auto _interp13 = Zc(1.02 * k);
      const auto _interp14 = Zc(l1);
      const auto _interp15 = RB(powr<2>(k), fma(2., cosl1p2 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp18 = ZAcbc(0.816496580927726 * sqrt(fma(cosl1p2, l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _interp19 = ZAcbc(sqrt(
          fma(0.6666666666666666, powr<2>(l1), fma(0.6666666666666666, (-cosl1p1 + cosl1p2) * l1 * p, powr<2>(p)))));
      const auto _interp20 = RB(powr<2>(k), fma(-2., cosl1p1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp22 = ZAcbc(sqrt(fma(0.6666666666666666, powr<2>(l1),
                                            fma(-0.6666666666666666, (2. * cosl1p1 + cosl1p2) * l1 * p, powr<2>(p)))));
      const auto _interp23 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp24 = ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp25 = ZA(l1);
      const auto _interp29 = Zc(sqrt(fma(-2., (cosl1p1 + cosl1p2) * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _den2 = powr<-1>(1. + powr<6>(k));
      const auto _den3 = powr<-2>(fma(_interp10, _interp12, fma(_interp14, powr<2>(l1), 0.)));
      const auto _den4 = powr<-2>(fma(_interp10, _interp2, fma(_interp25, powr<2>(l1), 0.)));
      const auto _den9 = powr<-1>(fma(-2., cosl1p1 * l1 * p, fma(-2., cosl1p2 * l1 * p, powr<2>(l1) + powr<2>(p))));
      const auto _den17 = powr<-1>(fma(
          _interp12, _interp4, fma(_interp29, powr<2>(l1) + (-2. * cosl1p1 - 2. * cosl1p2) * l1 * p + powr<2>(p), 0.)));
      const auto _den18 = powr<-1>(fma(
          _interp2, _interp4, fma(_interp5, powr<2>(l1) + (-2. * cosl1p1 - 2. * cosl1p2) * l1 * p + powr<2>(p), 0.)));
      const auto _interp16 = ZA(sqrt(powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p)));
      const auto _interp17 = ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p + powr<2>(p)));
      const auto _interp21 = Zc(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p)));
      const auto _interp30 = ZAcbc(
          sqrt(0.6666666666666666 * powr<2>(l1) - 0.6666666666666666 * (cosl1p1 + 2. * cosl1p2) * l1 * p + powr<2>(p)));
      const auto _interp31 = Zc(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p)));
      const auto _interp3 = ZA(sqrt(powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p)));
      const auto _interp6 = ZA3(
          sqrt(0.6666666666666666 * powr<2>(l1) - 0.6666666666666666 * (cosl1p1 + 2. * cosl1p2) * l1 * p + powr<2>(p)));
      const auto _interp26 = ZA(sqrt(powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p)));
      const auto _interp27 = ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cosl1p1 * l1 * p + powr<2>(p)));
      const auto _interp28 = Zc(sqrt(powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p)));
      // clang-format off
      using _T = decltype(_den17 + _den18 + _den2 + _den3 + _den4 + _den9 + _interp1 + _interp10 + _interp11 + _interp12 + _interp13 + _interp14 + _interp15 + _interp16 + _interp17 + _interp18 + _interp19 + _interp2 + _interp20 + _interp21 + _interp22 + _interp23 + _interp24 + _interp25 + _interp26 + _interp27 + _interp28 + _interp29 + _interp3 + _interp30 + _interp31 + _interp4 + _interp5 + _interp6 + _interp7 + _interp8 + _interp9 + cosl1p1 + cosl1p2 + k + l1 + p);
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _den8 = powr<-1>(powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p));
        const auto _den11 =
            powr<-1>(_interp12 * _interp20 + _interp21 * (powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p)));
        const auto _den14 =
            powr<-1>(_interp1 * _interp12 + _interp31 * (powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p)));
        const auto _den16 =
            powr<-1>(_interp15 * _interp2 + _interp16 * (powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p)));
        const auto _cse1_k1 = 2. * cosl1p2;
        const auto _cse2_k1 = _cse1_k1 + cosl1p1;
        const auto _cse3_k1 = 2. * cosl1p1 * cosl1p2;
        const auto _cse4_k1 = 2. * powr<2>(cosl1p2);
        const auto _cse5_k1 = -1. + _cse3_k1 + _cse4_k1;
        const auto _cse6_k1 = _cse5_k1 * l1;
        const auto _cse7_k1 = _interp11 * _interp12;
        const auto _cse8_k1 = -50. * _interp12;
        const auto _cse9_k1 = 50. * _interp13;
        const auto _cse10_k1 = _cse8_k1 + _cse9_k1;
        const auto _cse11_k1 = _cse10_k1 * _interp10;
        const auto _cse12_k1 = _interp10 * _interp9;
        const auto _cse13_k1 = _cse11_k1 + _cse12_k1 + _cse7_k1;
        _acc += fma(-0.25,
                    _cse5_k1 * _den14 * _den17 * _den2 * _den4 * _interp30 * _interp7 * _interp8 *
                        (_interp10 * (-50. * _interp2 + 50. * _interp24) * powr<6>(k) +
                         _interp11 * _interp2 * (1. + powr<6>(k)) + _interp10 * _interp23 * (1. + 1. * powr<6>(k))) *
                        (2. * cosl1p1 * l1 + 4. * cosl1p2 * l1 - 3. * p) * p,
                    fma(0.5,
                        _cse13_k1 * _cse2_k1 * _den11 * _den18 * _den3 * _den9 * _interp17 * _interp22 * _interp8 *
                            powr<2>(l1) * p * (_cse6_k1 + (cosl1p1 - cosl1p2) * p),
                        fma(-0.5,
                            _cse13_k1 * _cse2_k1 * _den11 * _den16 * _den3 * _den8 * _interp17 * _interp18 * _interp19 *
                                powr<2>(l1) * p * (_cse6_k1 + (2. * cosl1p1 + cosl1p2) * p),
                            0.)));
      }
      { // subkernel 2
        const auto _den1 = powr<-1>(1. + powr<6>(k));
        const auto _den5 = powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p));
        const auto _den6 = powr<-1>(powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p));
        const auto _den7 = powr<-1>(powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p));
        const auto _den10 =
            powr<-1>(_interp2 * _interp20 + _interp26 * (powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p)));
        const auto _den12 =
            powr<-1>(_interp2 * _interp20 + _interp26 * (powr<2>(l1) - 2. * cosl1p1 * l1 * p + powr<2>(p)));
        const auto _den13 =
            powr<-1>(_interp1 * _interp2 + _interp3 * (powr<2>(l1) - 2. * cosl1p2 * l1 * p + powr<2>(p)));
        const auto _den15 =
            powr<-1>(_interp12 * _interp15 + _interp28 * (powr<2>(l1) + 2. * cosl1p2 * l1 * p + powr<2>(p)));
        const auto _cse1_k2 = -1.5 * powr<2>(l1) * p;
        const auto _cse2_k2 = 12. * powr<2>(cosl1p2);
        const auto _cse3_k2 = -8. * powr<2>(cosl1p2);
        // clang-format off
        _acc += fma(-2., _den13 * _den18 * _den3 * _den7 * _den9 * _interp6 * _interp7 * _interp8 * (_interp11 * _interp12 + _interp10 * (-50. * _interp12 + 50. * _interp13) + _interp10 * _interp9) * powr<2>(l1) * p * (2.25 * powr<2>(l1) * p - 4. * powr<4>(cosl1p2) * powr<2>(l1) * p + 1.5 * powr<3>(p) + powr<3>(cosl1p1) * l1 * p * (-cosl1p2 * l1 + 1. * p) + cosl1p2 * (-powr<3>(l1) - 4.5 * l1 * powr<2>(p)) + powr<3>(cosl1p2) * (2. * powr<3>(l1) + 8. * l1 * powr<2>(p)) + powr<2>(cosl1p2) * (_cse1_k2 - 2.75 * powr<3>(p)) + powr<2>(cosl1p1) * (_cse1_k2 + 1. * cosl1p2 * powr<3>(l1) - 5. * powr<2>(cosl1p2) * powr<2>(l1) * p + 6. * cosl1p2 * l1 * powr<2>(p) - 0.5 * powr<3>(p)) + cosl1p1 * ((-0.5 + 3. * powr<2>(cosl1p2)) * powr<3>(l1) + (-1.5 + _cse3_k2) * cosl1p2 * powr<2>(l1) * p + (-2.25 + _cse2_k2) * l1 * powr<2>(p) - 2.75 * cosl1p2 * powr<3>(p))), fma(0.5, _den12 * _den17 * _den2 * _den4 * _den6 * _interp22 * _interp27 * _interp8 * (_interp10 * (-50. * _interp2 + 50. * _interp24) * powr<6>(k) + _interp11 * _interp2 * (1. + powr<6>(k)) + _interp10 * _interp23 * (1. + 1. * powr<6>(k))) * p * (-4. * cosl1p2 * powr<3>(l1) + 8. * powr<3>(cosl1p2) * powr<3>(l1) - 6. * powr<2>(l1) * p + 4. * powr<2>(cosl1p2) * powr<2>(l1) * p + powr<3>(cosl1p1) * l1 * (-4. * cosl1p2 * l1 - 14. * p) * p + 2. * cosl1p2 * l1 * powr<2>(p) - 3. * powr<3>(p) + powr<2>(cosl1p1) * (4. * cosl1p2 * powr<3>(l1) + 12. * powr<2>(l1) * p - 12. * powr<2>(cosl1p2) * powr<2>(l1) * p - 18. * cosl1p2 * l1 * powr<2>(p) + 6. * powr<3>(p)) + cosl1p1 * ((-2. + _cse2_k2) * powr<3>(l1) + (14. + _cse3_k2) * cosl1p2 * powr<2>(l1) * p + (7. - 4. * powr<2>(cosl1p2)) * l1 * powr<2>(p) + 6. * cosl1p2 * powr<3>(p))), fma(-0.5, _den10 * _den15 * _den4 * _den5 * _interp18 * _interp19 * _interp27 * (_interp11 * _interp2 + _interp10 * (_interp23 + 50. * _den1 * (-_interp2 + _interp24) * powr<6>(k))) * p * (-4. * cosl1p2 * powr<3>(l1) + 8. * powr<3>(cosl1p2) * powr<3>(l1) + 6. * powr<2>(l1) * p - 4. * powr<3>(cosl1p1) * cosl1p2 * powr<2>(l1) * p - 4. * powr<2>(cosl1p2) * powr<2>(l1) * p + 2. * cosl1p2 * l1 * powr<2>(p) + 3. * powr<3>(p) - 2. * powr<2>(cosl1p1) * l1 * (-2. * cosl1p2 * powr<2>(l1) + l1 * p + 6. * powr<2>(cosl1p2) * l1 * p + 5. * cosl1p2 * powr<2>(p)) + cosl1p1 * (2. * (-1. + 6. * powr<2>(cosl1p2)) * powr<3>(l1) + (-5. + 4. * powr<2>(cosl1p2)) * l1 * powr<2>(p) + 6. * cosl1p2 * powr<3>(p) + powr<2>(l1) * (6. * cosl1p2 * p - 8. * powr<3>(cosl1p2) * p))), 0.)));
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
using DiFfRG::ZAcbc_kernel;