#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_INLINE_FUNCTION auto
    kernel(const double &l1, const double &cos1, const double &p, const double &k,
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
      const auto _interp1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp8 = ZA(sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp9 =
          atan2(2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p), 1.732050807568877 * p * (-2. * cos1 * l1 + p));
      const auto _interp10 = ZA3(
          0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
          powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
              sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                   2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
          atan2(2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p), -1.732050807568877 * (2. * cos1 * l1 - p) * p));
      const auto _interp11 =
          atan2(2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p), 1.732050807568877 * (2. * cos1 * l1 - p) * p);
      const auto _interp12 = ZA3(
          0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
          powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
              sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                   2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
          atan2(2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p), 1.732050807568877 * (2. * cos1 * l1 - p) * p));
      const auto _interp13 = atan2(4. * powr<2>(l1) - 3. * powr<2>(p), 6.928203230275509 * cos1 * l1 * p);
      const auto _interp14 = ZA4tadpole(
          0.408248290463863 * sqrt(4. * powr<2>(l1) + 3. * powr<2>(p)),
          powr<-1>(4. * powr<2>(l1) + 3. * powr<2>(p)) *
              sqrt(48. * powr<2>(cos1) * powr<2>(l1) * powr<2>(p) + powr<2>(4. * powr<2>(l1) - 3. * powr<2>(p))),
          atan2(4. * powr<2>(l1) - 3. * powr<2>(p), 6.928203230275509 * cos1 * l1 * p));
      const auto _interp15 = ZAcbc(
          0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
          powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
              sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                   2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
          atan2(2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p), -1.732050807568877 * (2. * cos1 * l1 - p) * p));
      const auto _interp16 = ZAcbc(
          0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
          powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
              sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                   2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
          atan2(2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p), 1.732050807568877 * (2. * cos1 * l1 - p) * p));
      const auto _interp17 = dtZc(k);
      const auto _interp18 = Zc(k);
      const auto _interp19 = Zc(1.02 * k);
      const auto _interp20 = Zc(l1);
      const auto _interp21 = Zc(sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _den1 = powr<-1>(1. + powr<6>(k));
      const auto _den2 = powr<-1>(1. + powr<6>(k));
      const auto _den3 = powr<-2>(_interp18 * _interp2 + _interp20 * powr<2>(l1));
      const auto _den4 = powr<-2>(_interp2 * _interp4 + _interp6 * powr<2>(l1));
      const auto _den5 = powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _den6 = powr<-1>(_interp18 * _interp7 + _interp21 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _den7 = powr<-1>(_interp4 * _interp7 + _interp8 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _cse1 = -6. * powr<2>(cos1);
      const auto _cse2 = 6. + _cse1;
      const auto _cse3 = 3. * powr<2>(cos1);
      const auto _cse4 = -3. + _cse3;
      const auto _cse5 = powr<-2>(p);
      return fma(
          -1.,
          _cse5 * _den4 * _interp14 * (7. - powr<2>(cos1)) *
              (_interp3 * _interp4 + _interp2 * (_interp1 + 50. * _den1 * (-_interp4 + _interp5) * powr<6>(k))),
          fma(2.,
              _cse5 * _den3 * _den6 * _interp15 * _interp16 *
                  (_interp17 * _interp2 + (-50. * _interp18 + 50. * _interp19) * _interp2 + _interp18 * _interp3) *
                  (-1. + powr<2>(cos1)) * powr<2>(l1),
              fma(-4.,
                  _cse5 * _den2 * _den4 * _den5 * _den7 * _interp10 * _interp12 *
                      (_interp2 * (-50. * _interp4 + 50. * _interp5) * powr<6>(k) +
                       _interp3 * _interp4 * (1. + powr<6>(k)) + _interp1 * _interp2 * (1. + 1. * powr<6>(k))) *
                      (_cse4 * powr<-2>(_cse5) +
                       powr<-1>(_cse5) * (-8. + 7. * powr<2>(cos1) + powr<4>(cos1)) * powr<2>(l1) +
                       _cse4 * powr<4>(l1) + _cse2 * cos1 * powr<3>(l1) * p + _cse2 * cos1 * l1 * powr<3>(p)),
                  0.)));
    }

    static KOKKOS_INLINE_FUNCTION auto
    constant(const double &p, const double &k,
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
using DiFfRG::ZA_kernel;