#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class Zc_kernel
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
      const auto _interp1 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp2 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp3 = ZA(sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp4 =
          atan2(-powr<2>(l1) + 4. * cos1 * l1 * p - powr<2>(p), 1.732050807568877 * (l1 - p) * (l1 + p));
      const auto _interp5 = ZAcbc(
          0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
          powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
              sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                   2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
          atan2(-powr<2>(l1) + 4. * cos1 * l1 * p - powr<2>(p), 1.732050807568877 * (powr<2>(l1) - powr<2>(p))));
      const auto _interp6 =
          atan2(-powr<2>(l1) + 4. * cos1 * l1 * p - powr<2>(p), 1.732050807568877 * (-powr<2>(l1) + powr<2>(p)));
      const auto _interp7 = ZAcbc(
          0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
          powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
              sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                   2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
          atan2(-powr<2>(l1) + 4. * cos1 * l1 * p - powr<2>(p), 1.732050807568877 * (-powr<2>(l1) + powr<2>(p))));
      const auto _interp8 = dtZc(k);
      const auto _interp9 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp10 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp11 = Zc(k);
      const auto _interp12 = Zc(1.02 * k);
      const auto _interp13 = Zc(l1);
      const auto _interp14 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp15 = ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp16 = ZA(l1);
      const auto _interp17 =
          atan2(-powr<2>(l1) - 2. * cos1 * l1 * p + 2. * powr<2>(p), -1.732050807568877 * l1 * (l1 - 2. * cos1 * p));
      const auto _interp18 = ZAcbc(
          0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
          powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
              sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                   2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
          atan2(-powr<2>(l1) - 2. * cos1 * l1 * p + 2. * powr<2>(p), -1.732050807568877 * l1 * (l1 - 2. * cos1 * p)));
      const auto _interp19 =
          atan2(-powr<2>(l1) - 2. * cos1 * l1 * p + 2. * powr<2>(p), 1.732050807568877 * l1 * (l1 - 2. * cos1 * p));
      const auto _interp20 = ZAcbc(
          0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
          powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
              sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                   2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
          atan2(-powr<2>(l1) - 2. * cos1 * l1 * p + 2. * powr<2>(p), 1.732050807568877 * l1 * (l1 - 2. * cos1 * p)));
      const auto _interp21 = Zc(sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _den1 = powr<-1>(1. + powr<6>(k));
      const auto _den2 = powr<-2>(_interp11 * _interp9 + _interp13 * powr<2>(l1));
      const auto _den3 = powr<-2>(_interp2 * _interp9 + _interp16 * powr<2>(l1));
      const auto _den4 = powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _den5 = powr<-1>(_interp1 * _interp11 + _interp21 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _den6 = powr<-1>(_interp1 * _interp2 + _interp3 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _cse1 = -1. + powr<2>(cos1);
      return fma(
          -3.,
          _cse1 * _den1 * _den3 * _den5 * _interp18 * _interp20 *
              ((50. * _interp15 - 50. * _interp2) * _interp9 * powr<6>(k) + _interp10 * _interp2 * (1. + powr<6>(k)) +
               _interp14 * _interp9 * (1. + 1. * powr<6>(k))),
          fma(-3.,
              _cse1 * _den2 * _den4 * _den6 * _interp5 * _interp7 *
                  (_interp10 * _interp11 + (-50. * _interp11 + 50. * _interp12) * _interp9 + _interp8 * _interp9) *
                  powr<2>(l1),
              0.));
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
using DiFfRG::Zc_kernel;