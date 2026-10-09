#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class Zc_kernel
  {
  public:
    using Regulator = _Regulator; // clang-format off


static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double &l1, const double &cos1, const double &p, const double &k,
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
      const auto _interp1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA(1.02 * pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = atan2(-1.732050807568877 * l1 * (l1 - 2. * cos1 * p), -powr<2>(l1) - 2. * cos1 * l1 * p + 2. * powr<2>(p));
      const auto _interp8 =
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
              powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
                sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                     2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
              atan2(-1.732050807568877 * l1 * (l1 - 2. * cos1 * p), -powr<2>(l1) - 2. * cos1 * l1 * p + 2. * powr<2>(p)));
      const auto _interp9 = atan2(1.732050807568877 * l1 * (l1 - 2. * cos1 * p), -powr<2>(l1) - 2. * cos1 * l1 * p + 2. * powr<2>(p));
      const auto _interp10 =
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
              powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
                sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                     2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
              atan2(1.732050807568877 * l1 * (l1 - 2. * cos1 * p), -powr<2>(l1) - 2. * cos1 * l1 * p + 2. * powr<2>(p)));
      const auto _interp11 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp12 = Zc(k);
      const auto _interp13 = Zc(sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp14 = atan2(-1.732050807568877 * l1 * (l1 + 2. * cos1 * p), -powr<2>(l1) + 2. * cos1 * l1 * p + 2. * powr<2>(p));
      const auto _interp15 =
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cos1 * l1 * p + powr<2>(p)),
              sqrt(1. + 3. * (-1. + powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) * powr<-2>(powr<2>(l1) + cos1 * l1 * p + powr<2>(p))),
              atan2(-1.732050807568877 * l1 * (l1 + 2. * cos1 * p), -powr<2>(l1) + 2. * cos1 * l1 * p + 2. * powr<2>(p)));
      const auto _interp16 = atan2(1.732050807568877 * l1 * (l1 + 2. * cos1 * p), -powr<2>(l1) + 2. * cos1 * l1 * p + 2. * powr<2>(p));
      const auto _interp17 =
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) + cos1 * l1 * p + powr<2>(p)),
              sqrt(1. + 3. * (-1. + powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) * powr<-2>(powr<2>(l1) + cos1 * l1 * p + powr<2>(p))),
              atan2(1.732050807568877 * l1 * (l1 + 2. * cos1 * p), -powr<2>(l1) + 2. * cos1 * l1 * p + 2. * powr<2>(p)));
      const auto _interp18 = dtZc(k);
      const auto _interp19 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp20 = RBdot(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp21 = Zc(1.02 * k);
      const auto _interp22 = Zc(sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _cse1 = powr<2>(l1);
      const auto _cse2 = powr<2>(cos1);
      const auto _cse3 = -1. + _cse2;
      const auto _cse4 = powr<6>(k);
      const auto _cse5 = _interp2 * _interp4;
      const auto _cse6 = _cse1 * _interp6;
      const auto _cse7 = _cse5 + _cse6;
      const auto _cse8 = powr<2>(p);
      return fma(-3.,
                 _cse3 * powr<-1>(1. + _cse4) * powr<-2>(_cse7) * _interp10 *
                   ((1. + 1. * _cse4) * _interp1 * _interp2 + (1. + _cse4) * _interp3 * _interp4 +
                    _cse4 * _interp2 * (-50. * _interp4 + 50. * _interp5)) *
                   _interp8 * powr<-1>(_interp11 * _interp12 + _interp13 * (_cse1 + _cse8 - 2. * cos1 * l1 * p)),
                 fma(-3.,
                     _cse3 * powr<-1>(_cse7) * _interp15 * _interp17 *
                       (_interp18 * _interp19 + _interp12 * _interp20 + _interp19 * (-50. * _interp12 + 50. * _interp21)) *
                       powr<-2>(_interp12 * _interp19 + _interp22 * (_cse1 + _cse8 + 2. * cos1 * l1 * p)),
                     0.)); // clang-format off

}

static KOKKOS_FORCEINLINE_FUNCTION auto constant(const double &p, const double &k,
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
using DiFfRG::Zc_kernel;