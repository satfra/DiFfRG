#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA_kernel
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
      const auto _interp7 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp8 = ZA(sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp9 = atan2(-1.732050807568877 * p * (-2. * cos1 * l1 + p), 2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p));
      const auto _interp10 = ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
                                 powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
                                   sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                                        2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
                                 atan2(-1.732050807568877 * p * (-2. * cos1 * l1 + p), 2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p)));
      const auto _interp11 = atan2(1.732050807568877 * p * (-2. * cos1 * l1 + p), 2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p));
      const auto _interp12 = ZA3(0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
                                 powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
                                   sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                                        2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
                                 atan2(1.732050807568877 * p * (-2. * cos1 * l1 + p), 2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p)));
      const auto _interp13 = atan2(-6.928203230275509 * cos1 * l1 * p, 4. * powr<2>(l1) - 3. * powr<2>(p));
      const auto _interp14 =
        ZA4tadpole(0.408248290463863 * sqrt(4. * powr<2>(l1) + 3. * powr<2>(p)),
                   powr<-1>(4. * powr<2>(l1) + 3. * powr<2>(p)) *
                     sqrt(48. * powr<2>(cos1) * powr<2>(l1) * powr<2>(p) + powr<2>(4. * powr<2>(l1) - 3. * powr<2>(p))),
                   atan2(-6.928203230275509 * cos1 * l1 * p, 4. * powr<2>(l1) - 3. * powr<2>(p)));
      const auto _interp15 =
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
              powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
                sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                     2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
              atan2(-1.732050807568877 * p * (-2. * cos1 * l1 + p), 2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p)));
      const auto _interp16 =
        ZAcbc(0.816496580927726 * sqrt(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)),
              powr<-1>(powr<2>(l1) - cos1 * l1 * p + powr<2>(p)) *
                sqrt(powr<4>(l1) - 2. * cos1 * powr<3>(l1) * p + (-1. + 4. * powr<2>(cos1)) * powr<2>(l1) * powr<2>(p) -
                     2. * cos1 * l1 * powr<3>(p) + powr<4>(p)),
              atan2(1.732050807568877 * p * (-2. * cos1 * l1 + p), 2. * powr<2>(l1) - 2. * cos1 * l1 * p - powr<2>(p)));
      const auto _interp17 = dtZc(k);
      const auto _interp18 = Zc(k);
      const auto _interp19 = Zc(1.02 * k);
      const auto _interp20 = Zc(l1);
      const auto _interp21 = Zc(sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _cse1 = powr<6>(k);
      const auto _cse2 = 1. + _cse1;
      const auto _cse3 = powr<-1>(_cse2);
      const auto _cse7 = powr<2>(p);
      const auto _cse4 = powr<-1>(_cse7);
      const auto _cse5 = powr<2>(l1);
      const auto _cse6 = powr<2>(cos1);
      const auto _cse8 = -6. * _cse6;
      const auto _cse9 = 6. + _cse8;
      const auto _cse10 = 3. * _cse6;
      const auto _cse11 = -3. + _cse10;
      const auto _cse12 = -2. * cos1 * l1 * p;
      const auto _cse13 = _cse12 + _cse5 + _cse7;
      const auto _cse14 = -50. * _interp4;
      const auto _cse15 = 50. * _interp5;
      const auto _cse16 = _cse14 + _cse15;
      const auto _cse17 = _cse1 * _cse16 * _interp2;
      const auto _cse18 = 1. + _cse1;
      const auto _cse19 = _cse18 * _interp3 * _interp4;
      const auto _cse20 = 1. * _cse1;
      const auto _cse21 = 1. + _cse20;
      const auto _cse22 = _cse21 * _interp1 * _interp2;
      const auto _cse23 = _cse17 + _cse19 + _cse22;
      const auto _cse24 = _interp2 * _interp4;
      const auto _cse25 = _cse5 * _interp6;
      const auto _cse26 = _cse24 + _cse25;
      const auto _cse27 = powr<-2>(_cse26);
      return fma(_cse23, _cse27 * _cse3 * _cse4 * (-7. + _cse6) * _interp14,
                 fma(2.,
                     _cse4 * _cse5 * (-1. + _cse6) * _interp15 * _interp16 * powr<-2>(_interp18 * _interp2 + _cse5 * _interp20) *
                       (_interp17 * _interp2 + (-50. * _interp18 + 50. * _interp19) * _interp2 + _interp18 * _interp3) *
                       powr<-1>(_cse13 * _interp21 + _interp18 * _interp7),
                     fma(-4.,
                         powr<-1>(_cse13) * _cse23 * _cse27 * _cse3 * _cse4 * _interp10 * _interp12 *
                           powr<-1>(_interp4 * _interp7 + _cse13 * _interp8) *
                           (_cse11 * powr<2>(_cse5) + _cse5 * (-8. + 7. * _cse6 + powr<2>(_cse6)) * _cse7 + _cse11 * powr<2>(_cse7) +
                            _cse9 * cos1 * powr<3>(l1) * p + _cse9 * cos1 * l1 * powr<3>(p)),
                         0.))); // clang-format off

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
using DiFfRG::ZA_kernel;