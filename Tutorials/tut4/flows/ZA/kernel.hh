#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class ZA_kernel
  {
  public:
    using Regulator = _Regulator;

    static KOKKOS_FORCEINLINE_FUNCTION auto
    kernel(const double &l1, const double &cos1, const double &p, const double &k,
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
      const auto _interp1 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _interp6 = ZA(l1);
      const auto _interp7 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _interp8 = ZA(sqrt(powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)));
      const auto _interp9 =
          ZA3((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cos1) * ((l1) * (p))) + powr<2>(p))));
      const auto _interp10 = ZA4((0.7071067811865475) * (sqrt(powr<2>(l1) + powr<2>(p))));
      const auto _interp11 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cos1) * ((l1) * (p))) + powr<2>(p))));
      const auto _interp12 = dtZc(k);
      const auto _interp13 = Zc(k);
      const auto _interp14 = Zc((1.02) * (k));
      const auto _interp15 = Zc(l1);
      const auto _interp16 = Zc(sqrt(powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)));
      const auto _cse1 = powr<2>(cos1);
      const auto _cse2 = powr<2>(l1);
      const auto _cse3 = powr<2>(p);
      const auto _cse4 = (-6.) * (_cse1);
      const auto _cse5 = 6. + _cse4;
      const auto _cse6 = (3.) * (_cse1);
      const auto _cse7 = -3. + _cse6;
      const auto _cse8 = (-2.) * ((cos1) * ((l1) * (p)));
      const auto _cse9 = _cse2 + _cse3 + _cse8;
      const auto _cse10 = powr<6>(k);
      const auto _cse11 = powr<-2>(p);
      const auto _cse12 = 1. + _cse10;
      const auto _cse13 = (_interp2) * (_interp4);
      const auto _cse14 = (_cse2) * (_interp6);
      const auto _cse15 = _cse13 + _cse14;
      const auto _cse16 = powr<-2>(_cse15);
      return fma(
          -1.,
          (_cse11) * (7. + (-1.) * (_cse1)) *
              ((_cse16) *
               ((_interp3) * (_interp4) +
                (_interp1 + (50.) * ((-1.) * (_interp4) + _interp5) * ((_cse10) * (powr<-1>(_cse12)))) * (_interp2)) *
               (_interp10)),
          fma(2.,
              (_cse11) * (-1. + _cse1) *
                  ((_cse2) *
                   ((_interp12) * (_interp2) + ((-50.) * (_interp13) + (50.) * (_interp14)) * (_interp2) +
                    (_interp13) * (_interp3)) *
                   ((powr<2>(_interp11)) * ((powr<-2>((_cse2) * (_interp15) + (_interp13) * (_interp2))) *
                                            (powr<-1>((_cse9) * (_interp16) + (_interp13) * (_interp7)))))),
              fma(-4.,
                  (powr<-1>(1. + _cse10)) *
                      ((_interp1) * (1. + (1.) * (_cse10)) * (_interp2) + (_cse12) * ((_interp3) * (_interp4)) +
                       (_cse10) * ((-50.) * (_interp4) + (50.) * (_interp5)) * (_interp2)) *
                      ((_cse11) *
                       ((_cse2) * (-8. + (7.) * (_cse1) + powr<4>(cos1)) * (_cse3) + (_cse7) * (powr<4>(l1)) +
                        (_cse5) * ((cos1) * ((powr<3>(l1)) * (p))) + (_cse5) * ((cos1) * ((l1) * (powr<3>(p)))) +
                        (_cse7) * (powr<4>(p))) *
                       ((_cse16) * ((powr<-1>(_cse9)) * ((powr<-1>((_interp4) * (_interp7) + (_cse9) * (_interp8))) *
                                                         (powr<2>(_interp9)))))),
                  0.)));
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
using DiFfRG::ZA_kernel;