#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename _Regulator> class Zc_kernel
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
      const auto _interp1 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp2 = ZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp3 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp4 = dtZA(pow(1. + powr<6>(k), 0.16666666666666666667));
      const auto _interp5 = ZA((1.02) * (pow(1. + powr<6>(k), 0.16666666666666666667)));
      const auto _interp6 = ZA(l1);
      const auto _interp7 =
          ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (-1.) * ((cos1) * ((l1) * (p))) + powr<2>(p))));
      const auto _interp8 = RB(powr<2>(k), powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _interp9 = Zc(k);
      const auto _interp10 = Zc(sqrt(powr<2>(l1) + (-2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)));
      const auto _interp11 = ZAcbc((0.816496580927726) * (sqrt(powr<2>(l1) + (cos1) * ((l1) * (p)) + powr<2>(p))));
      const auto _interp12 = RBdot(powr<2>(k), powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _interp13 = RB(powr<2>(k), powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p));
      const auto _interp14 = dtZc(k);
      const auto _interp15 = Zc((1.02) * (k));
      const auto _interp16 = Zc(sqrt(powr<2>(l1) + (2.) * ((cos1) * ((l1) * (p))) + powr<2>(p)));
      const auto _cse1 = powr<2>(l1);
      const auto _cse2 = powr<2>(cos1);
      const auto _cse3 = -1. + _cse2;
      const auto _cse4 = powr<6>(k);
      const auto _cse5 = (_interp2) * (_interp3);
      const auto _cse6 = (_cse1) * (_interp6);
      const auto _cse7 = _cse5 + _cse6;
      const auto _cse8 = powr<2>(p);
      return fma(
          -3.,
          (_cse3) *
              ((_interp1) * (_interp2) +
               (_interp4 + (50.) * ((-1.) * (_interp2) + _interp5) * ((_cse4) * (powr<-1>(1. + _cse4)))) * (_interp3)) *
              ((powr<-2>(_cse7)) *
               ((powr<2>(_interp7)) *
                (powr<-1>((_interp8) * (_interp9) + (_cse1 + _cse8 + (-2.) * ((cos1) * ((l1) * (p)))) * (_interp10))))),
          fma(-3.,
              (_cse3) *
                  ((_interp14 + (50.) * (_interp15) + (-50.) * (_interp9)) * (_interp13) + (_interp12) * (_interp9)) *
                  ((powr<-1>(_cse7)) *
                   ((powr<2>(_interp11)) * (powr<-2>((_interp13) * (_interp9) +
                                                     (_cse1 + _cse8 + (2.) * ((cos1) * ((l1) * (p)))) * (_interp16))))),
              0.));
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
using DiFfRG::Zc_kernel;