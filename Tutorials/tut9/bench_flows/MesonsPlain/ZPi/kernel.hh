#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"
#include "numtracer/sun/sun_data.hpp"
#ifndef NT_TRACE_COMPLEX
#define NT_TRACE_COMPLEX DiFfRG::complex<double>
#endif
#include "kernels.hh"
#include "numtrace_verdict.hh"

namespace DiFfRG
{
  template<typename REG>
  class MesonsPlain_ZPi_kernel
  {
    public:
    using Regulator = REG;

    #if NT_ZPI_MESONSPLAIN_VERDICT == 2   // Pure: the Complex -> Re projection is exact
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      double fenv[(DiFfRG::zpi_mesonsplain::nenv) > 0 ? (DiFfRG::zpi_mesonsplain::nenv) : 1];
      const double dr_0 = -f0 - 3.141592653589793 * T;
      const double dr_1 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const double dr_2 = 0.7071067811865475;
      const double dr_3 = hPhi;
      const double dr_4 = sqrt(rhoPhi);
      const double dr_5 = f0 + 3.141592653589793 * T;
      const double dr_6 = -f0 + p0 - 3.141592653589793 * T;
      const double dr_7 = 1. + sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_8 = -1. - sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_9 = f0 - p0 + 3.141592653589793 * T;
      const double dr_10 = -f0 - p0 - 3.141592653589793 * T;
      const double dr_11 = 1. + sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_12 = f0 + p0 + 3.141592653589793 * T;
      DiFfRG::zpi_mesonsplain::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = ntRe(DiFfRG::zpi_mesonsplain::tr1(fenv));
      const auto _interp2 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ntRe(DiFfRG::zpi_mesonsplain::tr3(fenv));
      const auto _interp5 = ntRe(DiFfRG::zpi_mesonsplain::tr6(fenv));
      const auto _interp6 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp7 = ntRe(DiFfRG::zpi_mesonsplain::tr8(fenv));
      const auto _interp8 = ntRe(DiFfRG::zpi_mesonsplain::tr10(fenv));
      const auto _interp9 = ntRe(DiFfRG::zpi_mesonsplain::tr12(fenv));
      const auto _interp10 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp11 = ntRe(DiFfRG::zpi_mesonsplain::tr14(fenv));
      const auto _interp12 = ntRe(DiFfRG::zpi_mesonsplain::tr16(fenv));
      const auto _interp13 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp14 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp15 = ZPi(-f0, l1);
      const auto _interp16 = ZPi(f0, l1);
      const auto _interp17 = ntRe(DiFfRG::zpi_mesonsplain::tr17(fenv));
      const auto _interp18 = ntRe(DiFfRG::zpi_mesonsplain::tr18(fenv));
      const auto _interp19 = ZSig(-f0, l1);
      const auto _interp20 = ntRe(DiFfRG::zpi_mesonsplain::tr19(fenv));
      const auto _interp21 = ZSig(f0, l1);
      const auto _interp22 = ntRe(DiFfRG::zpi_mesonsplain::tr20(fenv));
      const auto _interp23 = ntRe(DiFfRG::zpi_mesonsplain::tr21(fenv));
      const auto _interp24 = ntRe(DiFfRG::zpi_mesonsplain::tr22(fenv));
      const auto _interp25 = ntRe(DiFfRG::zpi_mesonsplain::tr23(fenv));
      const auto _interp26 = ntRe(DiFfRG::zpi_mesonsplain::tr24(fenv));
      const auto _interp27 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp28 = ZPi(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp29 = ntRe(DiFfRG::zpi_mesonsplain::tr25(fenv));
      const auto _interp30 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp31 = ZPi(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp32 = ntRe(DiFfRG::zpi_mesonsplain::tr26(fenv));
      const auto _interp33 = ZSig(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp34 = ntRe(DiFfRG::zpi_mesonsplain::tr27(fenv));
      const auto _interp35 = ZSig(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den2 = powr<-1>(_interp13 + d1V + _interp15 * (powr<2>(f0) + powr<2>(l1)));
      const auto _den3 = powr<-1>(_interp13 + d1V + _interp16 * (powr<2>(f0) + powr<2>(l1)));
      const auto _den4 = powr<-1>(_interp13 + d1V + _interp19 * (powr<2>(f0) + powr<2>(l1)) + 2. * d2V * rhoPhi);
      const auto _den5 = powr<-1>(_interp13 + d1V + _interp21 * (powr<2>(f0) + powr<2>(l1)) + 2. * d2V * rhoPhi);
      const auto _den6 = powr<-1>(_interp30 + d1V + _interp31 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
      const auto _den7 = powr<-1>(_interp27 + d1V + _interp28 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
      const auto _den8 = powr<-3>(powr<2>(_interp2 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den9 = powr<-2>(powr<2>(_interp2 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den10 = powr<-1>(_interp30 + d1V + _interp35 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
      const auto _den11 = powr<-1>(_interp27 + d1V + _interp33 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
      const auto _den12 = powr<-1>(powr<2>(_interp6 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den13 = powr<-1>(powr<2>(_interp6 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den14 = powr<-1>(powr<2>(_interp10 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _cse1 = powr<-1>(l1);
      const auto _cse2 = -_interp14;
      const auto _cse3 = _interp13 * etaPi0;
      const auto _cse4 = _cse2 + _cse3;
      const auto _cse5 = -_interp13 * etaPi0;
      const auto _cse6 = _cse5 + _interp14;
      // clang-format off
      return 0.01388888888888889 * fma(3., _cse4 * _den1 * _den2 * _den3 * _interp12 * d2V, fma(-6., _cse4 * _den1 * _den2 * _den3 * _interp17 * d2V, fma(3., _cse4 * _den1 * _den4 * _den5 * _interp20 * d2V, fma(-6., _cse4 * _den1 * _den4 * _den5 * _interp23 * d2V, fma(-12., _cse1 * _den1 * _den14 * _den9 * _interp11 * _interp3 * powr<2>(hPhi), fma(24., _cse1 * _den1 * _den8 * _interp3 * _interp4 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den13 * _den9 * _interp3 * _interp7 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den12 * _den9 * _interp3 * _interp8 * powr<2>(hPhi), fma(-12., _cse6 * _den1 * _den2 * _den3 * _den4 * _interp18 * powr<2>(d2V) * rhoPhi, fma(-12., _cse6 * _den1 * _den2 * _den4 * _den5 * _interp25 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den4 * _den5 * _den7 * _interp26 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den4 * _den5 * _den6 * _interp29 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den11 * _den2 * _den3 * _interp32 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den10 * _den2 * _den3 * _interp34 * powr<2>(d2V) * rhoPhi, fma(4., _cse4 * _den1 * _den4 * _den5 * _interp22 * d3V * rhoPhi, fma(-8., _cse4 * _den1 * _den4 * _den5 * _interp24 * d3V * rhoPhi, fma(-12., _cse1 * _den1 * _den8 * _interp1 * _interp3 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1 * _den1 * _den13 * _den9 * _interp3 * _interp5 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1 * _den1 * _den14 * _den9 * _interp3 * _interp9 * powr<4>(hPhi) * rhoPhi, 0.)))))))))))))))))));
      // clang-format on
    }
    #elif NT_ZPI_MESONSPLAIN_VERDICT == 1   // RePart: real value via complex trace(s), re/im split
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      double fenv[(DiFfRG::zpi_mesonsplain::nenv) > 0 ? (DiFfRG::zpi_mesonsplain::nenv) : 1];
      const double dr_0 = -f0 - 3.141592653589793 * T;
      const double dr_1 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const double dr_2 = 0.7071067811865475;
      const double dr_3 = hPhi;
      const double dr_4 = sqrt(rhoPhi);
      const double dr_5 = f0 + 3.141592653589793 * T;
      const double dr_6 = -f0 + p0 - 3.141592653589793 * T;
      const double dr_7 = 1. + sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_8 = -1. - sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_9 = f0 - p0 + 3.141592653589793 * T;
      const double dr_10 = -f0 - p0 - 3.141592653589793 * T;
      const double dr_11 = 1. + sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_12 = f0 + p0 + 3.141592653589793 * T;
      DiFfRG::zpi_mesonsplain::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp2 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp9 = RF(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp16 = RF(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp22 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp23 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp24 = ZPi(-f0, l1);
      const auto _interp25 = ZPi(f0, l1);
      const auto _interp28 = ZSig(-f0, l1);
      const auto _interp30 = ZSig(f0, l1);
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den2 = powr<-1>(fma(_interp24, powr<2>(f0) + powr<2>(l1), _interp22 + d1V));
      const auto _den3 = powr<-1>(fma(_interp25, powr<2>(f0) + powr<2>(l1), _interp22 + d1V));
      const auto _den4 = powr<-1>(fma(_interp28, powr<2>(f0) + powr<2>(l1), fma(2., d2V * rhoPhi, _interp22 + d1V)));
      const auto _den5 = powr<-1>(fma(_interp30, powr<2>(f0) + powr<2>(l1), fma(2., d2V * rhoPhi, _interp22 + d1V)));
      const auto _den8 = powr<-3>(fma(powr<-2>(l1) * powr<2>(_interp2 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den9 = powr<-2>(fma(powr<-2>(l1) * powr<2>(_interp2 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den14 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp9 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den15 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp16 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _interp4 = ntIm(DiFfRG::zpi_mesonsplain::tr2(fenv));
      const auto _interp5 = ntIm(DiFfRG::zpi_mesonsplain::tr4(fenv));
      const auto _interp7 = ntRe(DiFfRG::zpi_mesonsplain::tr3(fenv));
      const auto _interp10 = ntIm(DiFfRG::zpi_mesonsplain::tr7(fenv));
      const auto _interp12 = ntRe(DiFfRG::zpi_mesonsplain::tr8(fenv));
      const auto _interp13 = ntIm(DiFfRG::zpi_mesonsplain::tr9(fenv));
      const auto _interp14 = ntRe(DiFfRG::zpi_mesonsplain::tr10(fenv));
      const auto _interp17 = ntIm(DiFfRG::zpi_mesonsplain::tr13(fenv));
      const auto _interp19 = ntRe(DiFfRG::zpi_mesonsplain::tr14(fenv));
      const auto _interp20 = ntIm(DiFfRG::zpi_mesonsplain::tr15(fenv));
      const auto _interp21 = ntRe(DiFfRG::zpi_mesonsplain::tr16(fenv));
      const auto _interp26 = ntRe(DiFfRG::zpi_mesonsplain::tr17(fenv));
      const auto _interp29 = ntRe(DiFfRG::zpi_mesonsplain::tr19(fenv));
      const auto _interp32 = ntRe(DiFfRG::zpi_mesonsplain::tr21(fenv));
      const auto _den12 = powr<-1>(powr<2>(_interp16 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den13 = powr<-1>(powr<2>(_interp9 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _cse5_k1 = sqrt(rhoPhi);
      const auto _interp1 = ntIm(DiFfRG::zpi_mesonsplain::tr0(fenv));
      const auto _interp6 = ntRe(DiFfRG::zpi_mesonsplain::tr1(fenv));
      const auto _interp8 = ntIm(DiFfRG::zpi_mesonsplain::tr5(fenv));
      const auto _interp11 = ntRe(DiFfRG::zpi_mesonsplain::tr6(fenv));
      const auto _interp15 = ntIm(DiFfRG::zpi_mesonsplain::tr11(fenv));
      const auto _interp18 = ntRe(DiFfRG::zpi_mesonsplain::tr12(fenv));
      const auto _interp27 = ntRe(DiFfRG::zpi_mesonsplain::tr18(fenv));
      const auto _interp31 = ntRe(DiFfRG::zpi_mesonsplain::tr20(fenv));
      const auto _interp33 = ntRe(DiFfRG::zpi_mesonsplain::tr22(fenv));
      const auto _interp34 = ntRe(DiFfRG::zpi_mesonsplain::tr23(fenv));
      const auto _interp35 = ntRe(DiFfRG::zpi_mesonsplain::tr24(fenv));
      const auto _interp36 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp37 = ZPi(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp38 = ntRe(DiFfRG::zpi_mesonsplain::tr25(fenv));
      const auto _interp39 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp40 = ZPi(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp41 = ntRe(DiFfRG::zpi_mesonsplain::tr26(fenv));
      const auto _interp42 = ZSig(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp43 = ntRe(DiFfRG::zpi_mesonsplain::tr27(fenv));
      const auto _interp44 = ZSig(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _cse7_k2 = sqrt(powr<3>(rhoPhi));
      // clang-format off
      using _T = decltype(_cse5_k1 + _cse7_k2 + _den1 + _den12 + _den13 + _den14 + _den15 + _den2 + _den3 + _den4 + _den5 + _den8 + _den9 + _interp1 + _interp10 + _interp11 + _interp12 + _interp13 + _interp14 + _interp15 + _interp16 + _interp17 + _interp18 + _interp19 + _interp2 + _interp20 + _interp21 + _interp22 + _interp23 + _interp24 + _interp25 + _interp26 + _interp27 + _interp28 + _interp29 + _interp3 + _interp30 + _interp31 + _interp32 + _interp33 + _interp34 + _interp35 + _interp36 + _interp37 + _interp38 + _interp39 + _interp4 + _interp40 + _interp41 + _interp42 + _interp43 + _interp44 + _interp5 + _interp6 + _interp7 + _interp8 + _interp9 + cos1 + d1V + d2V + d3V + etaPi0 + f0 + hPhi + l1 + p + p0 + rhoPhi);
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _cse1_k1 = powr<-1>(l1);
        const auto _cse2_k1 = -_interp23;
        const auto _cse3_k1 = _interp22 * etaPi0;
        const auto _cse4_k1 = _cse2_k1 + _cse3_k1;
        // clang-format off
        _acc += 0.04166666666666666 * fma(_cse4_k1, _den1 * _den2 * _den3 * _interp21 * d2V, fma(-2., _cse4_k1 * _den1 * _den2 * _den3 * _interp26 * d2V, fma(_cse4_k1, _den1 * _den4 * _den5 * _interp29 * d2V, fma(-2., _cse4_k1 * _den1 * _den4 * _den5 * _interp32 * d2V, fma(-4., _cse1_k1 * _den1 * _den14 * _den9 * _interp12 * _interp3 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den13 * _den9 * _interp14 * _interp3 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den15 * _den9 * _interp19 * _interp3 * powr<2>(hPhi), fma(4., _cse1_k1 * _den1 * _den12 * _den9 * _interp20 * _interp3 * powr<2>(hPhi), fma(-8., _cse1_k1 * _den1 * _den8 * _interp3 * _interp5 * powr<2>(hPhi), fma(8., _cse1_k1 * _den1 * _den8 * _interp3 * _interp7 * powr<2>(hPhi), fma(-2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den14 * _den9 * _interp10 * _interp3 * powr<3>(hPhi), fma(-2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den13 * _den9 * _interp13 * _interp3 * powr<3>(hPhi), fma(-2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den15 * _den9 * _interp17 * _interp3 * powr<3>(hPhi), fma(4., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den8 * _interp3 * _interp4 * powr<3>(hPhi), 0.))))))))))))));
        // clang-format on
      }
      { // subkernel 2
        const auto _den6 = powr<-1>(_interp39 + d1V + _interp40 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
        const auto _den7 = powr<-1>(_interp36 + d1V + _interp37 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
        const auto _den10 = powr<-1>(_interp39 + d1V + _interp44 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
        const auto _den11 = powr<-1>(_interp36 + d1V + _interp42 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
        const auto _cse1_k2 = powr<-1>(l1);
        const auto _cse2_k2 = -_interp22 * etaPi0;
        const auto _cse3_k2 = _cse2_k2 + _interp23;
        const auto _cse4_k2 = -_interp23;
        const auto _cse5_k2 = _interp22 * etaPi0;
        const auto _cse6_k2 = _cse4_k2 + _cse5_k2;
        // clang-format off
        _acc += 0.01388888888888889 * fma(-6., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den8 * _interp1 * _interp3 * powr<5>(hPhi), fma(3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den15 * _den9 * _interp15 * _interp3 * powr<5>(hPhi), fma(3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den14 * _den9 * _interp3 * _interp8 * powr<5>(hPhi), fma(-12., _cse3_k2 * _den1 * _den2 * _den3 * _den4 * _interp27 * powr<2>(d2V) * rhoPhi, fma(-12., _cse3_k2 * _den1 * _den2 * _den4 * _den5 * _interp34 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den4 * _den5 * _den7 * _interp35 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den4 * _den5 * _den6 * _interp38 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den11 * _den2 * _den3 * _interp41 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den10 * _den2 * _den3 * _interp43 * powr<2>(d2V) * rhoPhi, fma(4., _cse6_k2 * _den1 * _den4 * _den5 * _interp31 * d3V * rhoPhi, fma(-8., _cse6_k2 * _den1 * _den4 * _den5 * _interp33 * d3V * rhoPhi, fma(6., _cse1_k2 * _den1 * _den14 * _den9 * _interp11 * _interp3 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1_k2 * _den1 * _den15 * _den9 * _interp18 * _interp3 * powr<4>(hPhi) * rhoPhi, fma(-12., _cse1_k2 * _den1 * _den8 * _interp3 * _interp6 * powr<4>(hPhi) * rhoPhi, 0.))))))))))))));
        // clang-format on
      }
      return _acc;
    }
    #else                              // the imaginary part survives: genuinely complex
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      double fenv[(DiFfRG::zpi_mesonsplain::nenv) > 0 ? (DiFfRG::zpi_mesonsplain::nenv) : 1];
      const double dr_0 = -f0 - 3.141592653589793 * T;
      const double dr_1 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const double dr_2 = 0.7071067811865475;
      const double dr_3 = hPhi;
      const double dr_4 = sqrt(rhoPhi);
      const double dr_5 = f0 + 3.141592653589793 * T;
      const double dr_6 = -f0 + p0 - 3.141592653589793 * T;
      const double dr_7 = 1. + sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_8 = -1. - sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_9 = f0 - p0 + 3.141592653589793 * T;
      const double dr_10 = -f0 - p0 - 3.141592653589793 * T;
      const double dr_11 = 1. + sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_12 = f0 + p0 + 3.141592653589793 * T;
      DiFfRG::zpi_mesonsplain::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RF(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp4 = RF(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp5 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp6 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp7 = ZPi(-f0, l1);
      const auto _interp8 = ZPi(f0, l1);
      const auto _interp9 = ZSig(-f0, l1);
      const auto _interp10 = ZSig(f0, l1);
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den2 = powr<-1>(fma(_interp7, powr<2>(f0) + powr<2>(l1), _interp5 + d1V));
      const auto _den3 = powr<-1>(fma(_interp8, powr<2>(f0) + powr<2>(l1), _interp5 + d1V));
      const auto _den4 = powr<-1>(fma(_interp10, powr<2>(f0) + powr<2>(l1), fma(2., d2V * rhoPhi, _interp5 + d1V)));
      const auto _den5 = powr<-1>(fma(_interp9, powr<2>(f0) + powr<2>(l1), fma(2., d2V * rhoPhi, _interp5 + d1V)));
      const auto _den8 = powr<-3>(fma(powr<-2>(l1) * powr<2>(_interp1 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den9 = powr<-2>(fma(powr<-2>(l1) * powr<2>(_interp1 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den14 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp3 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den15 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp4 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _den12 = powr<-1>(powr<2>(_interp4 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den13 = powr<-1>(powr<2>(_interp3 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _cse5_k1 = sqrt(rhoPhi);
      const auto _interp11 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp12 = ZPi(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp13 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp14 = ZPi(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp15 = ZSig(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _interp16 = ZSig(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      const auto _cse7_k2 = sqrt(powr<3>(rhoPhi));
      // clang-format off
      using _T = decltype(complex<double>(0.,0.04166666666666666) + _cse5_k1 + _cse7_k2 + _den1 + _den12 + _den13 + _den14 + _den15 + _den2 + _den3 + _den4 + _den5 + _den8 + _den9 + DiFfRG::zpi_mesonsplain::tr0(fenv) + DiFfRG::zpi_mesonsplain::tr10(fenv) + DiFfRG::zpi_mesonsplain::tr11(fenv) + DiFfRG::zpi_mesonsplain::tr12(fenv) + DiFfRG::zpi_mesonsplain::tr13(fenv) + DiFfRG::zpi_mesonsplain::tr14(fenv) + DiFfRG::zpi_mesonsplain::tr15(fenv) + DiFfRG::zpi_mesonsplain::tr16(fenv) + DiFfRG::zpi_mesonsplain::tr17(fenv) + DiFfRG::zpi_mesonsplain::tr18(fenv) + DiFfRG::zpi_mesonsplain::tr19(fenv) + DiFfRG::zpi_mesonsplain::tr1(fenv) + DiFfRG::zpi_mesonsplain::tr20(fenv) + DiFfRG::zpi_mesonsplain::tr21(fenv) + DiFfRG::zpi_mesonsplain::tr22(fenv) + DiFfRG::zpi_mesonsplain::tr23(fenv) + DiFfRG::zpi_mesonsplain::tr24(fenv) + DiFfRG::zpi_mesonsplain::tr25(fenv) + DiFfRG::zpi_mesonsplain::tr26(fenv) + DiFfRG::zpi_mesonsplain::tr27(fenv) + DiFfRG::zpi_mesonsplain::tr2(fenv) + DiFfRG::zpi_mesonsplain::tr3(fenv) + DiFfRG::zpi_mesonsplain::tr4(fenv) + DiFfRG::zpi_mesonsplain::tr5(fenv) + DiFfRG::zpi_mesonsplain::tr6(fenv) + DiFfRG::zpi_mesonsplain::tr7(fenv) + DiFfRG::zpi_mesonsplain::tr8(fenv) + DiFfRG::zpi_mesonsplain::tr9(fenv) + _interp1 + _interp10 + _interp11 + _interp12 + _interp13 + _interp14 + _interp15 + _interp16 + _interp2 + _interp3 + _interp4 + _interp5 + _interp6 + _interp7 + _interp8 + _interp9 + cos1 + d1V + d2V + d3V + etaPi0 + f0 + hPhi + l1 + p + p0 + rhoPhi);
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _cse1_k1 = powr<-1>(l1);
        const auto _cse2_k1 = -_interp6;
        const auto _cse3_k1 = _interp5 * etaPi0;
        const auto _cse4_k1 = _cse2_k1 + _cse3_k1;
        // clang-format off
        _acc += complex<double>(0.,0.04166666666666666) * fma(complex<double>(0.,-1.), _cse4_k1 * _den1 * _den2 * _den3 * DiFfRG::zpi_mesonsplain::tr16(fenv) * d2V, fma(complex<double>(0.,2.), _cse4_k1 * _den1 * _den2 * _den3 * DiFfRG::zpi_mesonsplain::tr17(fenv) * d2V, fma(complex<double>(0.,-1.), _cse4_k1 * _den1 * _den4 * _den5 * DiFfRG::zpi_mesonsplain::tr19(fenv) * d2V, fma(complex<double>(0.,2.), _cse4_k1 * _den1 * _den4 * _den5 * DiFfRG::zpi_mesonsplain::tr21(fenv) * d2V, fma(complex<double>(0.,4.), _cse1_k1 * _den1 * _den13 * _den9 * DiFfRG::zpi_mesonsplain::tr10(fenv) * _interp2 * powr<2>(hPhi), fma(complex<double>(0.,4.), _cse1_k1 * _den1 * _den15 * _den9 * DiFfRG::zpi_mesonsplain::tr14(fenv) * _interp2 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den12 * _den9 * DiFfRG::zpi_mesonsplain::tr15(fenv) * _interp2 * powr<2>(hPhi), fma(complex<double>(0.,-8.), _cse1_k1 * _den1 * _den8 * DiFfRG::zpi_mesonsplain::tr3(fenv) * _interp2 * powr<2>(hPhi), fma(8., _cse1_k1 * _den1 * _den8 * DiFfRG::zpi_mesonsplain::tr4(fenv) * _interp2 * powr<2>(hPhi), fma(complex<double>(0.,4.), _cse1_k1 * _den1 * _den14 * _den9 * DiFfRG::zpi_mesonsplain::tr8(fenv) * _interp2 * powr<2>(hPhi), fma(2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den15 * _den9 * DiFfRG::zpi_mesonsplain::tr13(fenv) * _interp2 * powr<3>(hPhi), fma(-4., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den8 * DiFfRG::zpi_mesonsplain::tr2(fenv) * _interp2 * powr<3>(hPhi), fma(2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den14 * _den9 * DiFfRG::zpi_mesonsplain::tr7(fenv) * _interp2 * powr<3>(hPhi), fma(2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den13 * _den9 * DiFfRG::zpi_mesonsplain::tr9(fenv) * _interp2 * powr<3>(hPhi), 0.))))))))))))));
        // clang-format on
      }
      { // subkernel 2
        const auto _den6 = powr<-1>(_interp13 + d1V + _interp14 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
        const auto _den7 = powr<-1>(_interp11 + d1V + _interp12 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
        const auto _den10 = powr<-1>(_interp13 + d1V + _interp16 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
        const auto _den11 = powr<-1>(_interp11 + d1V + _interp15 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
        const auto _cse1_k2 = powr<-1>(l1);
        const auto _cse2_k2 = -_interp5 * etaPi0;
        const auto _cse3_k2 = _cse2_k2 + _interp6;
        const auto _cse4_k2 = -_interp6;
        const auto _cse5_k2 = _interp5 * etaPi0;
        const auto _cse6_k2 = _cse4_k2 + _cse5_k2;
        // clang-format off
        _acc += complex<double>(0.,0.01388888888888889) * fma(6., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den8 * DiFfRG::zpi_mesonsplain::tr0(fenv) * _interp2 * powr<5>(hPhi), fma(-3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den15 * _den9 * DiFfRG::zpi_mesonsplain::tr11(fenv) * _interp2 * powr<5>(hPhi), fma(-3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den14 * _den9 * DiFfRG::zpi_mesonsplain::tr5(fenv) * _interp2 * powr<5>(hPhi), fma(complex<double>(0.,12.), _cse3_k2 * _den1 * _den2 * _den3 * _den5 * DiFfRG::zpi_mesonsplain::tr18(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,12.), _cse3_k2 * _den1 * _den2 * _den4 * _den5 * DiFfRG::zpi_mesonsplain::tr23(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den4 * _den5 * _den7 * DiFfRG::zpi_mesonsplain::tr24(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den4 * _den5 * _den6 * DiFfRG::zpi_mesonsplain::tr25(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den11 * _den2 * _den3 * DiFfRG::zpi_mesonsplain::tr26(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den10 * _den2 * _den3 * DiFfRG::zpi_mesonsplain::tr27(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-4.), _cse6_k2 * _den1 * _den4 * _den5 * DiFfRG::zpi_mesonsplain::tr20(fenv) * d3V * rhoPhi, fma(complex<double>(0.,8.), _cse6_k2 * _den1 * _den4 * _den5 * DiFfRG::zpi_mesonsplain::tr22(fenv) * d3V * rhoPhi, fma(complex<double>(0.,-6.), _cse1_k2 * _den1 * _den15 * _den9 * DiFfRG::zpi_mesonsplain::tr12(fenv) * _interp2 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,12.), _cse1_k2 * _den1 * _den8 * DiFfRG::zpi_mesonsplain::tr1(fenv) * _interp2 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-6.), _cse1_k2 * _den1 * _den14 * _den9 * DiFfRG::zpi_mesonsplain::tr6(fenv) * _interp2 * powr<4>(hPhi) * rhoPhi, 0.))))))))))))));
        // clang-format on
      }
      return _acc;
    }
    #endif

    static KOKKOS_INLINE_FUNCTION auto constant(const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const auto& ZPi, const auto& ZSig)
    {
      return 0.;
    }
    private:
    static KOKKOS_INLINE_FUNCTION auto RB(const auto &k2, const auto &p2) { return REG::RB(k2, p2); }
    static KOKKOS_INLINE_FUNCTION auto RF(const auto &k2, const auto &p2) { return REG::RF(k2, p2); }
    static KOKKOS_INLINE_FUNCTION auto RBdot(const auto &k2, const auto &p2) { return REG::RBdot(k2, p2); }
    static KOKKOS_INLINE_FUNCTION auto RFdot(const auto &k2, const auto &p2) { return REG::RFdot(k2, p2); }
    static KOKKOS_INLINE_FUNCTION auto dq2RB(const auto &k2, const auto &p2) { return REG::dq2RB(k2, p2); }
    static KOKKOS_INLINE_FUNCTION auto dq2RF(const auto &k2, const auto &p2) { return REG::dq2RF(k2, p2); }

    static KOKKOS_INLINE_FUNCTION double ntRe(double x) { return x; }
    static KOKKOS_INLINE_FUNCTION double ntIm(double) { return 0.0; }
    template <class T> static KOKKOS_INLINE_FUNCTION auto ntRe(const T &z) -> decltype(z.real()) { return z.real(); }
    template <class T> static KOKKOS_INLINE_FUNCTION auto ntRe(const T &z) -> decltype(real(z)) requires (!requires { z.real(); }) { return real(z); }
    template <class T> static KOKKOS_INLINE_FUNCTION auto ntIm(const T &z) -> decltype(z.imag()) { return z.imag(); }
    template <class T> static KOKKOS_INLINE_FUNCTION auto ntIm(const T &z) -> decltype(imag(z)) requires (!requires { z.imag(); }) { return imag(z); }
  };
}
using DiFfRG::MesonsPlain_ZPi_kernel;