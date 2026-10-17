#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/common/multi_value.hh"
#include "numtracer/sun/sun_data.hpp"
#ifndef NT_TRACE_COMPLEX
#define NT_TRACE_COMPLEX DiFfRG::complex<double>
#endif
#include "ZPi/kernels.hh"
#include "ZSig/kernels.hh"
#include "ZPi/numtrace_verdict.hh"
#include "ZSig/numtrace_verdict.hh"

namespace DiFfRG
{
  template<typename REG>
  class MesonsPlain_kernel
  {
    public:
    using Regulator = REG;

    #if (NT_ZPI_MESONSPLAIN_VERDICT == 2) && (NT_ZSIG_MESONSPLAIN_VERDICT == 2)   // Pure: the Complex -> Re projection is exact
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto ntgix0 = ZPi.index(-f0, l1);
      [[maybe_unused]] const auto ntgix1 = ZPi.index(f0, l1);
      [[maybe_unused]] const auto ntgix2 = ZPi.index(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntgix3 = ZPi.index(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntglk0 = ZPi.at(ntgix0);
      [[maybe_unused]] const auto ntglk1 = ZPi.at(ntgix1);
      [[maybe_unused]] const auto ntglk2 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk3 = ZSig.at(ntgix1);
      [[maybe_unused]] const auto ntglk4 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk5 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      constexpr auto ntnenv0 = 1;
      constexpr auto ntnenv1 = ntnenv0 > DiFfRG::zpi_mesonsplain::nenv ? ntnenv0 : DiFfRG::zpi_mesonsplain::nenv;
      constexpr auto ntnenv2 = ntnenv1 > DiFfRG::zsig_mesonsplain::nenv ? ntnenv1 : DiFfRG::zsig_mesonsplain::nenv;
      double fenv[ntnenv2];
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
      [[maybe_unused]] const auto ntt0_0 = DiFfRG::zpi_mesonsplain::tr3(fenv);
      [[maybe_unused]] const auto ntt0_1 = DiFfRG::zpi_mesonsplain::tr4(fenv);
      [[maybe_unused]] const auto ntt0_2 = DiFfRG::zpi_mesonsplain::tr2(fenv);
      [[maybe_unused]] const auto ntt0_3 = DiFfRG::zpi_mesonsplain::tr1(fenv);
      [[maybe_unused]] const auto ntt0_4 = DiFfRG::zpi_mesonsplain::tr0(fenv);
      [[maybe_unused]] const auto ntt0_5 = DiFfRG::zpi_mesonsplain::tr8(fenv);
      [[maybe_unused]] const auto ntt0_6 = DiFfRG::zpi_mesonsplain::tr7(fenv);
      [[maybe_unused]] const auto ntt0_7 = DiFfRG::zpi_mesonsplain::tr6(fenv);
      [[maybe_unused]] const auto ntt0_8 = DiFfRG::zpi_mesonsplain::tr5(fenv);
      [[maybe_unused]] const auto ntt0_9 = DiFfRG::zpi_mesonsplain::tr10(fenv);
      [[maybe_unused]] const auto ntt0_10 = DiFfRG::zpi_mesonsplain::tr9(fenv);
      [[maybe_unused]] const auto ntt0_11 = DiFfRG::zpi_mesonsplain::tr14(fenv);
      [[maybe_unused]] const auto ntt0_12 = DiFfRG::zpi_mesonsplain::tr13(fenv);
      [[maybe_unused]] const auto ntt0_13 = DiFfRG::zpi_mesonsplain::tr12(fenv);
      [[maybe_unused]] const auto ntt0_14 = DiFfRG::zpi_mesonsplain::tr11(fenv);
      [[maybe_unused]] const auto ntt0_15 = DiFfRG::zpi_mesonsplain::tr15(fenv);
      [[maybe_unused]] const auto ntt0_16 = DiFfRG::zpi_mesonsplain::tr16(fenv);
      [[maybe_unused]] const auto ntt0_17 = DiFfRG::zpi_mesonsplain::tr17(fenv);
      [[maybe_unused]] const auto ntt0_18 = DiFfRG::zpi_mesonsplain::tr18(fenv);
      [[maybe_unused]] const auto ntt0_19 = DiFfRG::zpi_mesonsplain::tr19(fenv);
      [[maybe_unused]] const auto ntt0_20 = DiFfRG::zpi_mesonsplain::tr21(fenv);
      [[maybe_unused]] const auto ntt0_21 = DiFfRG::zpi_mesonsplain::tr20(fenv);
      [[maybe_unused]] const auto ntt0_22 = DiFfRG::zpi_mesonsplain::tr22(fenv);
      [[maybe_unused]] const auto ntt0_23 = DiFfRG::zpi_mesonsplain::tr23(fenv);
      [[maybe_unused]] const auto ntt0_24 = DiFfRG::zpi_mesonsplain::tr24(fenv);
      [[maybe_unused]] const auto ntt0_25 = DiFfRG::zpi_mesonsplain::tr25(fenv);
      [[maybe_unused]] const auto ntt0_26 = DiFfRG::zpi_mesonsplain::tr26(fenv);
      [[maybe_unused]] const auto ntt0_27 = DiFfRG::zpi_mesonsplain::tr27(fenv);
      DiFfRG::zsig_mesonsplain::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      [[maybe_unused]] const auto ntt1_0 = DiFfRG::zsig_mesonsplain::tr1(fenv);
      [[maybe_unused]] const auto ntt1_1 = DiFfRG::zsig_mesonsplain::tr2(fenv);
      [[maybe_unused]] const auto ntt1_2 = DiFfRG::zsig_mesonsplain::tr0(fenv);
      [[maybe_unused]] const auto ntt1_3 = DiFfRG::zsig_mesonsplain::tr4(fenv);
      [[maybe_unused]] const auto ntt1_4 = DiFfRG::zsig_mesonsplain::tr3(fenv);
      [[maybe_unused]] const auto ntt1_5 = DiFfRG::zsig_mesonsplain::tr6(fenv);
      [[maybe_unused]] const auto ntt1_6 = DiFfRG::zsig_mesonsplain::tr5(fenv);
      [[maybe_unused]] const auto ntt1_7 = DiFfRG::zsig_mesonsplain::tr8(fenv);
      [[maybe_unused]] const auto ntt1_8 = DiFfRG::zsig_mesonsplain::tr7(fenv);
      [[maybe_unused]] const auto ntt1_9 = DiFfRG::zsig_mesonsplain::tr9(fenv);
      [[maybe_unused]] const auto ntt1_10 = DiFfRG::zsig_mesonsplain::tr10(fenv);
      [[maybe_unused]] const auto ntt1_11 = DiFfRG::zsig_mesonsplain::tr11(fenv);
      [[maybe_unused]] const auto ntt1_12 = DiFfRG::zsig_mesonsplain::tr13(fenv);
      [[maybe_unused]] const auto ntt1_13 = DiFfRG::zsig_mesonsplain::tr12(fenv);
      [[maybe_unused]] const auto ntt1_14 = DiFfRG::zsig_mesonsplain::tr14(fenv);
      [[maybe_unused]] const auto ntt1_15 = DiFfRG::zsig_mesonsplain::tr15(fenv);
      [[maybe_unused]] const auto ntt1_16 = DiFfRG::zsig_mesonsplain::tr16(fenv);
      [[maybe_unused]] const auto ntt1_17 = DiFfRG::zsig_mesonsplain::tr17(fenv);
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp12 = RB(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp15 = RB(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp18 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp19 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp22 = RF(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp26 = RF(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den3 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk0, _interp2 + d1V));
      const auto _den4 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk1, _interp2 + d1V));
      const auto _den7 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk3, fma(2., d2V * rhoPhi, _interp2 + d1V)));
      const auto _den8 = powr<-1>(fma(ntglk5, powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0), _interp12 + d1V));
      const auto _den9 = powr<-1>(fma(ntglk4, powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0), _interp15 + d1V));
      const auto _den10 = powr<-3>(fma(powr<-2>(l1) * powr<2>(_interp18 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den11 = powr<-2>(fma(powr<-2>(l1) * powr<2>(_interp18 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den12 = powr<-1>(fma(ntglk7, powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0), fma(2., d2V * rhoPhi, _interp12 + d1V)));
      const auto _den13 = powr<-1>(fma(ntglk6, powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0), fma(2., d2V * rhoPhi, _interp15 + d1V)));
      const auto _den14 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp22 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den15 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp22 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den16 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp26 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _interp1 = ntRe(ntt0_16);
      const auto _interp4 = ntRe(ntt0_17);
      const auto _interp5 = ntRe(ntt0_19);
      const auto _interp6 = ntRe(ntt0_20);
      const auto _interp7 = ntRe(ntt0_21);
      const auto _interp8 = ntRe(ntt0_22);
      const auto _interp9 = ntRe(ntt0_18);
      const auto _interp10 = ntRe(ntt0_23);
      const auto _interp11 = ntRe(ntt0_25);
      const auto _interp13 = ntRe(ntt0_27);
      const auto _interp14 = ntRe(ntt0_24);
      const auto _interp16 = ntRe(ntt0_26);
      const auto _interp17 = ntRe(ntt0_0);
      const auto _interp20 = ntRe(ntt0_3);
      const auto _interp21 = ntRe(ntt0_5);
      const auto _interp24 = ntRe(ntt0_9);
      const auto _interp25 = ntRe(ntt0_11);
      const auto _interp23 = ntRe(ntt0_7);
      const auto _interp27 = ntRe(ntt0_13);
      const auto _interp28 = ntRe(ntt1_11);
      const auto _interp29 = ntRe(ntt1_12);
      const auto _interp30 = ntRe(ntt1_13);
      const auto _interp31 = ntRe(ntt1_14);
      const auto _interp32 = ntRe(ntt1_10);
      const auto _interp33 = ntRe(ntt1_16);
      const auto _interp34 = ntRe(ntt1_15);
      const auto _interp35 = ntRe(ntt1_17);
      const auto _interp36 = ntRe(ntt1_0);
      const auto _interp37 = ntRe(ntt1_2);
      const auto _interp38 = ntRe(ntt1_3);
      const auto _interp39 = ntRe(ntt1_4);
      const auto _interp40 = ntRe(ntt1_5);
      const auto _interp41 = ntRe(ntt1_7);
      const auto _interp42 = ntRe(ntt1_8);
      // clang-format off
      using _T = decltype(_den1 + _den10 + _den11 + _den12 + _den13 + _den14 + _den15 + _den16 + _den3 + _den4 + _den7 + _den8 + _den9 + _interp1 + _interp10 + _interp11 + _interp12 + _interp13 + _interp14 + _interp15 + _interp16 + _interp17 + _interp18 + _interp19 + _interp2 + _interp20 + _interp21 + _interp22 + _interp23 + _interp24 + _interp25 + _interp26 + _interp27 + _interp28 + _interp29 + _interp3 + _interp30 + _interp31 + _interp32 + _interp33 + _interp34 + _interp35 + _interp36 + _interp37 + _interp38 + _interp39 + _interp4 + _interp40 + _interp41 + _interp42 + _interp5 + _interp6 + _interp7 + _interp8 + _interp9 + d1V + d2V + d3V + etaPi0 + f0 + hPhi + l1 + ntglk0 + ntglk2 + rhoPhi);
      // clang-format on
      _T _acc0{};
      _T _acc1{};
      { // subkernel 1
        const auto _den6 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk2 + 2. * d2V * rhoPhi);
        const auto _cse1_k1 = powr<-1>(l1);
        const auto _cse2_k1 = -_interp3;
        const auto _cse3_k1 = _interp2 * etaPi0;
        const auto _cse4_k1 = _cse2_k1 + _cse3_k1;
        const auto _cse5_k1 = -_interp2 * etaPi0;
        const auto _cse6_k1 = _cse5_k1 + _interp3;
        // clang-format off
        _acc0 += 0.01388888888888889 * fma(3., _cse4_k1 * _den1 * _den3 * _den4 * _interp1 * d2V, fma(-6., _cse4_k1 * _den1 * _den3 * _den4 * _interp4 * d2V, fma(3., _cse4_k1 * _den1 * _den6 * _den7 * _interp5 * d2V, fma(-6., _cse4_k1 * _den1 * _den6 * _den7 * _interp6 * d2V, fma(24., _cse1_k1 * _den1 * _den10 * _interp17 * _interp19 * powr<2>(hPhi), fma(-12., _cse1_k1 * _den1 * _den11 * _den15 * _interp19 * _interp21 * powr<2>(hPhi), fma(-12., _cse1_k1 * _den1 * _den11 * _den14 * _interp19 * _interp24 * powr<2>(hPhi), fma(-12., _cse1_k1 * _den1 * _den11 * _den16 * _interp19 * _interp25 * powr<2>(hPhi), fma(-12., _cse6_k1 * _den1 * _den3 * _den6 * _den7 * _interp10 * powr<2>(d2V) * rhoPhi, fma(6., _cse6_k1 * _den1 * _den6 * _den7 * _den8 * _interp11 * powr<2>(d2V) * rhoPhi, fma(6., _cse6_k1 * _den1 * _den12 * _den3 * _den4 * _interp13 * powr<2>(d2V) * rhoPhi, fma(6., _cse6_k1 * _den1 * _den6 * _den7 * _den9 * _interp14 * powr<2>(d2V) * rhoPhi, fma(6., _cse6_k1 * _den1 * _den13 * _den3 * _den4 * _interp16 * powr<2>(d2V) * rhoPhi, fma(-12., _cse6_k1 * _den1 * _den3 * _den4 * _den6 * _interp9 * powr<2>(d2V) * rhoPhi, fma(4., _cse4_k1 * _den1 * _den6 * _den7 * _interp7 * d3V * rhoPhi, fma(-8., _cse4_k1 * _den1 * _den6 * _den7 * _interp8 * d3V * rhoPhi, fma(-12., _cse1_k1 * _den1 * _den10 * _interp19 * _interp20 * powr<4>(hPhi) * rhoPhi, 0.)))))))))))))))));
        // clang-format on
      }
      { // subkernel 2
        const auto _den2 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
        const auto _den5 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk2 + 2. * d2V * rhoPhi);
        const auto _cse1_k2 = powr<-1>(l1);
        const auto _cse2_k2 = -_interp3;
        const auto _cse3_k2 = _interp2 * etaPi0;
        const auto _cse4_k2 = _cse2_k2 + _cse3_k2;
        const auto _cse5_k2 = -_interp2 * etaPi0;
        const auto _cse6_k2 = _cse5_k2 + _interp3;
        const auto _cse7_k2 = powr<2>(f0) + powr<2>(l1);
        const auto _cse8_k2 = _cse7_k2 * ntglk2;
        const auto _cse9_k2 = 2. * d2V * rhoPhi;
        const auto _cse10_k2 = _cse8_k2 + _cse9_k2 + _interp2 + d1V;
        _acc0 += 0.0833333333333333 * fma(_cse1_k2, _den1 * _den11 * _den15 * _interp19 * _interp23 * powr<4>(hPhi) * rhoPhi, fma(_cse1_k2, _den1 * _den11 * _den16 * _interp19 * _interp27 * powr<4>(hPhi) * rhoPhi, 0.));
        // clang-format off
        _acc1 += 0.04166666666666666 * fma(3., _cse4_k2 * _den1 * _den3 * _den4 * _interp28 * d2V, fma(-6., _cse4_k2 * _den1 * _den3 * _den4 * _interp29 * d2V, fma(-12., _cse1_k2 * _den1 * _den10 * _interp19 * _interp36 * powr<2>(hPhi), fma(6., _cse1_k2 * _den1 * _den11 * _den15 * _interp19 * _interp38 * powr<2>(hPhi), fma(6., _cse1_k2 * _den1 * _den11 * _den14 * _interp19 * _interp40 * powr<2>(hPhi), fma(6., _cse1_k2 * _den1 * _den11 * _den16 * _interp19 * _interp41 * powr<2>(hPhi), fma(-12., _cse6_k2 * _den1 * _den2 * _den4 * _interp32 * powr<2>(d2V) * rhoPhi, fma(6., _cse6_k2 * _den1 * _den3 * _den4 * _den8 * _interp33 * powr<2>(d2V) * rhoPhi, fma(6., _cse6_k2 * _den1 * _den3 * _den4 * _den9 * _interp34 * powr<2>(d2V) * rhoPhi, fma(-432., _cse6_k2 * _den1 * _den5 * _den7 * _interp35 * powr<2>(d2V) * rhoPhi, fma(216., _cse10_k2 * _cse6_k2 * _den1 * _den12 * _den5 * _den7 * _interp35 * powr<2>(d2V) * rhoPhi, fma(216., _cse10_k2 * _cse6_k2 * _den1 * _den13 * _den5 * _den7 * _interp35 * powr<2>(d2V) * rhoPhi, fma(4., _cse4_k2 * _den1 * _den3 * _den4 * _interp30 * d3V * rhoPhi, fma(-8., _cse4_k2 * _den1 * _den3 * _den4 * _interp31 * d3V * rhoPhi, fma(6., _cse1_k2 * _den1 * _den10 * _interp19 * _interp37 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse1_k2 * _den1 * _den11 * _den15 * _interp19 * _interp39 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse1_k2 * _den1 * _den11 * _den16 * _interp19 * _interp42 * powr<4>(hPhi) * rhoPhi, fma(-576., _cse6_k2 * _den1 * _den5 * _den7 * _interp35 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10_k2 * _cse6_k2 * _den1 * _den12 * _den5 * _den7 * _interp35 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10_k2 * _cse6_k2 * _den1 * _den13 * _den5 * _den7 * _interp35 * d2V * d3V * powr<2>(rhoPhi), fma(-192., _cse6_k2 * _den1 * _den5 * _den7 * _interp35 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10_k2 * _cse6_k2 * _den1 * _den12 * _den5 * _den7 * _interp35 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10_k2 * _cse6_k2 * _den1 * _den13 * _den5 * _den7 * _interp35 * powr<2>(d3V) * powr<3>(rhoPhi), 0.)))))))))))))))))))))));
        // clang-format on
      }
      return DiFfRG::MultiValue<double, 2>(_acc0, _acc1);
    }
    #elif (NT_ZPI_MESONSPLAIN_VERDICT != 0) && (NT_ZSIG_MESONSPLAIN_VERDICT != 0)   // RePart, exact for every real member
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto ntgix0 = ZPi.index(-f0, l1);
      [[maybe_unused]] const auto ntgix1 = ZPi.index(f0, l1);
      [[maybe_unused]] const auto ntgix2 = ZPi.index(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntgix3 = ZPi.index(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntglk0 = ZPi.at(ntgix0);
      [[maybe_unused]] const auto ntglk1 = ZPi.at(ntgix1);
      [[maybe_unused]] const auto ntglk2 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk3 = ZSig.at(ntgix1);
      [[maybe_unused]] const auto ntglk4 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk5 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      constexpr auto ntnenv0 = 1;
      constexpr auto ntnenv1 = ntnenv0 > DiFfRG::zpi_mesonsplain::nenv ? ntnenv0 : DiFfRG::zpi_mesonsplain::nenv;
      constexpr auto ntnenv2 = ntnenv1 > DiFfRG::zsig_mesonsplain::nenv ? ntnenv1 : DiFfRG::zsig_mesonsplain::nenv;
      double fenv[ntnenv2];
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
      [[maybe_unused]] const auto ntt0_0 = DiFfRG::zpi_mesonsplain::tr3(fenv);
      [[maybe_unused]] const auto ntt0_1 = DiFfRG::zpi_mesonsplain::tr4(fenv);
      [[maybe_unused]] const auto ntt0_2 = DiFfRG::zpi_mesonsplain::tr2(fenv);
      [[maybe_unused]] const auto ntt0_3 = DiFfRG::zpi_mesonsplain::tr1(fenv);
      [[maybe_unused]] const auto ntt0_4 = DiFfRG::zpi_mesonsplain::tr0(fenv);
      [[maybe_unused]] const auto ntt0_5 = DiFfRG::zpi_mesonsplain::tr8(fenv);
      [[maybe_unused]] const auto ntt0_6 = DiFfRG::zpi_mesonsplain::tr7(fenv);
      [[maybe_unused]] const auto ntt0_7 = DiFfRG::zpi_mesonsplain::tr6(fenv);
      [[maybe_unused]] const auto ntt0_8 = DiFfRG::zpi_mesonsplain::tr5(fenv);
      [[maybe_unused]] const auto ntt0_9 = DiFfRG::zpi_mesonsplain::tr10(fenv);
      [[maybe_unused]] const auto ntt0_10 = DiFfRG::zpi_mesonsplain::tr9(fenv);
      [[maybe_unused]] const auto ntt0_11 = DiFfRG::zpi_mesonsplain::tr14(fenv);
      [[maybe_unused]] const auto ntt0_12 = DiFfRG::zpi_mesonsplain::tr13(fenv);
      [[maybe_unused]] const auto ntt0_13 = DiFfRG::zpi_mesonsplain::tr12(fenv);
      [[maybe_unused]] const auto ntt0_14 = DiFfRG::zpi_mesonsplain::tr11(fenv);
      [[maybe_unused]] const auto ntt0_15 = DiFfRG::zpi_mesonsplain::tr15(fenv);
      [[maybe_unused]] const auto ntt0_16 = DiFfRG::zpi_mesonsplain::tr16(fenv);
      [[maybe_unused]] const auto ntt0_17 = DiFfRG::zpi_mesonsplain::tr17(fenv);
      [[maybe_unused]] const auto ntt0_18 = DiFfRG::zpi_mesonsplain::tr18(fenv);
      [[maybe_unused]] const auto ntt0_19 = DiFfRG::zpi_mesonsplain::tr19(fenv);
      [[maybe_unused]] const auto ntt0_20 = DiFfRG::zpi_mesonsplain::tr21(fenv);
      [[maybe_unused]] const auto ntt0_21 = DiFfRG::zpi_mesonsplain::tr20(fenv);
      [[maybe_unused]] const auto ntt0_22 = DiFfRG::zpi_mesonsplain::tr22(fenv);
      [[maybe_unused]] const auto ntt0_23 = DiFfRG::zpi_mesonsplain::tr23(fenv);
      [[maybe_unused]] const auto ntt0_24 = DiFfRG::zpi_mesonsplain::tr24(fenv);
      [[maybe_unused]] const auto ntt0_25 = DiFfRG::zpi_mesonsplain::tr25(fenv);
      [[maybe_unused]] const auto ntt0_26 = DiFfRG::zpi_mesonsplain::tr26(fenv);
      [[maybe_unused]] const auto ntt0_27 = DiFfRG::zpi_mesonsplain::tr27(fenv);
      DiFfRG::zsig_mesonsplain::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      [[maybe_unused]] const auto ntt1_0 = DiFfRG::zsig_mesonsplain::tr1(fenv);
      [[maybe_unused]] const auto ntt1_1 = DiFfRG::zsig_mesonsplain::tr2(fenv);
      [[maybe_unused]] const auto ntt1_2 = DiFfRG::zsig_mesonsplain::tr0(fenv);
      [[maybe_unused]] const auto ntt1_3 = DiFfRG::zsig_mesonsplain::tr4(fenv);
      [[maybe_unused]] const auto ntt1_4 = DiFfRG::zsig_mesonsplain::tr3(fenv);
      [[maybe_unused]] const auto ntt1_5 = DiFfRG::zsig_mesonsplain::tr6(fenv);
      [[maybe_unused]] const auto ntt1_6 = DiFfRG::zsig_mesonsplain::tr5(fenv);
      [[maybe_unused]] const auto ntt1_7 = DiFfRG::zsig_mesonsplain::tr8(fenv);
      [[maybe_unused]] const auto ntt1_8 = DiFfRG::zsig_mesonsplain::tr7(fenv);
      [[maybe_unused]] const auto ntt1_9 = DiFfRG::zsig_mesonsplain::tr9(fenv);
      [[maybe_unused]] const auto ntt1_10 = DiFfRG::zsig_mesonsplain::tr10(fenv);
      [[maybe_unused]] const auto ntt1_11 = DiFfRG::zsig_mesonsplain::tr11(fenv);
      [[maybe_unused]] const auto ntt1_12 = DiFfRG::zsig_mesonsplain::tr13(fenv);
      [[maybe_unused]] const auto ntt1_13 = DiFfRG::zsig_mesonsplain::tr12(fenv);
      [[maybe_unused]] const auto ntt1_14 = DiFfRG::zsig_mesonsplain::tr14(fenv);
      [[maybe_unused]] const auto ntt1_15 = DiFfRG::zsig_mesonsplain::tr15(fenv);
      [[maybe_unused]] const auto ntt1_16 = DiFfRG::zsig_mesonsplain::tr16(fenv);
      [[maybe_unused]] const auto ntt1_17 = DiFfRG::zsig_mesonsplain::tr17(fenv);
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp12 = RB(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp15 = RB(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp18 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp19 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp25 = RF(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp32 = RF(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den3 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk0, _interp2 + d1V));
      const auto _den4 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk1, _interp2 + d1V));
      const auto _den7 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk3, fma(2., d2V * rhoPhi, _interp2 + d1V)));
      const auto _den8 = powr<-1>(fma(ntglk5, powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0), _interp12 + d1V));
      const auto _den9 = powr<-1>(fma(ntglk4, powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0), _interp15 + d1V));
      const auto _den10 = powr<-3>(fma(powr<-2>(l1) * powr<2>(_interp18 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den11 = powr<-2>(fma(powr<-2>(l1) * powr<2>(_interp18 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den12 = powr<-1>(fma(ntglk7, powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0), fma(2., d2V * rhoPhi, _interp12 + d1V)));
      const auto _den13 = powr<-1>(fma(ntglk6, powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0), fma(2., d2V * rhoPhi, _interp15 + d1V)));
      const auto _den14 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp32 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _den15 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp25 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den16 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp25 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den17 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp32 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _interp1 = ntRe(ntt0_16);
      const auto _interp4 = ntRe(ntt0_17);
      const auto _interp5 = ntRe(ntt0_19);
      const auto _interp6 = ntRe(ntt0_20);
      const auto _interp7 = ntRe(ntt0_21);
      const auto _interp8 = ntRe(ntt0_22);
      const auto _interp9 = ntRe(ntt0_18);
      const auto _interp10 = ntRe(ntt0_23);
      const auto _interp11 = ntRe(ntt0_25);
      const auto _interp13 = ntRe(ntt0_27);
      const auto _interp14 = ntRe(ntt0_24);
      const auto _interp16 = ntRe(ntt0_26);
      const auto _interp17 = ntIm(ntt0_1);
      const auto _interp20 = ntIm(ntt0_2);
      const auto _interp22 = ntRe(ntt0_0);
      const auto _interp23 = ntRe(ntt0_3);
      const auto _interp24 = ntIm(ntt0_6);
      const auto _interp27 = ntRe(ntt0_5);
      const auto _interp29 = ntIm(ntt0_10);
      const auto _interp30 = ntRe(ntt0_9);
      const auto _interp31 = ntIm(ntt0_12);
      const auto _interp34 = ntRe(ntt0_11);
      const auto _interp36 = ntIm(ntt0_15);
      const auto _cse5_k1 = sqrt(rhoPhi);
      const auto _interp21 = ntIm(ntt0_4);
      const auto _interp26 = ntIm(ntt0_8);
      const auto _interp28 = ntRe(ntt0_7);
      const auto _interp33 = ntIm(ntt0_14);
      const auto _interp35 = ntRe(ntt0_13);
      const auto _interp37 = ntRe(ntt1_11);
      const auto _interp38 = ntRe(ntt1_12);
      const auto _interp39 = ntRe(ntt1_13);
      const auto _interp40 = ntRe(ntt1_14);
      const auto _interp41 = ntRe(ntt1_10);
      const auto _interp42 = ntRe(ntt1_16);
      const auto _interp43 = ntRe(ntt1_15);
      const auto _interp44 = ntRe(ntt1_17);
      const auto _interp45 = ntIm(ntt1_1);
      const auto _interp46 = ntRe(ntt1_0);
      const auto _interp47 = ntRe(ntt1_2);
      const auto _interp48 = ntRe(ntt1_3);
      const auto _interp49 = ntRe(ntt1_4);
      const auto _interp50 = ntIm(ntt1_6);
      const auto _interp51 = ntRe(ntt1_5);
      const auto _interp52 = ntRe(ntt1_7);
      const auto _interp53 = ntRe(ntt1_8);
      const auto _interp54 = ntIm(ntt1_9);
      const auto _cse2_k2 = sqrt(powr<3>(rhoPhi));
      // clang-format off
      using _T = decltype(_cse2_k2 + _cse5_k1 + _den1 + _den10 + _den11 + _den12 + _den13 + _den14 + _den15 + _den16 + _den17 + _den3 + _den4 + _den7 + _den8 + _den9 + _interp1 + _interp10 + _interp11 + _interp12 + _interp13 + _interp14 + _interp15 + _interp16 + _interp17 + _interp18 + _interp19 + _interp2 + _interp20 + _interp21 + _interp22 + _interp23 + _interp24 + _interp25 + _interp26 + _interp27 + _interp28 + _interp29 + _interp3 + _interp30 + _interp31 + _interp32 + _interp33 + _interp34 + _interp35 + _interp36 + _interp37 + _interp38 + _interp39 + _interp4 + _interp40 + _interp41 + _interp42 + _interp43 + _interp44 + _interp45 + _interp46 + _interp47 + _interp48 + _interp49 + _interp5 + _interp50 + _interp51 + _interp52 + _interp53 + _interp54 + _interp6 + _interp7 + _interp8 + _interp9 + d1V + d2V + d3V + etaPi0 + f0 + hPhi + l1 + ntglk0 + ntglk2 + rhoPhi);
      // clang-format on
      _T _acc0{};
      _T _acc1{};
      { // subkernel 1
        const auto _den6 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk2 + 2. * d2V * rhoPhi);
        const auto _cse1_k1 = powr<-1>(l1);
        const auto _cse2_k1 = -_interp3;
        const auto _cse3_k1 = _interp2 * etaPi0;
        const auto _cse4_k1 = _cse2_k1 + _cse3_k1;
        const auto _cse6_k1 = -_interp2 * etaPi0;
        const auto _cse7_k1 = _cse6_k1 + _interp3;
        // clang-format off
        _acc0 += 0.01388888888888889 * fma(3., _cse4_k1 * _den1 * _den3 * _den4 * _interp1 * d2V, fma(-6., _cse4_k1 * _den1 * _den3 * _den4 * _interp4 * d2V, fma(3., _cse4_k1 * _den1 * _den6 * _den7 * _interp5 * d2V, fma(-6., _cse4_k1 * _den1 * _den6 * _den7 * _interp6 * d2V, fma(-24., _cse1_k1 * _den1 * _den10 * _interp17 * _interp19 * powr<2>(hPhi), fma(24., _cse1_k1 * _den1 * _den10 * _interp19 * _interp22 * powr<2>(hPhi), fma(-12., _cse1_k1 * _den1 * _den11 * _den16 * _interp19 * _interp27 * powr<2>(hPhi), fma(-12., _cse1_k1 * _den1 * _den11 * _den15 * _interp19 * _interp30 * powr<2>(hPhi), fma(-12., _cse1_k1 * _den1 * _den11 * _den17 * _interp19 * _interp34 * powr<2>(hPhi), fma(12., _cse1_k1 * _den1 * _den11 * _den14 * _interp19 * _interp36 * powr<2>(hPhi), fma(12., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den10 * _interp19 * _interp20 * powr<3>(hPhi), fma(-6., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den11 * _den16 * _interp19 * _interp24 * powr<3>(hPhi), fma(-6., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den11 * _den15 * _interp19 * _interp29 * powr<3>(hPhi), fma(-6., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den11 * _den17 * _interp19 * _interp31 * powr<3>(hPhi), fma(-12., _cse7_k1 * _den1 * _den3 * _den6 * _den7 * _interp10 * powr<2>(d2V) * rhoPhi, fma(6., _cse7_k1 * _den1 * _den6 * _den7 * _den8 * _interp11 * powr<2>(d2V) * rhoPhi, fma(6., _cse7_k1 * _den1 * _den12 * _den3 * _den4 * _interp13 * powr<2>(d2V) * rhoPhi, fma(6., _cse7_k1 * _den1 * _den6 * _den7 * _den9 * _interp14 * powr<2>(d2V) * rhoPhi, fma(6., _cse7_k1 * _den1 * _den13 * _den3 * _den4 * _interp16 * powr<2>(d2V) * rhoPhi, fma(-12., _cse7_k1 * _den1 * _den3 * _den4 * _den6 * _interp9 * powr<2>(d2V) * rhoPhi, fma(4., _cse4_k1 * _den1 * _den6 * _den7 * _interp7 * d3V * rhoPhi, fma(-8., _cse4_k1 * _den1 * _den6 * _den7 * _interp8 * d3V * rhoPhi, fma(-12., _cse1_k1 * _den1 * _den10 * _interp19 * _interp23 * powr<4>(hPhi) * rhoPhi, 0.)))))))))))))))))))))));
        // clang-format on
      }
      { // subkernel 2
        const auto _den2 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
        const auto _den5 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk2 + 2. * d2V * rhoPhi);
        const auto _cse1_k2 = powr<-1>(l1);
        const auto _cse3_k2 = -_interp3;
        const auto _cse4_k2 = _interp2 * etaPi0;
        const auto _cse5_k2 = _cse3_k2 + _cse4_k2;
        const auto _cse6_k2 = -_interp2 * etaPi0;
        const auto _cse7_k2 = _cse6_k2 + _interp3;
        const auto _cse8_k2 = powr<2>(f0) + powr<2>(l1);
        const auto _cse9_k2 = _cse8_k2 * ntglk2;
        const auto _cse10_k2 = 2. * d2V * rhoPhi;
        const auto _cse11_k2 = _cse10_k2 + _cse9_k2 + _interp2 + d1V;
        _acc0 += 0.04166666666666666 * fma(-2., 1.414213562373095 * _cse1_k2 * _cse2_k2 * _den1 * _den10 * _interp19 * _interp21 * powr<5>(hPhi), fma(1.414213562373095, _cse1_k2 * _cse2_k2 * _den1 * _den11 * _den16 * _interp19 * _interp26 * powr<5>(hPhi), fma(1.414213562373095, _cse1_k2 * _cse2_k2 * _den1 * _den11 * _den17 * _interp19 * _interp33 * powr<5>(hPhi), fma(2., _cse1_k2 * _den1 * _den11 * _den16 * _interp19 * _interp28 * powr<4>(hPhi) * rhoPhi, fma(2., _cse1_k2 * _den1 * _den11 * _den17 * _interp19 * _interp35 * powr<4>(hPhi) * rhoPhi, 0.)))));
        // clang-format off
        _acc1 += 0.04166666666666666 * fma(3., _cse5_k2 * _den1 * _den3 * _den4 * _interp37 * d2V, fma(-6., _cse5_k2 * _den1 * _den3 * _den4 * _interp38 * d2V, fma(12., _cse1_k2 * _den1 * _den10 * _interp19 * _interp45 * powr<2>(hPhi), fma(-12., _cse1_k2 * _den1 * _den10 * _interp19 * _interp46 * powr<2>(hPhi), fma(6., _cse1_k2 * _den1 * _den11 * _den16 * _interp19 * _interp48 * powr<2>(hPhi), fma(6., _cse1_k2 * _den1 * _den11 * _den15 * _interp19 * _interp51 * powr<2>(hPhi), fma(6., _cse1_k2 * _den1 * _den11 * _den17 * _interp19 * _interp52 * powr<2>(hPhi), fma(-6., _cse1_k2 * _den1 * _den11 * _den14 * _interp19 * _interp54 * powr<2>(hPhi), fma(3., 1.414213562373095 * _cse1_k2 * _den1 * _den11 * _den15 * _interp19 * _interp50 * powr<3>(hPhi) * sqrt(rhoPhi), fma(-12., _cse7_k2 * _den1 * _den2 * _den4 * _interp41 * powr<2>(d2V) * rhoPhi, fma(6., _cse7_k2 * _den1 * _den3 * _den4 * _den8 * _interp42 * powr<2>(d2V) * rhoPhi, fma(6., _cse7_k2 * _den1 * _den3 * _den4 * _den9 * _interp43 * powr<2>(d2V) * rhoPhi, fma(-432., _cse7_k2 * _den1 * _den5 * _den7 * _interp44 * powr<2>(d2V) * rhoPhi, fma(216., _cse11_k2 * _cse7_k2 * _den1 * _den12 * _den5 * _den7 * _interp44 * powr<2>(d2V) * rhoPhi, fma(216., _cse11_k2 * _cse7_k2 * _den1 * _den13 * _den5 * _den7 * _interp44 * powr<2>(d2V) * rhoPhi, fma(4., _cse5_k2 * _den1 * _den3 * _den4 * _interp39 * d3V * rhoPhi, fma(-8., _cse5_k2 * _den1 * _den3 * _den4 * _interp40 * d3V * rhoPhi, fma(6., _cse1_k2 * _den1 * _den10 * _interp19 * _interp47 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse1_k2 * _den1 * _den11 * _den16 * _interp19 * _interp49 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse1_k2 * _den1 * _den11 * _den17 * _interp19 * _interp53 * powr<4>(hPhi) * rhoPhi, fma(-576., _cse7_k2 * _den1 * _den5 * _den7 * _interp44 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse11_k2 * _cse7_k2 * _den1 * _den12 * _den5 * _den7 * _interp44 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse11_k2 * _cse7_k2 * _den1 * _den13 * _den5 * _den7 * _interp44 * d2V * d3V * powr<2>(rhoPhi), fma(-192., _cse7_k2 * _den1 * _den5 * _den7 * _interp44 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse11_k2 * _cse7_k2 * _den1 * _den12 * _den5 * _den7 * _interp44 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse11_k2 * _cse7_k2 * _den1 * _den13 * _den5 * _den7 * _interp44 * powr<2>(d3V) * powr<3>(rhoPhi), 0.))))))))))))))))))))))))));
        // clang-format on
      }
      return DiFfRG::MultiValue<double, 2>(_acc0, _acc1);
    }
    #elif (NT_ZPI_MESONSPLAIN_VERDICT == 0) && (NT_ZSIG_MESONSPLAIN_VERDICT == 0)   // genuinely complex
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto ntgix0 = ZPi.index(-f0, l1);
      [[maybe_unused]] const auto ntgix1 = ZPi.index(f0, l1);
      [[maybe_unused]] const auto ntgix2 = ZPi.index(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntgix3 = ZPi.index(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntglk0 = ZPi.at(ntgix0);
      [[maybe_unused]] const auto ntglk1 = ZPi.at(ntgix1);
      [[maybe_unused]] const auto ntglk2 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk3 = ZSig.at(ntgix1);
      [[maybe_unused]] const auto ntglk4 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk5 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      constexpr auto ntnenv0 = 1;
      constexpr auto ntnenv1 = ntnenv0 > DiFfRG::zpi_mesonsplain::nenv ? ntnenv0 : DiFfRG::zpi_mesonsplain::nenv;
      constexpr auto ntnenv2 = ntnenv1 > DiFfRG::zsig_mesonsplain::nenv ? ntnenv1 : DiFfRG::zsig_mesonsplain::nenv;
      double fenv[ntnenv2];
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
      [[maybe_unused]] const auto ntt0_0 = DiFfRG::zpi_mesonsplain::tr3(fenv);
      [[maybe_unused]] const auto ntt0_1 = DiFfRG::zpi_mesonsplain::tr4(fenv);
      [[maybe_unused]] const auto ntt0_2 = DiFfRG::zpi_mesonsplain::tr2(fenv);
      [[maybe_unused]] const auto ntt0_3 = DiFfRG::zpi_mesonsplain::tr1(fenv);
      [[maybe_unused]] const auto ntt0_4 = DiFfRG::zpi_mesonsplain::tr0(fenv);
      [[maybe_unused]] const auto ntt0_5 = DiFfRG::zpi_mesonsplain::tr8(fenv);
      [[maybe_unused]] const auto ntt0_6 = DiFfRG::zpi_mesonsplain::tr7(fenv);
      [[maybe_unused]] const auto ntt0_7 = DiFfRG::zpi_mesonsplain::tr6(fenv);
      [[maybe_unused]] const auto ntt0_8 = DiFfRG::zpi_mesonsplain::tr5(fenv);
      [[maybe_unused]] const auto ntt0_9 = DiFfRG::zpi_mesonsplain::tr10(fenv);
      [[maybe_unused]] const auto ntt0_10 = DiFfRG::zpi_mesonsplain::tr9(fenv);
      [[maybe_unused]] const auto ntt0_11 = DiFfRG::zpi_mesonsplain::tr14(fenv);
      [[maybe_unused]] const auto ntt0_12 = DiFfRG::zpi_mesonsplain::tr13(fenv);
      [[maybe_unused]] const auto ntt0_13 = DiFfRG::zpi_mesonsplain::tr12(fenv);
      [[maybe_unused]] const auto ntt0_14 = DiFfRG::zpi_mesonsplain::tr11(fenv);
      [[maybe_unused]] const auto ntt0_15 = DiFfRG::zpi_mesonsplain::tr15(fenv);
      [[maybe_unused]] const auto ntt0_16 = DiFfRG::zpi_mesonsplain::tr16(fenv);
      [[maybe_unused]] const auto ntt0_17 = DiFfRG::zpi_mesonsplain::tr17(fenv);
      [[maybe_unused]] const auto ntt0_18 = DiFfRG::zpi_mesonsplain::tr18(fenv);
      [[maybe_unused]] const auto ntt0_19 = DiFfRG::zpi_mesonsplain::tr19(fenv);
      [[maybe_unused]] const auto ntt0_20 = DiFfRG::zpi_mesonsplain::tr21(fenv);
      [[maybe_unused]] const auto ntt0_21 = DiFfRG::zpi_mesonsplain::tr20(fenv);
      [[maybe_unused]] const auto ntt0_22 = DiFfRG::zpi_mesonsplain::tr22(fenv);
      [[maybe_unused]] const auto ntt0_23 = DiFfRG::zpi_mesonsplain::tr23(fenv);
      [[maybe_unused]] const auto ntt0_24 = DiFfRG::zpi_mesonsplain::tr24(fenv);
      [[maybe_unused]] const auto ntt0_25 = DiFfRG::zpi_mesonsplain::tr25(fenv);
      [[maybe_unused]] const auto ntt0_26 = DiFfRG::zpi_mesonsplain::tr26(fenv);
      [[maybe_unused]] const auto ntt0_27 = DiFfRG::zpi_mesonsplain::tr27(fenv);
      DiFfRG::zsig_mesonsplain::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      [[maybe_unused]] const auto ntt1_0 = DiFfRG::zsig_mesonsplain::tr1(fenv);
      [[maybe_unused]] const auto ntt1_1 = DiFfRG::zsig_mesonsplain::tr2(fenv);
      [[maybe_unused]] const auto ntt1_2 = DiFfRG::zsig_mesonsplain::tr0(fenv);
      [[maybe_unused]] const auto ntt1_3 = DiFfRG::zsig_mesonsplain::tr4(fenv);
      [[maybe_unused]] const auto ntt1_4 = DiFfRG::zsig_mesonsplain::tr3(fenv);
      [[maybe_unused]] const auto ntt1_5 = DiFfRG::zsig_mesonsplain::tr6(fenv);
      [[maybe_unused]] const auto ntt1_6 = DiFfRG::zsig_mesonsplain::tr5(fenv);
      [[maybe_unused]] const auto ntt1_7 = DiFfRG::zsig_mesonsplain::tr8(fenv);
      [[maybe_unused]] const auto ntt1_8 = DiFfRG::zsig_mesonsplain::tr7(fenv);
      [[maybe_unused]] const auto ntt1_9 = DiFfRG::zsig_mesonsplain::tr9(fenv);
      [[maybe_unused]] const auto ntt1_10 = DiFfRG::zsig_mesonsplain::tr10(fenv);
      [[maybe_unused]] const auto ntt1_11 = DiFfRG::zsig_mesonsplain::tr11(fenv);
      [[maybe_unused]] const auto ntt1_12 = DiFfRG::zsig_mesonsplain::tr13(fenv);
      [[maybe_unused]] const auto ntt1_13 = DiFfRG::zsig_mesonsplain::tr12(fenv);
      [[maybe_unused]] const auto ntt1_14 = DiFfRG::zsig_mesonsplain::tr14(fenv);
      [[maybe_unused]] const auto ntt1_15 = DiFfRG::zsig_mesonsplain::tr15(fenv);
      [[maybe_unused]] const auto ntt1_16 = DiFfRG::zsig_mesonsplain::tr16(fenv);
      [[maybe_unused]] const auto ntt1_17 = DiFfRG::zsig_mesonsplain::tr17(fenv);
      const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RB(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp4 = RB(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp5 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp6 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp7 = RF(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp8 = RF(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den3 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk0, _interp1 + d1V));
      const auto _den4 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk1, _interp1 + d1V));
      const auto _den7 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk3, fma(2., d2V * rhoPhi, _interp1 + d1V)));
      const auto _den8 = powr<-1>(fma(ntglk5, powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0), _interp3 + d1V));
      const auto _den9 = powr<-1>(fma(ntglk4, powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0), _interp4 + d1V));
      const auto _den10 = powr<-3>(fma(powr<-2>(l1) * powr<2>(_interp5 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den11 = powr<-2>(fma(powr<-2>(l1) * powr<2>(_interp5 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den12 = powr<-1>(fma(ntglk7, powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0), fma(2., d2V * rhoPhi, _interp3 + d1V)));
      const auto _den13 = powr<-1>(fma(ntglk6, powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0), fma(2., d2V * rhoPhi, _interp4 + d1V)));
      const auto _den14 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp8 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _den15 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp7 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den16 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp7 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den17 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp8 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _cse5_k1 = sqrt(rhoPhi);
      const auto _cse2_k2 = sqrt(powr<3>(rhoPhi));
      // clang-format off
      using _T = decltype(complex<double>(0.,-0.01388888888888889) + _cse2_k2 + _cse5_k1 + _den1 + _den10 + _den11 + _den12 + _den13 + _den14 + _den15 + _den16 + _den17 + _den3 + _den4 + _den7 + _den8 + _den9 + _interp1 + _interp2 + _interp3 + _interp4 + _interp5 + _interp6 + _interp7 + _interp8 + ntt0_0 + ntt0_1 + ntt0_10 + ntt0_11 + ntt0_12 + ntt0_13 + ntt0_14 + ntt0_15 + ntt0_16 + ntt0_17 + ntt0_18 + ntt0_19 + ntt0_2 + ntt0_20 + ntt0_21 + ntt0_22 + ntt0_23 + ntt0_24 + ntt0_25 + ntt0_26 + ntt0_27 + ntt0_3 + ntt0_4 + ntt0_5 + ntt0_6 + ntt0_7 + ntt0_8 + ntt0_9 + ntt1_0 + ntt1_1 + ntt1_10 + ntt1_11 + ntt1_12 + ntt1_13 + ntt1_14 + ntt1_15 + ntt1_16 + ntt1_17 + ntt1_2 + ntt1_3 + ntt1_4 + ntt1_5 + ntt1_6 + ntt1_7 + ntt1_8 + ntt1_9 + d1V + d2V + d3V + etaPi0 + f0 + hPhi + l1 + ntglk0 + ntglk2 + rhoPhi);
      // clang-format on
      _T _acc0{};
      _T _acc1{};
      { // subkernel 1
        const auto _den6 = powr<-1>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk2 + 2. * d2V * rhoPhi);
        const auto _cse1_k1 = powr<-1>(l1);
        const auto _cse2_k1 = -_interp2;
        const auto _cse3_k1 = _interp1 * etaPi0;
        const auto _cse4_k1 = _cse2_k1 + _cse3_k1;
        const auto _cse6_k1 = -_interp1 * etaPi0;
        const auto _cse7_k1 = _cse6_k1 + _interp2;
        // clang-format off
        _acc0 += complex<double>(0.,-0.01388888888888889) * fma(complex<double>(0.,3.), _cse4_k1 * _den1 * _den3 * _den4 * ntt0_16 * d2V, fma(complex<double>(0.,-6.), _cse4_k1 * _den1 * _den3 * _den4 * ntt0_17 * d2V, fma(complex<double>(0.,3.), _cse4_k1 * _den1 * _den6 * _den7 * ntt0_19 * d2V, fma(complex<double>(0.,-6.), _cse4_k1 * _den1 * _den6 * _den7 * ntt0_20 * d2V, fma(complex<double>(0.,24.), _cse1_k1 * _den1 * _den10 * _interp6 * ntt0_0 * powr<2>(hPhi), fma(-24., _cse1_k1 * _den1 * _den10 * _interp6 * ntt0_1 * powr<2>(hPhi), fma(complex<double>(0.,-12.), _cse1_k1 * _den1 * _den11 * _den17 * _interp6 * ntt0_11 * powr<2>(hPhi), fma(12., _cse1_k1 * _den1 * _den11 * _den14 * _interp6 * ntt0_15 * powr<2>(hPhi), fma(complex<double>(0.,-12.), _cse1_k1 * _den1 * _den11 * _den16 * _interp6 * ntt0_5 * powr<2>(hPhi), fma(complex<double>(0.,-12.), _cse1_k1 * _den1 * _den11 * _den15 * _interp6 * ntt0_9 * powr<2>(hPhi), fma(-6., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den11 * _den15 * _interp6 * ntt0_10 * powr<3>(hPhi), fma(-6., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den11 * _den17 * _interp6 * ntt0_12 * powr<3>(hPhi), fma(12., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den10 * _interp6 * ntt0_2 * powr<3>(hPhi), fma(-6., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den11 * _den16 * _interp6 * ntt0_6 * powr<3>(hPhi), fma(complex<double>(0.,-12.), _cse7_k1 * _den1 * _den3 * _den4 * _den6 * ntt0_18 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-12.), _cse7_k1 * _den1 * _den3 * _den6 * _den7 * ntt0_23 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse7_k1 * _den1 * _den6 * _den7 * _den9 * ntt0_24 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse7_k1 * _den1 * _den6 * _den7 * _den8 * ntt0_25 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse7_k1 * _den1 * _den13 * _den3 * _den4 * ntt0_26 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse7_k1 * _den1 * _den12 * _den3 * _den4 * ntt0_27 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,4.), _cse4_k1 * _den1 * _den6 * _den7 * ntt0_21 * d3V * rhoPhi, fma(complex<double>(0.,-8.), _cse4_k1 * _den1 * _den6 * _den7 * ntt0_22 * d3V * rhoPhi, fma(complex<double>(0.,6.), _cse1_k1 * _den1 * _den11 * _den17 * _interp6 * ntt0_13 * powr<4>(hPhi) * rhoPhi, 0.)))))))))))))))))))))));
        // clang-format on
      }
      { // subkernel 2
        const auto _den2 = powr<-2>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
        const auto _den5 = powr<-2>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk2 + 2. * d2V * rhoPhi);
        const auto _cse1_k2 = powr<-1>(l1);
        const auto _cse3_k2 = -_interp2;
        const auto _cse4_k2 = _interp1 * etaPi0;
        const auto _cse5_k2 = _cse3_k2 + _cse4_k2;
        const auto _cse6_k2 = -_interp1 * etaPi0;
        const auto _cse7_k2 = _cse6_k2 + _interp2;
        const auto _cse8_k2 = powr<2>(f0) + powr<2>(l1);
        const auto _cse9_k2 = _cse8_k2 * ntglk2;
        const auto _cse10_k2 = 2. * d2V * rhoPhi;
        const auto _cse11_k2 = _cse10_k2 + _cse9_k2 + _interp1 + d1V;
        _acc0 += complex<double>(0.,-0.04166666666666666) * fma(1.414213562373095, _cse1_k2 * _cse2_k2 * _den1 * _den11 * _den17 * _interp6 * ntt0_14 * powr<5>(hPhi), fma(-2., 1.414213562373095 * _cse1_k2 * _cse2_k2 * _den1 * _den10 * _interp6 * ntt0_4 * powr<5>(hPhi), fma(1.414213562373095, _cse1_k2 * _cse2_k2 * _den1 * _den11 * _den16 * _interp6 * ntt0_8 * powr<5>(hPhi), fma(complex<double>(0.,-4.), _cse1_k2 * _den1 * _den10 * _interp6 * ntt0_3 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,2.), _cse1_k2 * _den1 * _den11 * _den16 * _interp6 * ntt0_7 * powr<4>(hPhi) * rhoPhi, 0.)))));
        // clang-format off
        _acc1 += complex<double>(0.,-0.04166666666666666) * fma(complex<double>(0.,3.), _cse5_k2 * _den1 * _den3 * _den4 * ntt1_11 * d2V, fma(complex<double>(0.,-6.), _cse5_k2 * _den1 * _den3 * _den4 * ntt1_12 * d2V, fma(complex<double>(0.,-12.), _cse1_k2 * _den1 * _den10 * _interp6 * ntt1_0 * powr<2>(hPhi), fma(12., _cse1_k2 * _den1 * _den10 * _interp6 * ntt1_1 * powr<2>(hPhi), fma(complex<double>(0.,6.), _cse1_k2 * _den1 * _den11 * _den16 * _interp6 * ntt1_3 * powr<2>(hPhi), fma(complex<double>(0.,6.), _cse1_k2 * _den1 * _den11 * _den15 * _interp6 * ntt1_5 * powr<2>(hPhi), fma(complex<double>(0.,6.), _cse1_k2 * _den1 * _den11 * _den17 * _interp6 * ntt1_7 * powr<2>(hPhi), fma(-6., _cse1_k2 * _den1 * _den11 * _den14 * _interp6 * ntt1_9 * powr<2>(hPhi), fma(3., 1.414213562373095 * _cse1_k2 * _den1 * _den11 * _den15 * _interp6 * ntt1_6 * powr<3>(hPhi) * sqrt(rhoPhi), fma(complex<double>(0.,-12.), _cse7_k2 * _den1 * _den2 * _den4 * ntt1_10 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse7_k2 * _den1 * _den3 * _den4 * _den9 * ntt1_15 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse7_k2 * _den1 * _den3 * _den4 * _den8 * ntt1_16 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-432.), _cse7_k2 * _den1 * _den5 * _den7 * ntt1_17 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,216.), _cse11_k2 * _cse7_k2 * _den1 * _den12 * _den5 * _den7 * ntt1_17 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,216.), _cse11_k2 * _cse7_k2 * _den1 * _den13 * _den5 * _den7 * ntt1_17 * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,4.), _cse5_k2 * _den1 * _den3 * _den4 * ntt1_13 * d3V * rhoPhi, fma(complex<double>(0.,-8.), _cse5_k2 * _den1 * _den3 * _den4 * ntt1_14 * d3V * rhoPhi, fma(complex<double>(0.,6.), _cse1_k2 * _den1 * _den10 * _interp6 * ntt1_2 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-3.), _cse1_k2 * _den1 * _den11 * _den16 * _interp6 * ntt1_4 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-3.), _cse1_k2 * _den1 * _den11 * _den17 * _interp6 * ntt1_8 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-576.), _cse7_k2 * _den1 * _den5 * _den7 * ntt1_17 * d2V * d3V * powr<2>(rhoPhi), fma(complex<double>(0.,288.), _cse11_k2 * _cse7_k2 * _den1 * _den12 * _den5 * _den7 * ntt1_17 * d2V * d3V * powr<2>(rhoPhi), fma(complex<double>(0.,288.), _cse11_k2 * _cse7_k2 * _den1 * _den13 * _den5 * _den7 * ntt1_17 * d2V * d3V * powr<2>(rhoPhi), fma(complex<double>(0.,-192.), _cse7_k2 * _den1 * _den5 * _den7 * ntt1_17 * powr<2>(d3V) * powr<3>(rhoPhi), fma(complex<double>(0.,96.), _cse11_k2 * _cse7_k2 * _den1 * _den12 * _den5 * _den7 * ntt1_17 * powr<2>(d3V) * powr<3>(rhoPhi), fma(complex<double>(0.,96.), _cse11_k2 * _cse7_k2 * _den1 * _den13 * _den5 * _den7 * ntt1_17 * powr<2>(d3V) * powr<3>(rhoPhi), 0.))))))))))))))))))))))))));
        // clang-format on
      }
      return DiFfRG::MultiValue<double, 2>(_acc0, _acc1);
    }
    #else
    #error "NumTracer: kernel group 'MesonsPlain' mixes genuinely complex (verdict 0) and real members; generate them as separate flows."
    #endif

    static KOKKOS_INLINE_FUNCTION auto constant(const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      const auto _interp1 = ZPi(p0, p);
      const auto _interp2 = ZSig(p0, p);
      return DiFfRG::MultiValue<double, 2>(_interp1 * etaPi0, _interp2 * etaPi0);
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
using DiFfRG::MesonsPlain_kernel;