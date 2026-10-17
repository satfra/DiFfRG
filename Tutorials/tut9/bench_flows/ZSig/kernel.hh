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
  class ZSig_kernel
  {
    public:
    using Regulator = REG;

    static constexpr bool has_pre = true;

    #if NT_ZSIG_VERDICT == 2   // Pure: the Complex -> Re projection is exact
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
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
      [[maybe_unused]] const auto ntglk2 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk3 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk4 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk5 = ZSig.at(ntgix1);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zsig::nenv) > 0 ? (DiFfRG::zsig::nenv) : 1];
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
      DiFfRG::zsig::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = ntRe(DiFfRG::zsig::tr11(fenv));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ntRe(DiFfRG::zsig::tr12(fenv));
      const auto _interp5 = ntRe(DiFfRG::zsig::tr13(fenv));
      const auto _interp6 = ntRe(DiFfRG::zsig::tr14(fenv));
      const auto _interp7 = ntRe(DiFfRG::zsig::tr10(fenv));
      const auto _interp8 = ntRe(DiFfRG::zsig::tr16(fenv));
      const auto _interp9 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp10 = ntRe(DiFfRG::zsig::tr15(fenv));
      const auto _interp11 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp12 = ntRe(DiFfRG::zsig::tr17(fenv));
      const auto _interp13 = ntRe(DiFfRG::zsig::tr0(fenv));
      const auto _interp14 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp15 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp16 = ntRe(DiFfRG::zsig::tr1(fenv));
      const auto _interp17 = ntRe(DiFfRG::zsig::tr3(fenv));
      const auto _interp18 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp19 = ntRe(DiFfRG::zsig::tr4(fenv));
      const auto _interp20 = ntRe(DiFfRG::zsig::tr6(fenv));
      const auto _interp21 = ntRe(DiFfRG::zsig::tr7(fenv));
      const auto _interp22 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp23 = ntRe(DiFfRG::zsig::tr8(fenv));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den2 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den3 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den4 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk1);
      const auto _den5 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk4 + 2. * d2V * rhoPhi);
      const auto _den6 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk5 + 2. * d2V * rhoPhi);
      const auto _den7 = powr<-1>(_interp9 + d1V + ntglk3 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
      const auto _den8 = powr<-1>(_interp11 + d1V + ntglk2 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
      const auto _den9 = powr<-3>(powr<2>(_interp14 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den10 = powr<-2>(powr<2>(_interp14 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den11 = powr<-1>(_interp9 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
      const auto _den12 = powr<-1>(_interp11 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
      const auto _den13 = powr<-1>(powr<2>(_interp18 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den14 = powr<-1>(powr<2>(_interp18 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den15 = powr<-1>(powr<2>(_interp22 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _cse1 = -_interp3;
      const auto _cse2 = _interp2 * etaPi0;
      const auto _cse3 = _cse1 + _cse2;
      const auto _cse4 = powr<-1>(l1);
      const auto _cse5 = -_interp2 * etaPi0;
      const auto _cse6 = _cse5 + _interp3;
      const auto _cse7 = powr<2>(f0) + powr<2>(l1);
      const auto _cse8 = _cse7 * ntglk4;
      const auto _cse9 = 2. * d2V * rhoPhi;
      const auto _cse10 = _cse8 + _cse9 + _interp2 + d1V;
      // clang-format off
      return 0.04166666666666666 * fma(3., _cse3 * _den1 * _den3 * _den4 * _interp1 * d2V, fma(-6., _cse3 * _den1 * _den3 * _den4 * _interp5 * d2V, fma(-12., _cse4 * _den1 * _den9 * _interp15 * _interp16 * powr<2>(hPhi), fma(6., _cse4 * _den1 * _den10 * _den14 * _interp15 * _interp19 * powr<2>(hPhi), fma(6., _cse4 * _den1 * _den10 * _den13 * _interp15 * _interp20 * powr<2>(hPhi), fma(6., _cse4 * _den1 * _den10 * _den15 * _interp15 * _interp23 * powr<2>(hPhi), fma(6., _cse6 * _den1 * _den3 * _den4 * _den8 * _interp10 * powr<2>(d2V) * rhoPhi, fma(-432., _cse6 * _den1 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(216., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(216., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(-12., _cse6 * _den1 * _den2 * _den4 * _interp7 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den3 * _den4 * _den7 * _interp8 * powr<2>(d2V) * rhoPhi, fma(4., _cse3 * _den1 * _den3 * _den4 * _interp4 * d3V * rhoPhi, fma(-8., _cse3 * _den1 * _den3 * _den4 * _interp6 * d3V * rhoPhi, fma(6., _cse4 * _den1 * _den9 * _interp13 * _interp15 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse4 * _den1 * _den10 * _den14 * _interp15 * _interp17 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse4 * _den1 * _den10 * _den15 * _interp15 * _interp21 * powr<4>(hPhi) * rhoPhi, fma(-576., _cse6 * _den1 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(-192., _cse6 * _den1 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), 0.)))))))))))))))))))))));
      // clang-format on
    }
    static constexpr std::array<bool, 15> pre_reads{true, false, true, false, false, true, true, true, true, true, true, true, false, true, true};
    static KOKKOS_INLINE_FUNCTION auto pre(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto ntgix0 = ZPi.index(-f0, l1);
      [[maybe_unused]] const auto ntgix1 = ZPi.index(f0, l1);
      [[maybe_unused]] const auto ntglk0 = ZPi.at(ntgix0);
      [[maybe_unused]] const auto ntglk1 = ZPi.at(ntgix1);
      [[maybe_unused]] const auto ntglk4 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk5 = ZSig.at(ntgix1);
      const double dr_0 = -f0 - 3.141592653589793 * T;
      const double dr_1 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const double dr_2 = 0.7071067811865475;
      const double dr_3 = hPhi;
      const double dr_4 = sqrt(rhoPhi);
      const double dr_5 = f0 + 3.141592653589793 * T;
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp14 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp15 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _den2 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den3 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den4 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk1);
      const auto _den5 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk4 + 2. * d2V * rhoPhi);
      const auto _den6 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk5 + 2. * d2V * rhoPhi);
      const auto _den9 = powr<-3>(powr<2>(_interp14 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den10 = powr<-2>(powr<2>(_interp14 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _cse1 = -_interp3;
      const auto _cse2 = _interp2 * etaPi0;
      const auto _cse3 = _cse1 + _cse2;
      const auto _cse4 = powr<-1>(l1);
      const auto _cse5 = -_interp2 * etaPi0;
      const auto _cse6 = _cse5 + _interp3;
      const auto _cse7 = powr<2>(f0) + powr<2>(l1);
      const auto _cse8 = _cse7 * ntglk4;
      const auto _cse9 = 2. * d2V * rhoPhi;
      const auto _cse10 = _cse8 + _cse9 + _interp2 + d1V;
      return DiFfRG::device::make_tuple(dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, _interp15, _den2, _den3, _den4, _den5, _den6, _den9, _den10, _cse3, _cse4, _cse6, _cse10);
    }
    static KOKKOS_INLINE_FUNCTION auto post(const auto &P_, const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto &dr_0 = DiFfRG::device::get<0>(P_);
      [[maybe_unused]] const auto &dr_1 = DiFfRG::device::get<1>(P_);
      [[maybe_unused]] const auto &dr_2 = DiFfRG::device::get<2>(P_);
      [[maybe_unused]] const auto &dr_3 = DiFfRG::device::get<3>(P_);
      [[maybe_unused]] const auto &dr_4 = DiFfRG::device::get<4>(P_);
      [[maybe_unused]] const auto &dr_5 = DiFfRG::device::get<5>(P_);
      [[maybe_unused]] const auto &_interp15 = DiFfRG::device::get<6>(P_);
      [[maybe_unused]] const auto &_den2 = DiFfRG::device::get<7>(P_);
      [[maybe_unused]] const auto &_den3 = DiFfRG::device::get<8>(P_);
      [[maybe_unused]] const auto &_den4 = DiFfRG::device::get<9>(P_);
      [[maybe_unused]] const auto &_den5 = DiFfRG::device::get<10>(P_);
      [[maybe_unused]] const auto &_den6 = DiFfRG::device::get<11>(P_);
      [[maybe_unused]] const auto &_den9 = DiFfRG::device::get<12>(P_);
      [[maybe_unused]] const auto &_den10 = DiFfRG::device::get<13>(P_);
      [[maybe_unused]] const auto &_cse3 = DiFfRG::device::get<14>(P_);
      [[maybe_unused]] const auto &_cse4 = DiFfRG::device::get<15>(P_);
      [[maybe_unused]] const auto &_cse6 = DiFfRG::device::get<16>(P_);
      [[maybe_unused]] const auto &_cse10 = DiFfRG::device::get<17>(P_);
      [[maybe_unused]] const auto ntgix2 = ZPi.index(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntgix3 = ZPi.index(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntglk2 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk3 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zsig::nenv) > 0 ? (DiFfRG::zsig::nenv) : 1];
      const double dr_6 = -f0 + p0 - 3.141592653589793 * T;
      const double dr_7 = 1. + sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_8 = -1. - sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_9 = f0 - p0 + 3.141592653589793 * T;
      const double dr_10 = -f0 - p0 - 3.141592653589793 * T;
      const double dr_11 = 1. + sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_12 = f0 + p0 + 3.141592653589793 * T;
      DiFfRG::zsig::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = ntRe(DiFfRG::zsig::tr11(fenv));
      const auto _interp4 = ntRe(DiFfRG::zsig::tr12(fenv));
      const auto _interp5 = ntRe(DiFfRG::zsig::tr13(fenv));
      const auto _interp6 = ntRe(DiFfRG::zsig::tr14(fenv));
      const auto _interp7 = ntRe(DiFfRG::zsig::tr10(fenv));
      const auto _interp8 = ntRe(DiFfRG::zsig::tr16(fenv));
      const auto _interp9 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp10 = ntRe(DiFfRG::zsig::tr15(fenv));
      const auto _interp11 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp12 = ntRe(DiFfRG::zsig::tr17(fenv));
      const auto _interp13 = ntRe(DiFfRG::zsig::tr0(fenv));
      const auto _interp16 = ntRe(DiFfRG::zsig::tr1(fenv));
      const auto _interp17 = ntRe(DiFfRG::zsig::tr3(fenv));
      const auto _interp18 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp19 = ntRe(DiFfRG::zsig::tr4(fenv));
      const auto _interp20 = ntRe(DiFfRG::zsig::tr6(fenv));
      const auto _interp21 = ntRe(DiFfRG::zsig::tr7(fenv));
      const auto _interp22 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp23 = ntRe(DiFfRG::zsig::tr8(fenv));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den7 = powr<-1>(_interp9 + d1V + ntglk3 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
      const auto _den8 = powr<-1>(_interp11 + d1V + ntglk2 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
      const auto _den11 = powr<-1>(_interp9 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
      const auto _den12 = powr<-1>(_interp11 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
      const auto _den13 = powr<-1>(powr<2>(_interp18 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den14 = powr<-1>(powr<2>(_interp18 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den15 = powr<-1>(powr<2>(_interp22 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      // clang-format off
      return 0.04166666666666666 * fma(3., _cse3 * _den1 * _den3 * _den4 * _interp1 * d2V, fma(-6., _cse3 * _den1 * _den3 * _den4 * _interp5 * d2V, fma(-12., _cse4 * _den1 * _den9 * _interp15 * _interp16 * powr<2>(hPhi), fma(6., _cse4 * _den1 * _den10 * _den14 * _interp15 * _interp19 * powr<2>(hPhi), fma(6., _cse4 * _den1 * _den10 * _den13 * _interp15 * _interp20 * powr<2>(hPhi), fma(6., _cse4 * _den1 * _den10 * _den15 * _interp15 * _interp23 * powr<2>(hPhi), fma(6., _cse6 * _den1 * _den3 * _den4 * _den8 * _interp10 * powr<2>(d2V) * rhoPhi, fma(-432., _cse6 * _den1 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(216., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(216., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(-12., _cse6 * _den1 * _den2 * _den4 * _interp7 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den3 * _den4 * _den7 * _interp8 * powr<2>(d2V) * rhoPhi, fma(4., _cse3 * _den1 * _den3 * _den4 * _interp4 * d3V * rhoPhi, fma(-8., _cse3 * _den1 * _den3 * _den4 * _interp6 * d3V * rhoPhi, fma(6., _cse4 * _den1 * _den9 * _interp13 * _interp15 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse4 * _den1 * _den10 * _den14 * _interp15 * _interp17 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse4 * _den1 * _den10 * _den15 * _interp15 * _interp21 * powr<4>(hPhi) * rhoPhi, fma(-576., _cse6 * _den1 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(-192., _cse6 * _den1 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), 0.)))))))))))))))))))))));
      // clang-format on
    }
    #elif NT_ZSIG_VERDICT == 1   // RePart: real value via complex trace(s), re/im split
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
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
      [[maybe_unused]] const auto ntglk2 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk3 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk4 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk5 = ZSig.at(ntgix1);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zsig::nenv) > 0 ? (DiFfRG::zsig::nenv) : 1];
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
      DiFfRG::zsig::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = ntRe(DiFfRG::zsig::tr11(fenv));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ntRe(DiFfRG::zsig::tr12(fenv));
      const auto _interp5 = ntRe(DiFfRG::zsig::tr13(fenv));
      const auto _interp6 = ntRe(DiFfRG::zsig::tr14(fenv));
      const auto _interp7 = ntRe(DiFfRG::zsig::tr10(fenv));
      const auto _interp8 = ntRe(DiFfRG::zsig::tr16(fenv));
      const auto _interp9 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp10 = ntRe(DiFfRG::zsig::tr15(fenv));
      const auto _interp11 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp12 = ntRe(DiFfRG::zsig::tr17(fenv));
      const auto _interp13 = ntIm(DiFfRG::zsig::tr2(fenv));
      const auto _interp14 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp15 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp16 = ntRe(DiFfRG::zsig::tr0(fenv));
      const auto _interp17 = ntRe(DiFfRG::zsig::tr1(fenv));
      const auto _interp18 = ntRe(DiFfRG::zsig::tr3(fenv));
      const auto _interp19 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp20 = ntRe(DiFfRG::zsig::tr4(fenv));
      const auto _interp21 = ntIm(DiFfRG::zsig::tr5(fenv));
      const auto _interp22 = ntRe(DiFfRG::zsig::tr6(fenv));
      const auto _interp23 = ntRe(DiFfRG::zsig::tr7(fenv));
      const auto _interp24 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp25 = ntRe(DiFfRG::zsig::tr8(fenv));
      const auto _interp26 = ntIm(DiFfRG::zsig::tr9(fenv));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den2 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den3 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den4 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk1);
      const auto _den5 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk4 + 2. * d2V * rhoPhi);
      const auto _den6 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk5 + 2. * d2V * rhoPhi);
      const auto _den7 = powr<-1>(_interp9 + d1V + ntglk3 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
      const auto _den8 = powr<-1>(_interp11 + d1V + ntglk2 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
      const auto _den9 = powr<-3>(powr<2>(_interp14 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den10 = powr<-2>(powr<2>(_interp14 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den11 = powr<-1>(_interp9 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
      const auto _den12 = powr<-1>(_interp11 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
      const auto _den13 = powr<-1>(powr<2>(_interp24 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den14 = powr<-1>(powr<2>(_interp19 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den15 = powr<-1>(powr<2>(_interp19 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den16 = powr<-1>(powr<2>(_interp24 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _cse1 = powr<-1>(l1);
      const auto _cse2 = -_interp3;
      const auto _cse3 = _interp2 * etaPi0;
      const auto _cse4 = _cse2 + _cse3;
      const auto _cse5 = -_interp2 * etaPi0;
      const auto _cse6 = _cse5 + _interp3;
      const auto _cse7 = powr<2>(f0) + powr<2>(l1);
      const auto _cse8 = _cse7 * ntglk4;
      const auto _cse9 = 2. * d2V * rhoPhi;
      const auto _cse10 = _cse8 + _cse9 + _interp2 + d1V;
      // clang-format off
      return 0.04166666666666666 * fma(3., _cse4 * _den1 * _den3 * _den4 * _interp1 * d2V, fma(-6., _cse4 * _den1 * _den3 * _den4 * _interp5 * d2V, fma(12., _cse1 * _den1 * _den9 * _interp13 * _interp15 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den9 * _interp15 * _interp17 * powr<2>(hPhi), fma(6., _cse1 * _den1 * _den10 * _den15 * _interp15 * _interp20 * powr<2>(hPhi), fma(6., _cse1 * _den1 * _den10 * _den14 * _interp15 * _interp22 * powr<2>(hPhi), fma(6., _cse1 * _den1 * _den10 * _den16 * _interp15 * _interp25 * powr<2>(hPhi), fma(-6., _cse1 * _den1 * _den10 * _den13 * _interp15 * _interp26 * powr<2>(hPhi), fma(3., 1.414213562373095 * _cse1 * _den1 * _den10 * _den14 * _interp15 * _interp21 * powr<3>(hPhi) * sqrt(rhoPhi), fma(6., _cse6 * _den1 * _den3 * _den4 * _den8 * _interp10 * powr<2>(d2V) * rhoPhi, fma(-432., _cse6 * _den1 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(216., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(216., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(-12., _cse6 * _den1 * _den2 * _den4 * _interp7 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den3 * _den4 * _den7 * _interp8 * powr<2>(d2V) * rhoPhi, fma(4., _cse4 * _den1 * _den3 * _den4 * _interp4 * d3V * rhoPhi, fma(-8., _cse4 * _den1 * _den3 * _den4 * _interp6 * d3V * rhoPhi, fma(6., _cse1 * _den1 * _den9 * _interp15 * _interp16 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse1 * _den1 * _den10 * _den15 * _interp15 * _interp18 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse1 * _den1 * _den10 * _den16 * _interp15 * _interp23 * powr<4>(hPhi) * rhoPhi, fma(-576., _cse6 * _den1 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(-192., _cse6 * _den1 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), 0.))))))))))))))))))))))))));
      // clang-format on
    }
    static constexpr std::array<bool, 15> pre_reads{true, false, true, false, false, true, true, true, true, true, true, true, false, true, true};
    static KOKKOS_INLINE_FUNCTION auto pre(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto ntgix0 = ZPi.index(-f0, l1);
      [[maybe_unused]] const auto ntgix1 = ZPi.index(f0, l1);
      [[maybe_unused]] const auto ntglk0 = ZPi.at(ntgix0);
      [[maybe_unused]] const auto ntglk1 = ZPi.at(ntgix1);
      [[maybe_unused]] const auto ntglk4 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk5 = ZSig.at(ntgix1);
      const double dr_0 = -f0 - 3.141592653589793 * T;
      const double dr_1 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const double dr_2 = 0.7071067811865475;
      const double dr_3 = hPhi;
      const double dr_4 = sqrt(rhoPhi);
      const double dr_5 = f0 + 3.141592653589793 * T;
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp14 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp15 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _den2 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den3 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den4 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk1);
      const auto _den5 = powr<-2>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk4 + 2. * d2V * rhoPhi);
      const auto _den6 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk5 + 2. * d2V * rhoPhi);
      const auto _den9 = powr<-3>(powr<2>(_interp14 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den10 = powr<-2>(powr<2>(_interp14 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _cse1 = powr<-1>(l1);
      const auto _cse2 = -_interp3;
      const auto _cse3 = _interp2 * etaPi0;
      const auto _cse4 = _cse2 + _cse3;
      const auto _cse5 = -_interp2 * etaPi0;
      const auto _cse6 = _cse5 + _interp3;
      const auto _cse7 = powr<2>(f0) + powr<2>(l1);
      const auto _cse8 = _cse7 * ntglk4;
      const auto _cse9 = 2. * d2V * rhoPhi;
      const auto _cse10 = _cse8 + _cse9 + _interp2 + d1V;
      return DiFfRG::device::make_tuple(dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, _interp15, _den2, _den3, _den4, _den5, _den6, _den9, _den10, _cse1, _cse4, _cse6, _cse10);
    }
    static KOKKOS_INLINE_FUNCTION auto post(const auto &P_, const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto &dr_0 = DiFfRG::device::get<0>(P_);
      [[maybe_unused]] const auto &dr_1 = DiFfRG::device::get<1>(P_);
      [[maybe_unused]] const auto &dr_2 = DiFfRG::device::get<2>(P_);
      [[maybe_unused]] const auto &dr_3 = DiFfRG::device::get<3>(P_);
      [[maybe_unused]] const auto &dr_4 = DiFfRG::device::get<4>(P_);
      [[maybe_unused]] const auto &dr_5 = DiFfRG::device::get<5>(P_);
      [[maybe_unused]] const auto &_interp15 = DiFfRG::device::get<6>(P_);
      [[maybe_unused]] const auto &_den2 = DiFfRG::device::get<7>(P_);
      [[maybe_unused]] const auto &_den3 = DiFfRG::device::get<8>(P_);
      [[maybe_unused]] const auto &_den4 = DiFfRG::device::get<9>(P_);
      [[maybe_unused]] const auto &_den5 = DiFfRG::device::get<10>(P_);
      [[maybe_unused]] const auto &_den6 = DiFfRG::device::get<11>(P_);
      [[maybe_unused]] const auto &_den9 = DiFfRG::device::get<12>(P_);
      [[maybe_unused]] const auto &_den10 = DiFfRG::device::get<13>(P_);
      [[maybe_unused]] const auto &_cse1 = DiFfRG::device::get<14>(P_);
      [[maybe_unused]] const auto &_cse4 = DiFfRG::device::get<15>(P_);
      [[maybe_unused]] const auto &_cse6 = DiFfRG::device::get<16>(P_);
      [[maybe_unused]] const auto &_cse10 = DiFfRG::device::get<17>(P_);
      [[maybe_unused]] const auto ntgix2 = ZPi.index(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntgix3 = ZPi.index(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntglk2 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk3 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zsig::nenv) > 0 ? (DiFfRG::zsig::nenv) : 1];
      const double dr_6 = -f0 + p0 - 3.141592653589793 * T;
      const double dr_7 = 1. + sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_8 = -1. - sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_9 = f0 - p0 + 3.141592653589793 * T;
      const double dr_10 = -f0 - p0 - 3.141592653589793 * T;
      const double dr_11 = 1. + sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_12 = f0 + p0 + 3.141592653589793 * T;
      DiFfRG::zsig::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = ntRe(DiFfRG::zsig::tr11(fenv));
      const auto _interp4 = ntRe(DiFfRG::zsig::tr12(fenv));
      const auto _interp5 = ntRe(DiFfRG::zsig::tr13(fenv));
      const auto _interp6 = ntRe(DiFfRG::zsig::tr14(fenv));
      const auto _interp7 = ntRe(DiFfRG::zsig::tr10(fenv));
      const auto _interp8 = ntRe(DiFfRG::zsig::tr16(fenv));
      const auto _interp9 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp10 = ntRe(DiFfRG::zsig::tr15(fenv));
      const auto _interp11 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp12 = ntRe(DiFfRG::zsig::tr17(fenv));
      const auto _interp13 = ntIm(DiFfRG::zsig::tr2(fenv));
      const auto _interp16 = ntRe(DiFfRG::zsig::tr0(fenv));
      const auto _interp17 = ntRe(DiFfRG::zsig::tr1(fenv));
      const auto _interp18 = ntRe(DiFfRG::zsig::tr3(fenv));
      const auto _interp19 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp20 = ntRe(DiFfRG::zsig::tr4(fenv));
      const auto _interp21 = ntIm(DiFfRG::zsig::tr5(fenv));
      const auto _interp22 = ntRe(DiFfRG::zsig::tr6(fenv));
      const auto _interp23 = ntRe(DiFfRG::zsig::tr7(fenv));
      const auto _interp24 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp25 = ntRe(DiFfRG::zsig::tr8(fenv));
      const auto _interp26 = ntIm(DiFfRG::zsig::tr9(fenv));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den7 = powr<-1>(_interp9 + d1V + ntglk3 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
      const auto _den8 = powr<-1>(_interp11 + d1V + ntglk2 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
      const auto _den11 = powr<-1>(_interp9 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
      const auto _den12 = powr<-1>(_interp11 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
      const auto _den13 = powr<-1>(powr<2>(_interp24 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den14 = powr<-1>(powr<2>(_interp19 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den15 = powr<-1>(powr<2>(_interp19 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den16 = powr<-1>(powr<2>(_interp24 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      // clang-format off
      return 0.04166666666666666 * fma(3., _cse4 * _den1 * _den3 * _den4 * _interp1 * d2V, fma(-6., _cse4 * _den1 * _den3 * _den4 * _interp5 * d2V, fma(12., _cse1 * _den1 * _den9 * _interp13 * _interp15 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den9 * _interp15 * _interp17 * powr<2>(hPhi), fma(6., _cse1 * _den1 * _den10 * _den15 * _interp15 * _interp20 * powr<2>(hPhi), fma(6., _cse1 * _den1 * _den10 * _den14 * _interp15 * _interp22 * powr<2>(hPhi), fma(6., _cse1 * _den1 * _den10 * _den16 * _interp15 * _interp25 * powr<2>(hPhi), fma(-6., _cse1 * _den1 * _den10 * _den13 * _interp15 * _interp26 * powr<2>(hPhi), fma(3., 1.414213562373095 * _cse1 * _den1 * _den10 * _den14 * _interp15 * _interp21 * powr<3>(hPhi) * sqrt(rhoPhi), fma(6., _cse6 * _den1 * _den3 * _den4 * _den8 * _interp10 * powr<2>(d2V) * rhoPhi, fma(-432., _cse6 * _den1 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(216., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(216., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * powr<2>(d2V) * rhoPhi, fma(-12., _cse6 * _den1 * _den2 * _den4 * _interp7 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den3 * _den4 * _den7 * _interp8 * powr<2>(d2V) * rhoPhi, fma(4., _cse4 * _den1 * _den3 * _den4 * _interp4 * d3V * rhoPhi, fma(-8., _cse4 * _den1 * _den3 * _den4 * _interp6 * d3V * rhoPhi, fma(6., _cse1 * _den1 * _den9 * _interp15 * _interp16 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse1 * _den1 * _den10 * _den15 * _interp15 * _interp18 * powr<4>(hPhi) * rhoPhi, fma(-3., _cse1 * _den1 * _den10 * _den16 * _interp15 * _interp23 * powr<4>(hPhi) * rhoPhi, fma(-576., _cse6 * _den1 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(288., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * d2V * d3V * powr<2>(rhoPhi), fma(-192., _cse6 * _den1 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), fma(96., _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * _interp12 * powr<2>(d3V) * powr<3>(rhoPhi), 0.))))))))))))))))))))))))));
      // clang-format on
    }
    #else                              // the imaginary part survives: genuinely complex
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
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
      [[maybe_unused]] const auto ntglk2 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk3 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk4 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk5 = ZSig.at(ntgix1);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zsig::nenv) > 0 ? (DiFfRG::zsig::nenv) : 1];
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
      DiFfRG::zsig::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp4 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp5 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp6 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp7 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp8 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den2 = powr<-2>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den3 = powr<-1>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den4 = powr<-1>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk1);
      const auto _den5 = powr<-2>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk4 + 2. * d2V * rhoPhi);
      const auto _den6 = powr<-1>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk5 + 2. * d2V * rhoPhi);
      const auto _den7 = powr<-1>(_interp3 + d1V + ntglk3 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
      const auto _den8 = powr<-1>(_interp4 + d1V + ntglk2 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
      const auto _den9 = powr<-3>(powr<2>(_interp5 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den10 = powr<-2>(powr<2>(_interp5 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den11 = powr<-1>(_interp3 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
      const auto _den12 = powr<-1>(_interp4 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
      const auto _den13 = powr<-1>(powr<2>(_interp8 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den14 = powr<-1>(powr<2>(_interp7 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den15 = powr<-1>(powr<2>(_interp7 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den16 = powr<-1>(powr<2>(_interp8 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _cse1 = powr<-1>(l1);
      const auto _cse2 = -_interp2;
      const auto _cse3 = _interp1 * etaPi0;
      const auto _cse4 = _cse2 + _cse3;
      const auto _cse5 = -_interp1 * etaPi0;
      const auto _cse6 = _cse5 + _interp2;
      const auto _cse7 = powr<2>(f0) + powr<2>(l1);
      const auto _cse8 = _cse7 * ntglk4;
      const auto _cse9 = 2. * d2V * rhoPhi;
      const auto _cse10 = _cse8 + _cse9 + _interp1 + d1V;
      // clang-format off
      return complex<double>(0.,-0.04166666666666666) * fma(complex<double>(0.,3.), _cse4 * _den1 * _den3 * _den4 * DiFfRG::zsig::tr11(fenv) * d2V, fma(complex<double>(0.,-6.), _cse4 * _den1 * _den3 * _den4 * DiFfRG::zsig::tr13(fenv) * d2V, fma(complex<double>(0.,-12.), _cse1 * _den1 * _den9 * DiFfRG::zsig::tr1(fenv) * _interp6 * powr<2>(hPhi), fma(12., _cse1 * _den1 * _den9 * DiFfRG::zsig::tr2(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,6.), _cse1 * _den1 * _den10 * _den15 * DiFfRG::zsig::tr4(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,6.), _cse1 * _den1 * _den10 * _den14 * DiFfRG::zsig::tr6(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,6.), _cse1 * _den1 * _den10 * _den16 * DiFfRG::zsig::tr8(fenv) * _interp6 * powr<2>(hPhi), fma(-6., _cse1 * _den1 * _den10 * _den13 * DiFfRG::zsig::tr9(fenv) * _interp6 * powr<2>(hPhi), fma(3., 1.414213562373095 * _cse1 * _den1 * _den10 * _den14 * DiFfRG::zsig::tr5(fenv) * _interp6 * powr<3>(hPhi) * sqrt(rhoPhi), fma(complex<double>(0.,-12.), _cse6 * _den1 * _den2 * _den4 * DiFfRG::zsig::tr10(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse6 * _den1 * _den3 * _den4 * _den8 * DiFfRG::zsig::tr15(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse6 * _den1 * _den3 * _den4 * _den7 * DiFfRG::zsig::tr16(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-432.), _cse6 * _den1 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,216.), _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,216.), _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,4.), _cse4 * _den1 * _den3 * _den4 * DiFfRG::zsig::tr12(fenv) * d3V * rhoPhi, fma(complex<double>(0.,-8.), _cse4 * _den1 * _den3 * _den4 * DiFfRG::zsig::tr14(fenv) * d3V * rhoPhi, fma(complex<double>(0.,6.), _cse1 * _den1 * _den9 * DiFfRG::zsig::tr0(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-3.), _cse1 * _den1 * _den10 * _den15 * DiFfRG::zsig::tr3(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-3.), _cse1 * _den1 * _den10 * _den16 * DiFfRG::zsig::tr7(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-576.), _cse6 * _den1 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * d2V * d3V * powr<2>(rhoPhi), fma(complex<double>(0.,288.), _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * d2V * d3V * powr<2>(rhoPhi), fma(complex<double>(0.,288.), _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * d2V * d3V * powr<2>(rhoPhi), fma(complex<double>(0.,-192.), _cse6 * _den1 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d3V) * powr<3>(rhoPhi), fma(complex<double>(0.,96.), _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d3V) * powr<3>(rhoPhi), fma(complex<double>(0.,96.), _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d3V) * powr<3>(rhoPhi), 0.))))))))))))))))))))))))));
      // clang-format on
    }
    static constexpr std::array<bool, 15> pre_reads{true, false, true, false, false, true, true, true, true, true, true, true, false, true, true};
    static KOKKOS_INLINE_FUNCTION auto pre(const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto ntgix0 = ZPi.index(-f0, l1);
      [[maybe_unused]] const auto ntgix1 = ZPi.index(f0, l1);
      [[maybe_unused]] const auto ntglk0 = ZPi.at(ntgix0);
      [[maybe_unused]] const auto ntglk1 = ZPi.at(ntgix1);
      [[maybe_unused]] const auto ntglk4 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk5 = ZSig.at(ntgix1);
      const double dr_0 = -f0 - 3.141592653589793 * T;
      const double dr_1 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const double dr_2 = 0.7071067811865475;
      const double dr_3 = hPhi;
      const double dr_4 = sqrt(rhoPhi);
      const double dr_5 = f0 + 3.141592653589793 * T;
      const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp5 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp6 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _den2 = powr<-2>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den3 = powr<-1>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den4 = powr<-1>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk1);
      const auto _den5 = powr<-2>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk4 + 2. * d2V * rhoPhi);
      const auto _den6 = powr<-1>(_interp1 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk5 + 2. * d2V * rhoPhi);
      const auto _den9 = powr<-3>(powr<2>(_interp5 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den10 = powr<-2>(powr<2>(_interp5 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _cse1 = powr<-1>(l1);
      const auto _cse2 = -_interp2;
      const auto _cse3 = _interp1 * etaPi0;
      const auto _cse4 = _cse2 + _cse3;
      const auto _cse5 = -_interp1 * etaPi0;
      const auto _cse6 = _cse5 + _interp2;
      const auto _cse7 = powr<2>(f0) + powr<2>(l1);
      const auto _cse8 = _cse7 * ntglk4;
      const auto _cse9 = 2. * d2V * rhoPhi;
      const auto _cse10 = _cse8 + _cse9 + _interp1 + d1V;
      return DiFfRG::device::make_tuple(dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, _interp6, _den2, _den3, _den4, _den5, _den6, _den9, _den10, _cse1, _cse4, _cse6, _cse10);
    }
    static KOKKOS_INLINE_FUNCTION auto post(const auto &P_, const double& l1, const double& cos1, const double& f0, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto &dr_0 = DiFfRG::device::get<0>(P_);
      [[maybe_unused]] const auto &dr_1 = DiFfRG::device::get<1>(P_);
      [[maybe_unused]] const auto &dr_2 = DiFfRG::device::get<2>(P_);
      [[maybe_unused]] const auto &dr_3 = DiFfRG::device::get<3>(P_);
      [[maybe_unused]] const auto &dr_4 = DiFfRG::device::get<4>(P_);
      [[maybe_unused]] const auto &dr_5 = DiFfRG::device::get<5>(P_);
      [[maybe_unused]] const auto &_interp6 = DiFfRG::device::get<6>(P_);
      [[maybe_unused]] const auto &_den2 = DiFfRG::device::get<7>(P_);
      [[maybe_unused]] const auto &_den3 = DiFfRG::device::get<8>(P_);
      [[maybe_unused]] const auto &_den4 = DiFfRG::device::get<9>(P_);
      [[maybe_unused]] const auto &_den5 = DiFfRG::device::get<10>(P_);
      [[maybe_unused]] const auto &_den6 = DiFfRG::device::get<11>(P_);
      [[maybe_unused]] const auto &_den9 = DiFfRG::device::get<12>(P_);
      [[maybe_unused]] const auto &_den10 = DiFfRG::device::get<13>(P_);
      [[maybe_unused]] const auto &_cse1 = DiFfRG::device::get<14>(P_);
      [[maybe_unused]] const auto &_cse4 = DiFfRG::device::get<15>(P_);
      [[maybe_unused]] const auto &_cse6 = DiFfRG::device::get<16>(P_);
      [[maybe_unused]] const auto &_cse10 = DiFfRG::device::get<17>(P_);
      [[maybe_unused]] const auto ntgix2 = ZPi.index(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntgix3 = ZPi.index(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntglk2 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk3 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zsig::nenv) > 0 ? (DiFfRG::zsig::nenv) : 1];
      const double dr_6 = -f0 + p0 - 3.141592653589793 * T;
      const double dr_7 = 1. + sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_8 = -1. - sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_9 = f0 - p0 + 3.141592653589793 * T;
      const double dr_10 = -f0 - p0 - 3.141592653589793 * T;
      const double dr_11 = 1. + sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_12 = f0 + p0 + 3.141592653589793 * T;
      DiFfRG::zsig::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp3 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp4 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp7 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp8 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den7 = powr<-1>(_interp3 + d1V + ntglk3 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
      const auto _den8 = powr<-1>(_interp4 + d1V + ntglk2 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
      const auto _den11 = powr<-1>(_interp3 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
      const auto _den12 = powr<-1>(_interp4 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
      const auto _den13 = powr<-1>(powr<2>(_interp8 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den14 = powr<-1>(powr<2>(_interp7 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den15 = powr<-1>(powr<2>(_interp7 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den16 = powr<-1>(powr<2>(_interp8 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      // clang-format off
      return complex<double>(0.,-0.04166666666666666) * fma(complex<double>(0.,3.), _cse4 * _den1 * _den3 * _den4 * DiFfRG::zsig::tr11(fenv) * d2V, fma(complex<double>(0.,-6.), _cse4 * _den1 * _den3 * _den4 * DiFfRG::zsig::tr13(fenv) * d2V, fma(complex<double>(0.,-12.), _cse1 * _den1 * _den9 * DiFfRG::zsig::tr1(fenv) * _interp6 * powr<2>(hPhi), fma(12., _cse1 * _den1 * _den9 * DiFfRG::zsig::tr2(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,6.), _cse1 * _den1 * _den10 * _den15 * DiFfRG::zsig::tr4(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,6.), _cse1 * _den1 * _den10 * _den14 * DiFfRG::zsig::tr6(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,6.), _cse1 * _den1 * _den10 * _den16 * DiFfRG::zsig::tr8(fenv) * _interp6 * powr<2>(hPhi), fma(-6., _cse1 * _den1 * _den10 * _den13 * DiFfRG::zsig::tr9(fenv) * _interp6 * powr<2>(hPhi), fma(3., 1.414213562373095 * _cse1 * _den1 * _den10 * _den14 * DiFfRG::zsig::tr5(fenv) * _interp6 * powr<3>(hPhi) * sqrt(rhoPhi), fma(complex<double>(0.,-12.), _cse6 * _den1 * _den2 * _den4 * DiFfRG::zsig::tr10(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse6 * _den1 * _den3 * _den4 * _den8 * DiFfRG::zsig::tr15(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,6.), _cse6 * _den1 * _den3 * _den4 * _den7 * DiFfRG::zsig::tr16(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-432.), _cse6 * _den1 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,216.), _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,216.), _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,4.), _cse4 * _den1 * _den3 * _den4 * DiFfRG::zsig::tr12(fenv) * d3V * rhoPhi, fma(complex<double>(0.,-8.), _cse4 * _den1 * _den3 * _den4 * DiFfRG::zsig::tr14(fenv) * d3V * rhoPhi, fma(complex<double>(0.,6.), _cse1 * _den1 * _den9 * DiFfRG::zsig::tr0(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-3.), _cse1 * _den1 * _den10 * _den15 * DiFfRG::zsig::tr3(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-3.), _cse1 * _den1 * _den10 * _den16 * DiFfRG::zsig::tr7(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-576.), _cse6 * _den1 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * d2V * d3V * powr<2>(rhoPhi), fma(complex<double>(0.,288.), _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * d2V * d3V * powr<2>(rhoPhi), fma(complex<double>(0.,288.), _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * d2V * d3V * powr<2>(rhoPhi), fma(complex<double>(0.,-192.), _cse6 * _den1 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d3V) * powr<3>(rhoPhi), fma(complex<double>(0.,96.), _cse10 * _cse6 * _den1 * _den11 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d3V) * powr<3>(rhoPhi), fma(complex<double>(0.,96.), _cse10 * _cse6 * _den1 * _den12 * _den5 * _den6 * DiFfRG::zsig::tr17(fenv) * powr<2>(d3V) * powr<3>(rhoPhi), 0.))))))))))))))))))))))))));
      // clang-format on
    }
    #endif

    static KOKKOS_INLINE_FUNCTION auto constant(const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      const auto _interp1 = ZSig(p0, p);
      return _interp1 * etaPi0;
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
using DiFfRG::ZSig_kernel;