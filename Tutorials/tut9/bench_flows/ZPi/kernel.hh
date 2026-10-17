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
  class ZPi_kernel
  {
    public:
    using Regulator = REG;

    static constexpr bool has_pre = true;

    #if NT_ZPI_VERDICT == 2   // Pure: the Complex -> Re projection is exact
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
      [[maybe_unused]] const auto ntglk2 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk3 = ZSig.at(ntgix1);
      [[maybe_unused]] const auto ntglk4 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk5 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zpi::nenv) > 0 ? (DiFfRG::zpi::nenv) : 1];
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
      DiFfRG::zpi::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = ntRe(DiFfRG::zpi::tr16(fenv));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ntRe(DiFfRG::zpi::tr17(fenv));
      const auto _interp5 = ntRe(DiFfRG::zpi::tr19(fenv));
      const auto _interp6 = ntRe(DiFfRG::zpi::tr20(fenv));
      const auto _interp7 = ntRe(DiFfRG::zpi::tr21(fenv));
      const auto _interp8 = ntRe(DiFfRG::zpi::tr22(fenv));
      const auto _interp9 = ntRe(DiFfRG::zpi::tr18(fenv));
      const auto _interp10 = ntRe(DiFfRG::zpi::tr23(fenv));
      const auto _interp11 = ntRe(DiFfRG::zpi::tr25(fenv));
      const auto _interp12 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp13 = ntRe(DiFfRG::zpi::tr27(fenv));
      const auto _interp14 = ntRe(DiFfRG::zpi::tr24(fenv));
      const auto _interp15 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp16 = ntRe(DiFfRG::zpi::tr26(fenv));
      const auto _interp17 = ntRe(DiFfRG::zpi::tr1(fenv));
      const auto _interp18 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp19 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp20 = ntRe(DiFfRG::zpi::tr3(fenv));
      const auto _interp21 = ntRe(DiFfRG::zpi::tr6(fenv));
      const auto _interp22 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp23 = ntRe(DiFfRG::zpi::tr8(fenv));
      const auto _interp24 = ntRe(DiFfRG::zpi::tr10(fenv));
      const auto _interp25 = ntRe(DiFfRG::zpi::tr12(fenv));
      const auto _interp26 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp27 = ntRe(DiFfRG::zpi::tr14(fenv));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den2 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den3 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk1);
      const auto _den4 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk2 + 2. * d2V * rhoPhi);
      const auto _den5 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk3 + 2. * d2V * rhoPhi);
      const auto _den6 = powr<-1>(_interp12 + d1V + ntglk5 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
      const auto _den7 = powr<-1>(_interp15 + d1V + ntglk4 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
      const auto _den8 = powr<-3>(powr<2>(_interp18 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den9 = powr<-2>(powr<2>(_interp18 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den10 = powr<-1>(_interp12 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
      const auto _den11 = powr<-1>(_interp15 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
      const auto _den12 = powr<-1>(powr<2>(_interp22 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den13 = powr<-1>(powr<2>(_interp22 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den14 = powr<-1>(powr<2>(_interp26 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _cse1 = powr<-1>(l1);
      const auto _cse2 = -_interp3;
      const auto _cse3 = _interp2 * etaPi0;
      const auto _cse4 = _cse2 + _cse3;
      const auto _cse5 = -_interp2 * etaPi0;
      const auto _cse6 = _cse5 + _interp3;
      // clang-format off
      return 0.01388888888888889 * fma(3., _cse4 * _den1 * _den2 * _den3 * _interp1 * d2V, fma(-6., _cse4 * _den1 * _den2 * _den3 * _interp4 * d2V, fma(3., _cse4 * _den1 * _den4 * _den5 * _interp5 * d2V, fma(-6., _cse4 * _den1 * _den4 * _den5 * _interp7 * d2V, fma(24., _cse1 * _den1 * _den8 * _interp19 * _interp20 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den13 * _den9 * _interp19 * _interp23 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den12 * _den9 * _interp19 * _interp24 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den14 * _den9 * _interp19 * _interp27 * powr<2>(hPhi), fma(-12., _cse6 * _den1 * _den2 * _den4 * _den5 * _interp10 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den4 * _den5 * _den6 * _interp11 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den10 * _den2 * _den3 * _interp13 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den4 * _den5 * _den7 * _interp14 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den11 * _den2 * _den3 * _interp16 * powr<2>(d2V) * rhoPhi, fma(-12., _cse6 * _den1 * _den2 * _den3 * _den4 * _interp9 * powr<2>(d2V) * rhoPhi, fma(4., _cse4 * _den1 * _den4 * _den5 * _interp6 * d3V * rhoPhi, fma(-8., _cse4 * _den1 * _den4 * _den5 * _interp8 * d3V * rhoPhi, fma(-12., _cse1 * _den1 * _den8 * _interp17 * _interp19 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1 * _den1 * _den13 * _den9 * _interp19 * _interp21 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1 * _den1 * _den14 * _den9 * _interp19 * _interp25 * powr<4>(hPhi) * rhoPhi, 0.)))))))))))))))))));
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
      [[maybe_unused]] const auto ntglk2 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk3 = ZSig.at(ntgix1);
      const double dr_0 = -f0 - 3.141592653589793 * T;
      const double dr_1 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const double dr_2 = 0.7071067811865475;
      const double dr_3 = hPhi;
      const double dr_4 = sqrt(rhoPhi);
      const double dr_5 = f0 + 3.141592653589793 * T;
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp18 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp19 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _den2 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk0);
      const auto _den3 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk1);
      const auto _den4 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk2 + 2. * d2V * rhoPhi);
      const auto _den5 = powr<-1>(_interp2 + d1V + (powr<2>(f0) + powr<2>(l1)) * ntglk3 + 2. * d2V * rhoPhi);
      const auto _den8 = powr<-3>(powr<2>(_interp18 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _den9 = powr<-2>(powr<2>(_interp18 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _cse1 = powr<-1>(l1);
      const auto _cse2 = -_interp3;
      const auto _cse3 = _interp2 * etaPi0;
      const auto _cse4 = _cse2 + _cse3;
      const auto _cse5 = -_interp2 * etaPi0;
      const auto _cse6 = _cse5 + _interp3;
      return DiFfRG::device::make_tuple(dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, _interp19, _den2, _den3, _den4, _den5, _den8, _den9, _cse1, _cse4, _cse6);
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
      [[maybe_unused]] const auto &_interp19 = DiFfRG::device::get<6>(P_);
      [[maybe_unused]] const auto &_den2 = DiFfRG::device::get<7>(P_);
      [[maybe_unused]] const auto &_den3 = DiFfRG::device::get<8>(P_);
      [[maybe_unused]] const auto &_den4 = DiFfRG::device::get<9>(P_);
      [[maybe_unused]] const auto &_den5 = DiFfRG::device::get<10>(P_);
      [[maybe_unused]] const auto &_den8 = DiFfRG::device::get<11>(P_);
      [[maybe_unused]] const auto &_den9 = DiFfRG::device::get<12>(P_);
      [[maybe_unused]] const auto &_cse1 = DiFfRG::device::get<13>(P_);
      [[maybe_unused]] const auto &_cse4 = DiFfRG::device::get<14>(P_);
      [[maybe_unused]] const auto &_cse6 = DiFfRG::device::get<15>(P_);
      [[maybe_unused]] const auto ntgix2 = ZPi.index(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntgix3 = ZPi.index(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntglk4 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk5 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zpi::nenv) > 0 ? (DiFfRG::zpi::nenv) : 1];
      const double dr_6 = -f0 + p0 - 3.141592653589793 * T;
      const double dr_7 = 1. + sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_8 = -1. - sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_9 = f0 - p0 + 3.141592653589793 * T;
      const double dr_10 = -f0 - p0 - 3.141592653589793 * T;
      const double dr_11 = 1. + sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_12 = f0 + p0 + 3.141592653589793 * T;
      DiFfRG::zpi::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = ntRe(DiFfRG::zpi::tr16(fenv));
      const auto _interp4 = ntRe(DiFfRG::zpi::tr17(fenv));
      const auto _interp5 = ntRe(DiFfRG::zpi::tr19(fenv));
      const auto _interp6 = ntRe(DiFfRG::zpi::tr20(fenv));
      const auto _interp7 = ntRe(DiFfRG::zpi::tr21(fenv));
      const auto _interp8 = ntRe(DiFfRG::zpi::tr22(fenv));
      const auto _interp9 = ntRe(DiFfRG::zpi::tr18(fenv));
      const auto _interp10 = ntRe(DiFfRG::zpi::tr23(fenv));
      const auto _interp11 = ntRe(DiFfRG::zpi::tr25(fenv));
      const auto _interp12 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp13 = ntRe(DiFfRG::zpi::tr27(fenv));
      const auto _interp14 = ntRe(DiFfRG::zpi::tr24(fenv));
      const auto _interp15 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp16 = ntRe(DiFfRG::zpi::tr26(fenv));
      const auto _interp17 = ntRe(DiFfRG::zpi::tr1(fenv));
      const auto _interp20 = ntRe(DiFfRG::zpi::tr3(fenv));
      const auto _interp21 = ntRe(DiFfRG::zpi::tr6(fenv));
      const auto _interp22 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp23 = ntRe(DiFfRG::zpi::tr8(fenv));
      const auto _interp24 = ntRe(DiFfRG::zpi::tr10(fenv));
      const auto _interp25 = ntRe(DiFfRG::zpi::tr12(fenv));
      const auto _interp26 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp27 = ntRe(DiFfRG::zpi::tr14(fenv));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den6 = powr<-1>(_interp12 + d1V + ntglk5 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
      const auto _den7 = powr<-1>(_interp15 + d1V + ntglk4 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
      const auto _den10 = powr<-1>(_interp12 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
      const auto _den11 = powr<-1>(_interp15 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
      const auto _den12 = powr<-1>(powr<2>(_interp22 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den13 = powr<-1>(powr<2>(_interp22 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _den14 = powr<-1>(powr<2>(_interp26 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      // clang-format off
      return 0.01388888888888889 * fma(3., _cse4 * _den1 * _den2 * _den3 * _interp1 * d2V, fma(-6., _cse4 * _den1 * _den2 * _den3 * _interp4 * d2V, fma(3., _cse4 * _den1 * _den4 * _den5 * _interp5 * d2V, fma(-6., _cse4 * _den1 * _den4 * _den5 * _interp7 * d2V, fma(24., _cse1 * _den1 * _den8 * _interp19 * _interp20 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den13 * _den9 * _interp19 * _interp23 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den12 * _den9 * _interp19 * _interp24 * powr<2>(hPhi), fma(-12., _cse1 * _den1 * _den14 * _den9 * _interp19 * _interp27 * powr<2>(hPhi), fma(-12., _cse6 * _den1 * _den2 * _den4 * _den5 * _interp10 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den4 * _den5 * _den6 * _interp11 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den10 * _den2 * _den3 * _interp13 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den4 * _den5 * _den7 * _interp14 * powr<2>(d2V) * rhoPhi, fma(6., _cse6 * _den1 * _den11 * _den2 * _den3 * _interp16 * powr<2>(d2V) * rhoPhi, fma(-12., _cse6 * _den1 * _den2 * _den3 * _den4 * _interp9 * powr<2>(d2V) * rhoPhi, fma(4., _cse4 * _den1 * _den4 * _den5 * _interp6 * d3V * rhoPhi, fma(-8., _cse4 * _den1 * _den4 * _den5 * _interp8 * d3V * rhoPhi, fma(-12., _cse1 * _den1 * _den8 * _interp17 * _interp19 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1 * _den1 * _den13 * _den9 * _interp19 * _interp21 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1 * _den1 * _den14 * _den9 * _interp19 * _interp25 * powr<4>(hPhi) * rhoPhi, 0.)))))))))))))))))));
      // clang-format on
    }
    #elif NT_ZPI_VERDICT == 1   // RePart: real value via complex trace(s), re/im split
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
      [[maybe_unused]] const auto ntglk2 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk3 = ZSig.at(ntgix1);
      [[maybe_unused]] const auto ntglk4 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk5 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zpi::nenv) > 0 ? (DiFfRG::zpi::nenv) : 1];
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
      DiFfRG::zpi::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp18 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp19 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp25 = RF(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp32 = RF(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den2 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk0, _interp2 + d1V));
      const auto _den3 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk1, _interp2 + d1V));
      const auto _den4 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk2, fma(2., d2V * rhoPhi, _interp2 + d1V)));
      const auto _den5 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk3, fma(2., d2V * rhoPhi, _interp2 + d1V)));
      const auto _den8 = powr<-3>(fma(powr<-2>(l1) * powr<2>(_interp18 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den9 = powr<-2>(fma(powr<-2>(l1) * powr<2>(_interp18 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den14 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp25 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den15 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp32 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _interp1 = ntRe(DiFfRG::zpi::tr16(fenv));
      const auto _interp4 = ntRe(DiFfRG::zpi::tr17(fenv));
      const auto _interp5 = ntRe(DiFfRG::zpi::tr19(fenv));
      const auto _interp7 = ntRe(DiFfRG::zpi::tr21(fenv));
      const auto _interp20 = ntIm(DiFfRG::zpi::tr2(fenv));
      const auto _interp21 = ntIm(DiFfRG::zpi::tr4(fenv));
      const auto _interp23 = ntRe(DiFfRG::zpi::tr3(fenv));
      const auto _interp26 = ntIm(DiFfRG::zpi::tr7(fenv));
      const auto _interp28 = ntRe(DiFfRG::zpi::tr8(fenv));
      const auto _interp29 = ntIm(DiFfRG::zpi::tr9(fenv));
      const auto _interp30 = ntRe(DiFfRG::zpi::tr10(fenv));
      const auto _interp33 = ntIm(DiFfRG::zpi::tr13(fenv));
      const auto _interp35 = ntRe(DiFfRG::zpi::tr14(fenv));
      const auto _interp36 = ntIm(DiFfRG::zpi::tr15(fenv));
      const auto _den12 = powr<-1>(powr<2>(_interp32 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den13 = powr<-1>(powr<2>(_interp25 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _cse5_k1 = sqrt(rhoPhi);
      const auto _interp6 = ntRe(DiFfRG::zpi::tr20(fenv));
      const auto _interp8 = ntRe(DiFfRG::zpi::tr22(fenv));
      const auto _interp9 = ntRe(DiFfRG::zpi::tr18(fenv));
      const auto _interp10 = ntRe(DiFfRG::zpi::tr23(fenv));
      const auto _interp11 = ntRe(DiFfRG::zpi::tr25(fenv));
      const auto _interp12 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp13 = ntRe(DiFfRG::zpi::tr27(fenv));
      const auto _interp14 = ntRe(DiFfRG::zpi::tr24(fenv));
      const auto _interp15 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp16 = ntRe(DiFfRG::zpi::tr26(fenv));
      const auto _interp17 = ntIm(DiFfRG::zpi::tr0(fenv));
      const auto _interp22 = ntRe(DiFfRG::zpi::tr1(fenv));
      const auto _interp24 = ntIm(DiFfRG::zpi::tr5(fenv));
      const auto _interp27 = ntRe(DiFfRG::zpi::tr6(fenv));
      const auto _interp31 = ntIm(DiFfRG::zpi::tr11(fenv));
      const auto _interp34 = ntRe(DiFfRG::zpi::tr12(fenv));
      const auto _cse7_k2 = sqrt(powr<3>(rhoPhi));
      // clang-format off
      using _T = decltype(_cse5_k1 + _cse7_k2 + _den1 + _den12 + _den13 + _den14 + _den15 + _den2 + _den3 + _den4 + _den5 + _den8 + _den9 + _interp1 + _interp10 + _interp11 + _interp12 + _interp13 + _interp14 + _interp15 + _interp16 + _interp17 + _interp18 + _interp19 + _interp2 + _interp20 + _interp21 + _interp22 + _interp23 + _interp24 + _interp25 + _interp26 + _interp27 + _interp28 + _interp29 + _interp3 + _interp30 + _interp31 + _interp32 + _interp33 + _interp34 + _interp35 + _interp36 + _interp4 + _interp5 + _interp6 + _interp7 + _interp8 + _interp9 + cos1 + d1V + d2V + d3V + etaPi0 + f0 + hPhi + l1 + ntglk4 + ntglk5 + ntglk6 + ntglk7 + p + p0 + rhoPhi);
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _cse1_k1 = powr<-1>(l1);
        const auto _cse2_k1 = -_interp3;
        const auto _cse3_k1 = _interp2 * etaPi0;
        const auto _cse4_k1 = _cse2_k1 + _cse3_k1;
        // clang-format off
        _acc += 0.04166666666666666 * fma(_cse4_k1, _den1 * _den2 * _den3 * _interp1 * d2V, fma(-2., _cse4_k1 * _den1 * _den2 * _den3 * _interp4 * d2V, fma(_cse4_k1, _den1 * _den4 * _den5 * _interp5 * d2V, fma(-2., _cse4_k1 * _den1 * _den4 * _den5 * _interp7 * d2V, fma(-8., _cse1_k1 * _den1 * _den8 * _interp19 * _interp21 * powr<2>(hPhi), fma(8., _cse1_k1 * _den1 * _den8 * _interp19 * _interp23 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den14 * _den9 * _interp19 * _interp28 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den13 * _den9 * _interp19 * _interp30 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den15 * _den9 * _interp19 * _interp35 * powr<2>(hPhi), fma(4., _cse1_k1 * _den1 * _den12 * _den9 * _interp19 * _interp36 * powr<2>(hPhi), fma(4., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den8 * _interp19 * _interp20 * powr<3>(hPhi), fma(-2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den14 * _den9 * _interp19 * _interp26 * powr<3>(hPhi), fma(-2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den13 * _den9 * _interp19 * _interp29 * powr<3>(hPhi), fma(-2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den15 * _den9 * _interp19 * _interp33 * powr<3>(hPhi), 0.))))))))))))));
        // clang-format on
      }
      { // subkernel 2
        const auto _den6 = powr<-1>(_interp12 + d1V + ntglk5 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
        const auto _den7 = powr<-1>(_interp15 + d1V + ntglk4 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
        const auto _den10 = powr<-1>(_interp12 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
        const auto _den11 = powr<-1>(_interp15 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
        const auto _cse1_k2 = powr<-1>(l1);
        const auto _cse2_k2 = -_interp2 * etaPi0;
        const auto _cse3_k2 = _cse2_k2 + _interp3;
        const auto _cse4_k2 = -_interp3;
        const auto _cse5_k2 = _interp2 * etaPi0;
        const auto _cse6_k2 = _cse4_k2 + _cse5_k2;
        // clang-format off
        _acc += 0.01388888888888889 * fma(-6., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den8 * _interp17 * _interp19 * powr<5>(hPhi), fma(3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den14 * _den9 * _interp19 * _interp24 * powr<5>(hPhi), fma(3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den15 * _den9 * _interp19 * _interp31 * powr<5>(hPhi), fma(-12., _cse3_k2 * _den1 * _den2 * _den4 * _den5 * _interp10 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den4 * _den5 * _den6 * _interp11 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den10 * _den2 * _den3 * _interp13 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den4 * _den5 * _den7 * _interp14 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den11 * _den2 * _den3 * _interp16 * powr<2>(d2V) * rhoPhi, fma(-12., _cse3_k2 * _den1 * _den2 * _den3 * _den4 * _interp9 * powr<2>(d2V) * rhoPhi, fma(4., _cse6_k2 * _den1 * _den4 * _den5 * _interp6 * d3V * rhoPhi, fma(-8., _cse6_k2 * _den1 * _den4 * _den5 * _interp8 * d3V * rhoPhi, fma(-12., _cse1_k2 * _den1 * _den8 * _interp19 * _interp22 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1_k2 * _den1 * _den14 * _den9 * _interp19 * _interp27 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1_k2 * _den1 * _den15 * _den9 * _interp19 * _interp34 * powr<4>(hPhi) * rhoPhi, 0.))))))))))))));
        // clang-format on
      }
      return _acc;
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
      [[maybe_unused]] const auto ntglk2 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk3 = ZSig.at(ntgix1);
      const double dr_0 = -f0 - 3.141592653589793 * T;
      const double dr_1 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const double dr_2 = 0.7071067811865475;
      const double dr_3 = hPhi;
      const double dr_4 = sqrt(rhoPhi);
      const double dr_5 = f0 + 3.141592653589793 * T;
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp18 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp19 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _den2 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk0, _interp2 + d1V));
      const auto _den3 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk1, _interp2 + d1V));
      const auto _den4 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk2, fma(2., d2V * rhoPhi, _interp2 + d1V)));
      const auto _den5 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk3, fma(2., d2V * rhoPhi, _interp2 + d1V)));
      const auto _den8 = powr<-3>(fma(powr<-2>(l1) * powr<2>(_interp18 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den9 = powr<-2>(fma(powr<-2>(l1) * powr<2>(_interp18 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _cse5_k1 = sqrt(rhoPhi);
      const auto _cse7_k2 = sqrt(powr<3>(rhoPhi));
      const auto _cse1_k1 = powr<-1>(l1);
      const auto _cse2_k1 = -_interp3;
      const auto _cse3_k1 = _interp2 * etaPi0;
      const auto _cse4_k1 = _cse2_k1 + _cse3_k1;
      const auto _cse1_k2 = powr<-1>(l1);
      const auto _cse2_k2 = -_interp2 * etaPi0;
      const auto _cse3_k2 = _cse2_k2 + _interp3;
      const auto _cse4_k2 = -_interp3;
      const auto _cse5_k2 = _interp2 * etaPi0;
      const auto _cse6_k2 = _cse4_k2 + _cse5_k2;
      return DiFfRG::device::make_tuple(dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, _interp2, _interp3, _interp18, _interp19, _den2, _den3, _den4, _den5, _den8, _den9, _cse5_k1, _cse7_k2, _cse1_k1, _cse4_k1, _cse1_k2, _cse3_k2, _cse6_k2);
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
      [[maybe_unused]] const auto &_interp2 = DiFfRG::device::get<6>(P_);
      [[maybe_unused]] const auto &_interp3 = DiFfRG::device::get<7>(P_);
      [[maybe_unused]] const auto &_interp18 = DiFfRG::device::get<8>(P_);
      [[maybe_unused]] const auto &_interp19 = DiFfRG::device::get<9>(P_);
      [[maybe_unused]] const auto &_den2 = DiFfRG::device::get<10>(P_);
      [[maybe_unused]] const auto &_den3 = DiFfRG::device::get<11>(P_);
      [[maybe_unused]] const auto &_den4 = DiFfRG::device::get<12>(P_);
      [[maybe_unused]] const auto &_den5 = DiFfRG::device::get<13>(P_);
      [[maybe_unused]] const auto &_den8 = DiFfRG::device::get<14>(P_);
      [[maybe_unused]] const auto &_den9 = DiFfRG::device::get<15>(P_);
      [[maybe_unused]] const auto &_cse5_k1 = DiFfRG::device::get<16>(P_);
      [[maybe_unused]] const auto &_cse7_k2 = DiFfRG::device::get<17>(P_);
      [[maybe_unused]] const auto &_cse1_k1 = DiFfRG::device::get<18>(P_);
      [[maybe_unused]] const auto &_cse4_k1 = DiFfRG::device::get<19>(P_);
      [[maybe_unused]] const auto &_cse1_k2 = DiFfRG::device::get<20>(P_);
      [[maybe_unused]] const auto &_cse3_k2 = DiFfRG::device::get<21>(P_);
      [[maybe_unused]] const auto &_cse6_k2 = DiFfRG::device::get<22>(P_);
      [[maybe_unused]] const auto ntgix2 = ZPi.index(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntgix3 = ZPi.index(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntglk4 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk5 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zpi::nenv) > 0 ? (DiFfRG::zpi::nenv) : 1];
      const double dr_6 = -f0 + p0 - 3.141592653589793 * T;
      const double dr_7 = 1. + sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_8 = -1. - sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_9 = f0 - p0 + 3.141592653589793 * T;
      const double dr_10 = -f0 - p0 - 3.141592653589793 * T;
      const double dr_11 = 1. + sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_12 = f0 + p0 + 3.141592653589793 * T;
      DiFfRG::zpi::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp25 = RF(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp32 = RF(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den14 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp25 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den15 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp32 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _interp1 = ntRe(DiFfRG::zpi::tr16(fenv));
      const auto _interp4 = ntRe(DiFfRG::zpi::tr17(fenv));
      const auto _interp5 = ntRe(DiFfRG::zpi::tr19(fenv));
      const auto _interp7 = ntRe(DiFfRG::zpi::tr21(fenv));
      const auto _interp20 = ntIm(DiFfRG::zpi::tr2(fenv));
      const auto _interp21 = ntIm(DiFfRG::zpi::tr4(fenv));
      const auto _interp23 = ntRe(DiFfRG::zpi::tr3(fenv));
      const auto _interp26 = ntIm(DiFfRG::zpi::tr7(fenv));
      const auto _interp28 = ntRe(DiFfRG::zpi::tr8(fenv));
      const auto _interp29 = ntIm(DiFfRG::zpi::tr9(fenv));
      const auto _interp30 = ntRe(DiFfRG::zpi::tr10(fenv));
      const auto _interp33 = ntIm(DiFfRG::zpi::tr13(fenv));
      const auto _interp35 = ntRe(DiFfRG::zpi::tr14(fenv));
      const auto _interp36 = ntIm(DiFfRG::zpi::tr15(fenv));
      const auto _den12 = powr<-1>(powr<2>(_interp32 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den13 = powr<-1>(powr<2>(_interp25 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _interp6 = ntRe(DiFfRG::zpi::tr20(fenv));
      const auto _interp8 = ntRe(DiFfRG::zpi::tr22(fenv));
      const auto _interp9 = ntRe(DiFfRG::zpi::tr18(fenv));
      const auto _interp10 = ntRe(DiFfRG::zpi::tr23(fenv));
      const auto _interp11 = ntRe(DiFfRG::zpi::tr25(fenv));
      const auto _interp12 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp13 = ntRe(DiFfRG::zpi::tr27(fenv));
      const auto _interp14 = ntRe(DiFfRG::zpi::tr24(fenv));
      const auto _interp15 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp16 = ntRe(DiFfRG::zpi::tr26(fenv));
      const auto _interp17 = ntIm(DiFfRG::zpi::tr0(fenv));
      const auto _interp22 = ntRe(DiFfRG::zpi::tr1(fenv));
      const auto _interp24 = ntIm(DiFfRG::zpi::tr5(fenv));
      const auto _interp27 = ntRe(DiFfRG::zpi::tr6(fenv));
      const auto _interp31 = ntIm(DiFfRG::zpi::tr11(fenv));
      const auto _interp34 = ntRe(DiFfRG::zpi::tr12(fenv));
      // clang-format off
      using _T = decltype(_cse5_k1 + _cse7_k2 + _den1 + _den12 + _den13 + _den14 + _den15 + _den2 + _den3 + _den4 + _den5 + _den8 + _den9 + _interp1 + _interp10 + _interp11 + _interp12 + _interp13 + _interp14 + _interp15 + _interp16 + _interp17 + _interp18 + _interp19 + _interp2 + _interp20 + _interp21 + _interp22 + _interp23 + _interp24 + _interp25 + _interp26 + _interp27 + _interp28 + _interp29 + _interp3 + _interp30 + _interp31 + _interp32 + _interp33 + _interp34 + _interp35 + _interp36 + _interp4 + _interp5 + _interp6 + _interp7 + _interp8 + _interp9 + cos1 + d1V + d2V + d3V + etaPi0 + f0 + hPhi + l1 + ntglk4 + ntglk5 + ntglk6 + ntglk7 + p + p0 + rhoPhi);
      // clang-format on
      _T _acc{};
      { // subkernel 1
      // clang-format off
      _acc += 0.04166666666666666 * fma(_cse4_k1, _den1 * _den2 * _den3 * _interp1 * d2V, fma(-2., _cse4_k1 * _den1 * _den2 * _den3 * _interp4 * d2V, fma(_cse4_k1, _den1 * _den4 * _den5 * _interp5 * d2V, fma(-2., _cse4_k1 * _den1 * _den4 * _den5 * _interp7 * d2V, fma(-8., _cse1_k1 * _den1 * _den8 * _interp19 * _interp21 * powr<2>(hPhi), fma(8., _cse1_k1 * _den1 * _den8 * _interp19 * _interp23 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den14 * _den9 * _interp19 * _interp28 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den13 * _den9 * _interp19 * _interp30 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den15 * _den9 * _interp19 * _interp35 * powr<2>(hPhi), fma(4., _cse1_k1 * _den1 * _den12 * _den9 * _interp19 * _interp36 * powr<2>(hPhi), fma(4., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den8 * _interp19 * _interp20 * powr<3>(hPhi), fma(-2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den14 * _den9 * _interp19 * _interp26 * powr<3>(hPhi), fma(-2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den13 * _den9 * _interp19 * _interp29 * powr<3>(hPhi), fma(-2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den15 * _den9 * _interp19 * _interp33 * powr<3>(hPhi), 0.))))))))))))));
      // clang-format on
      }
      { // subkernel 2
        const auto _den6 = powr<-1>(_interp12 + d1V + ntglk5 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
        const auto _den7 = powr<-1>(_interp15 + d1V + ntglk4 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
        const auto _den10 = powr<-1>(_interp12 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
        const auto _den11 = powr<-1>(_interp15 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
        // clang-format off
        _acc += 0.01388888888888889 * fma(-6., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den8 * _interp17 * _interp19 * powr<5>(hPhi), fma(3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den14 * _den9 * _interp19 * _interp24 * powr<5>(hPhi), fma(3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den15 * _den9 * _interp19 * _interp31 * powr<5>(hPhi), fma(-12., _cse3_k2 * _den1 * _den2 * _den4 * _den5 * _interp10 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den4 * _den5 * _den6 * _interp11 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den10 * _den2 * _den3 * _interp13 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den4 * _den5 * _den7 * _interp14 * powr<2>(d2V) * rhoPhi, fma(6., _cse3_k2 * _den1 * _den11 * _den2 * _den3 * _interp16 * powr<2>(d2V) * rhoPhi, fma(-12., _cse3_k2 * _den1 * _den2 * _den3 * _den4 * _interp9 * powr<2>(d2V) * rhoPhi, fma(4., _cse6_k2 * _den1 * _den4 * _den5 * _interp6 * d3V * rhoPhi, fma(-8., _cse6_k2 * _den1 * _den4 * _den5 * _interp8 * d3V * rhoPhi, fma(-12., _cse1_k2 * _den1 * _den8 * _interp19 * _interp22 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1_k2 * _den1 * _den14 * _den9 * _interp19 * _interp27 * powr<4>(hPhi) * rhoPhi, fma(6., _cse1_k2 * _den1 * _den15 * _den9 * _interp19 * _interp34 * powr<4>(hPhi) * rhoPhi, 0.))))))))))))));
        // clang-format on
      }
      return _acc;
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
      [[maybe_unused]] const auto ntglk2 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk3 = ZSig.at(ntgix1);
      [[maybe_unused]] const auto ntglk4 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk5 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zpi::nenv) > 0 ? (DiFfRG::zpi::nenv) : 1];
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
      DiFfRG::zpi::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp5 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp6 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp7 = RF(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp8 = RF(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den2 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk0, _interp1 + d1V));
      const auto _den3 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk1, _interp1 + d1V));
      const auto _den4 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk2, fma(2., d2V * rhoPhi, _interp1 + d1V)));
      const auto _den5 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk3, fma(2., d2V * rhoPhi, _interp1 + d1V)));
      const auto _den8 = powr<-3>(fma(powr<-2>(l1) * powr<2>(_interp5 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den9 = powr<-2>(fma(powr<-2>(l1) * powr<2>(_interp5 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den14 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp7 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den15 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp8 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _den12 = powr<-1>(powr<2>(_interp8 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den13 = powr<-1>(powr<2>(_interp7 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _cse5_k1 = sqrt(rhoPhi);
      const auto _interp3 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp4 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const auto _cse7_k2 = sqrt(powr<3>(rhoPhi));
      // clang-format off
      using _T = decltype(complex<double>(0.,0.04166666666666666) + _cse5_k1 + _cse7_k2 + _den1 + _den12 + _den13 + _den14 + _den15 + _den2 + _den3 + _den4 + _den5 + _den8 + _den9 + DiFfRG::zpi::tr0(fenv) + DiFfRG::zpi::tr10(fenv) + DiFfRG::zpi::tr11(fenv) + DiFfRG::zpi::tr12(fenv) + DiFfRG::zpi::tr13(fenv) + DiFfRG::zpi::tr14(fenv) + DiFfRG::zpi::tr15(fenv) + DiFfRG::zpi::tr16(fenv) + DiFfRG::zpi::tr17(fenv) + DiFfRG::zpi::tr18(fenv) + DiFfRG::zpi::tr19(fenv) + DiFfRG::zpi::tr1(fenv) + DiFfRG::zpi::tr20(fenv) + DiFfRG::zpi::tr21(fenv) + DiFfRG::zpi::tr22(fenv) + DiFfRG::zpi::tr23(fenv) + DiFfRG::zpi::tr24(fenv) + DiFfRG::zpi::tr25(fenv) + DiFfRG::zpi::tr26(fenv) + DiFfRG::zpi::tr27(fenv) + DiFfRG::zpi::tr2(fenv) + DiFfRG::zpi::tr3(fenv) + DiFfRG::zpi::tr4(fenv) + DiFfRG::zpi::tr5(fenv) + DiFfRG::zpi::tr6(fenv) + DiFfRG::zpi::tr7(fenv) + DiFfRG::zpi::tr8(fenv) + DiFfRG::zpi::tr9(fenv) + _interp1 + _interp2 + _interp3 + _interp4 + _interp5 + _interp6 + _interp7 + _interp8 + cos1 + d1V + d2V + d3V + etaPi0 + f0 + hPhi + l1 + ntglk4 + ntglk5 + ntglk6 + ntglk7 + p + p0 + rhoPhi);
      // clang-format on
      _T _acc{};
      { // subkernel 1
        const auto _cse1_k1 = powr<-1>(l1);
        const auto _cse2_k1 = -_interp2;
        const auto _cse3_k1 = _interp1 * etaPi0;
        const auto _cse4_k1 = _cse2_k1 + _cse3_k1;
        // clang-format off
        _acc += complex<double>(0.,0.04166666666666666) * fma(complex<double>(0.,-1.), _cse4_k1 * _den1 * _den2 * _den3 * DiFfRG::zpi::tr16(fenv) * d2V, fma(complex<double>(0.,2.), _cse4_k1 * _den1 * _den2 * _den3 * DiFfRG::zpi::tr17(fenv) * d2V, fma(complex<double>(0.,-1.), _cse4_k1 * _den1 * _den4 * _den5 * DiFfRG::zpi::tr19(fenv) * d2V, fma(complex<double>(0.,2.), _cse4_k1 * _den1 * _den4 * _den5 * DiFfRG::zpi::tr21(fenv) * d2V, fma(complex<double>(0.,4.), _cse1_k1 * _den1 * _den13 * _den9 * DiFfRG::zpi::tr10(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,4.), _cse1_k1 * _den1 * _den15 * _den9 * DiFfRG::zpi::tr14(fenv) * _interp6 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den12 * _den9 * DiFfRG::zpi::tr15(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,-8.), _cse1_k1 * _den1 * _den8 * DiFfRG::zpi::tr3(fenv) * _interp6 * powr<2>(hPhi), fma(8., _cse1_k1 * _den1 * _den8 * DiFfRG::zpi::tr4(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,4.), _cse1_k1 * _den1 * _den14 * _den9 * DiFfRG::zpi::tr8(fenv) * _interp6 * powr<2>(hPhi), fma(2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den15 * _den9 * DiFfRG::zpi::tr13(fenv) * _interp6 * powr<3>(hPhi), fma(-4., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den8 * DiFfRG::zpi::tr2(fenv) * _interp6 * powr<3>(hPhi), fma(2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den14 * _den9 * DiFfRG::zpi::tr7(fenv) * _interp6 * powr<3>(hPhi), fma(2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den13 * _den9 * DiFfRG::zpi::tr9(fenv) * _interp6 * powr<3>(hPhi), 0.))))))))))))));
        // clang-format on
      }
      { // subkernel 2
        const auto _den6 = powr<-1>(_interp3 + d1V + ntglk5 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
        const auto _den7 = powr<-1>(_interp4 + d1V + ntglk4 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
        const auto _den10 = powr<-1>(_interp3 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
        const auto _den11 = powr<-1>(_interp4 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
        const auto _cse1_k2 = powr<-1>(l1);
        const auto _cse2_k2 = -_interp1 * etaPi0;
        const auto _cse3_k2 = _cse2_k2 + _interp2;
        const auto _cse4_k2 = -_interp2;
        const auto _cse5_k2 = _interp1 * etaPi0;
        const auto _cse6_k2 = _cse4_k2 + _cse5_k2;
        // clang-format off
        _acc += complex<double>(0.,0.01388888888888889) * fma(6., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den8 * DiFfRG::zpi::tr0(fenv) * _interp6 * powr<5>(hPhi), fma(-3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den15 * _den9 * DiFfRG::zpi::tr11(fenv) * _interp6 * powr<5>(hPhi), fma(-3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den14 * _den9 * DiFfRG::zpi::tr5(fenv) * _interp6 * powr<5>(hPhi), fma(complex<double>(0.,12.), _cse3_k2 * _den1 * _den2 * _den3 * _den4 * DiFfRG::zpi::tr18(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,12.), _cse3_k2 * _den1 * _den2 * _den4 * _den5 * DiFfRG::zpi::tr23(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den4 * _den5 * _den7 * DiFfRG::zpi::tr24(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den4 * _den5 * _den6 * DiFfRG::zpi::tr25(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den11 * _den2 * _den3 * DiFfRG::zpi::tr26(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den10 * _den2 * _den3 * DiFfRG::zpi::tr27(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-4.), _cse6_k2 * _den1 * _den4 * _den5 * DiFfRG::zpi::tr20(fenv) * d3V * rhoPhi, fma(complex<double>(0.,8.), _cse6_k2 * _den1 * _den4 * _den5 * DiFfRG::zpi::tr22(fenv) * d3V * rhoPhi, fma(complex<double>(0.,-6.), _cse1_k2 * _den1 * _den15 * _den9 * DiFfRG::zpi::tr12(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,12.), _cse1_k2 * _den1 * _den8 * DiFfRG::zpi::tr1(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-6.), _cse1_k2 * _den1 * _den14 * _den9 * DiFfRG::zpi::tr6(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, 0.))))))))))))));
        // clang-format on
      }
      return _acc;
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
      [[maybe_unused]] const auto ntglk2 = ZSig.at(ntgix0);
      [[maybe_unused]] const auto ntglk3 = ZSig.at(ntgix1);
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
      const auto _den2 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk0, _interp1 + d1V));
      const auto _den3 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk1, _interp1 + d1V));
      const auto _den4 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk2, fma(2., d2V * rhoPhi, _interp1 + d1V)));
      const auto _den5 = powr<-1>(fma(powr<2>(f0) + powr<2>(l1), ntglk3, fma(2., d2V * rhoPhi, _interp1 + d1V)));
      const auto _den8 = powr<-3>(fma(powr<-2>(l1) * powr<2>(_interp5 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _den9 = powr<-2>(fma(powr<-2>(l1) * powr<2>(_interp5 + l1), powr<2>(l1), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + 3.141592653589793 * T))));
      const auto _cse5_k1 = sqrt(rhoPhi);
      const auto _cse7_k2 = sqrt(powr<3>(rhoPhi));
      const auto _cse1_k1 = powr<-1>(l1);
      const auto _cse2_k1 = -_interp2;
      const auto _cse3_k1 = _interp1 * etaPi0;
      const auto _cse4_k1 = _cse2_k1 + _cse3_k1;
      const auto _cse1_k2 = powr<-1>(l1);
      const auto _cse2_k2 = -_interp1 * etaPi0;
      const auto _cse3_k2 = _cse2_k2 + _interp2;
      const auto _cse4_k2 = -_interp2;
      const auto _cse5_k2 = _interp1 * etaPi0;
      const auto _cse6_k2 = _cse4_k2 + _cse5_k2;
      return DiFfRG::device::make_tuple(dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, _interp1, _interp2, _interp5, _interp6, _den2, _den3, _den4, _den5, _den8, _den9, _cse5_k1, _cse7_k2, _cse1_k1, _cse4_k1, _cse1_k2, _cse3_k2, _cse6_k2);
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
      [[maybe_unused]] const auto &_interp1 = DiFfRG::device::get<6>(P_);
      [[maybe_unused]] const auto &_interp2 = DiFfRG::device::get<7>(P_);
      [[maybe_unused]] const auto &_interp5 = DiFfRG::device::get<8>(P_);
      [[maybe_unused]] const auto &_interp6 = DiFfRG::device::get<9>(P_);
      [[maybe_unused]] const auto &_den2 = DiFfRG::device::get<10>(P_);
      [[maybe_unused]] const auto &_den3 = DiFfRG::device::get<11>(P_);
      [[maybe_unused]] const auto &_den4 = DiFfRG::device::get<12>(P_);
      [[maybe_unused]] const auto &_den5 = DiFfRG::device::get<13>(P_);
      [[maybe_unused]] const auto &_den8 = DiFfRG::device::get<14>(P_);
      [[maybe_unused]] const auto &_den9 = DiFfRG::device::get<15>(P_);
      [[maybe_unused]] const auto &_cse5_k1 = DiFfRG::device::get<16>(P_);
      [[maybe_unused]] const auto &_cse7_k2 = DiFfRG::device::get<17>(P_);
      [[maybe_unused]] const auto &_cse1_k1 = DiFfRG::device::get<18>(P_);
      [[maybe_unused]] const auto &_cse4_k1 = DiFfRG::device::get<19>(P_);
      [[maybe_unused]] const auto &_cse1_k2 = DiFfRG::device::get<20>(P_);
      [[maybe_unused]] const auto &_cse3_k2 = DiFfRG::device::get<21>(P_);
      [[maybe_unused]] const auto &_cse6_k2 = DiFfRG::device::get<22>(P_);
      [[maybe_unused]] const auto ntgix2 = ZPi.index(-f0 - p0, sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntgix3 = ZPi.index(-f0 + p0, sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)));
      [[maybe_unused]] const auto ntglk4 = ZPi.at(ntgix2);
      [[maybe_unused]] const auto ntglk5 = ZPi.at(ntgix3);
      [[maybe_unused]] const auto ntglk6 = ZSig.at(ntgix2);
      [[maybe_unused]] const auto ntglk7 = ZSig.at(ntgix3);
      double fenv[(DiFfRG::zpi::nenv) > 0 ? (DiFfRG::zpi::nenv) : 1];
      const double dr_6 = -f0 + p0 - 3.141592653589793 * T;
      const double dr_7 = 1. + sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_8 = -1. - sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_9 = f0 - p0 + 3.141592653589793 * T;
      const double dr_10 = -f0 - p0 - 3.141592653589793 * T;
      const double dr_11 = 1. + sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) * RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      const double dr_12 = f0 + p0 + 3.141592653589793 * T;
      DiFfRG::zpi::fill(fenv, l1, cos1, f0, p0, p, T, dr_0, dr_1, dr_2, dr_3, dr_4, dr_5, dr_6, dr_7, dr_8, dr_9, dr_10, dr_11, dr_12);
      const auto _interp7 = RF(powr<2>(k), fma(-2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _interp8 = RF(powr<2>(k), fma(2., cos1 * l1 * p, powr<2>(l1) + powr<2>(p)));
      const auto _den1 = powr<-1>(powr<2>(p) + powr<2>(p0));
      const auto _den14 = powr<-1>(fma(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp7 * sqrt(powr<-1>(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 - p0 + 3.141592653589793 * T))));
      const auto _den15 = powr<-1>(fma(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p), powr<2>(1. + _interp8 * sqrt(powr<-1>(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p)))), fma(0.5, powr<2>(hPhi) * rhoPhi, powr<2>(f0 + p0 + 3.141592653589793 * T))));
      const auto _den12 = powr<-1>(powr<2>(_interp8 + sqrt(powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + p0 + 3.141592653589793 * T));
      const auto _den13 = powr<-1>(powr<2>(_interp7 + sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p))) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 - p0 + 3.141592653589793 * T));
      const auto _interp3 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp4 = RB(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
      // clang-format off
      using _T = decltype(complex<double>(0.,0.04166666666666666) + _cse5_k1 + _cse7_k2 + _den1 + _den12 + _den13 + _den14 + _den15 + _den2 + _den3 + _den4 + _den5 + _den8 + _den9 + DiFfRG::zpi::tr0(fenv) + DiFfRG::zpi::tr10(fenv) + DiFfRG::zpi::tr11(fenv) + DiFfRG::zpi::tr12(fenv) + DiFfRG::zpi::tr13(fenv) + DiFfRG::zpi::tr14(fenv) + DiFfRG::zpi::tr15(fenv) + DiFfRG::zpi::tr16(fenv) + DiFfRG::zpi::tr17(fenv) + DiFfRG::zpi::tr18(fenv) + DiFfRG::zpi::tr19(fenv) + DiFfRG::zpi::tr1(fenv) + DiFfRG::zpi::tr20(fenv) + DiFfRG::zpi::tr21(fenv) + DiFfRG::zpi::tr22(fenv) + DiFfRG::zpi::tr23(fenv) + DiFfRG::zpi::tr24(fenv) + DiFfRG::zpi::tr25(fenv) + DiFfRG::zpi::tr26(fenv) + DiFfRG::zpi::tr27(fenv) + DiFfRG::zpi::tr2(fenv) + DiFfRG::zpi::tr3(fenv) + DiFfRG::zpi::tr4(fenv) + DiFfRG::zpi::tr5(fenv) + DiFfRG::zpi::tr6(fenv) + DiFfRG::zpi::tr7(fenv) + DiFfRG::zpi::tr8(fenv) + DiFfRG::zpi::tr9(fenv) + _interp1 + _interp2 + _interp3 + _interp4 + _interp5 + _interp6 + _interp7 + _interp8 + cos1 + d1V + d2V + d3V + etaPi0 + f0 + hPhi + l1 + ntglk4 + ntglk5 + ntglk6 + ntglk7 + p + p0 + rhoPhi);
      // clang-format on
      _T _acc{};
      { // subkernel 1
      // clang-format off
      _acc += complex<double>(0.,0.04166666666666666) * fma(complex<double>(0.,-1.), _cse4_k1 * _den1 * _den2 * _den3 * DiFfRG::zpi::tr16(fenv) * d2V, fma(complex<double>(0.,2.), _cse4_k1 * _den1 * _den2 * _den3 * DiFfRG::zpi::tr17(fenv) * d2V, fma(complex<double>(0.,-1.), _cse4_k1 * _den1 * _den4 * _den5 * DiFfRG::zpi::tr19(fenv) * d2V, fma(complex<double>(0.,2.), _cse4_k1 * _den1 * _den4 * _den5 * DiFfRG::zpi::tr21(fenv) * d2V, fma(complex<double>(0.,4.), _cse1_k1 * _den1 * _den13 * _den9 * DiFfRG::zpi::tr10(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,4.), _cse1_k1 * _den1 * _den15 * _den9 * DiFfRG::zpi::tr14(fenv) * _interp6 * powr<2>(hPhi), fma(-4., _cse1_k1 * _den1 * _den12 * _den9 * DiFfRG::zpi::tr15(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,-8.), _cse1_k1 * _den1 * _den8 * DiFfRG::zpi::tr3(fenv) * _interp6 * powr<2>(hPhi), fma(8., _cse1_k1 * _den1 * _den8 * DiFfRG::zpi::tr4(fenv) * _interp6 * powr<2>(hPhi), fma(complex<double>(0.,4.), _cse1_k1 * _den1 * _den14 * _den9 * DiFfRG::zpi::tr8(fenv) * _interp6 * powr<2>(hPhi), fma(2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den15 * _den9 * DiFfRG::zpi::tr13(fenv) * _interp6 * powr<3>(hPhi), fma(-4., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den8 * DiFfRG::zpi::tr2(fenv) * _interp6 * powr<3>(hPhi), fma(2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den14 * _den9 * DiFfRG::zpi::tr7(fenv) * _interp6 * powr<3>(hPhi), fma(2., 1.414213562373095 * _cse1_k1 * _cse5_k1 * _den1 * _den13 * _den9 * DiFfRG::zpi::tr9(fenv) * _interp6 * powr<3>(hPhi), 0.))))))))))))));
      // clang-format on
      }
      { // subkernel 2
        const auto _den6 = powr<-1>(_interp3 + d1V + ntglk5 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)));
        const auto _den7 = powr<-1>(_interp4 + d1V + ntglk4 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)));
        const auto _den10 = powr<-1>(_interp3 + d1V + ntglk7 * (powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 - p0)) + 2. * d2V * rhoPhi);
        const auto _den11 = powr<-1>(_interp4 + d1V + ntglk6 * (powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(f0 + p0)) + 2. * d2V * rhoPhi);
        // clang-format off
        _acc += complex<double>(0.,0.01388888888888889) * fma(6., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den8 * DiFfRG::zpi::tr0(fenv) * _interp6 * powr<5>(hPhi), fma(-3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den15 * _den9 * DiFfRG::zpi::tr11(fenv) * _interp6 * powr<5>(hPhi), fma(-3., 1.414213562373095 * _cse1_k2 * _cse7_k2 * _den1 * _den14 * _den9 * DiFfRG::zpi::tr5(fenv) * _interp6 * powr<5>(hPhi), fma(complex<double>(0.,12.), _cse3_k2 * _den1 * _den2 * _den3 * _den4 * DiFfRG::zpi::tr18(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,12.), _cse3_k2 * _den1 * _den2 * _den4 * _den5 * DiFfRG::zpi::tr23(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den4 * _den5 * _den7 * DiFfRG::zpi::tr24(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den4 * _den5 * _den6 * DiFfRG::zpi::tr25(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den11 * _den2 * _den3 * DiFfRG::zpi::tr26(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-6.), _cse3_k2 * _den1 * _den10 * _den2 * _den3 * DiFfRG::zpi::tr27(fenv) * powr<2>(d2V) * rhoPhi, fma(complex<double>(0.,-4.), _cse6_k2 * _den1 * _den4 * _den5 * DiFfRG::zpi::tr20(fenv) * d3V * rhoPhi, fma(complex<double>(0.,8.), _cse6_k2 * _den1 * _den4 * _den5 * DiFfRG::zpi::tr22(fenv) * d3V * rhoPhi, fma(complex<double>(0.,-6.), _cse1_k2 * _den1 * _den15 * _den9 * DiFfRG::zpi::tr12(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,12.), _cse1_k2 * _den1 * _den8 * DiFfRG::zpi::tr1(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, fma(complex<double>(0.,-6.), _cse1_k2 * _den1 * _den14 * _den9 * DiFfRG::zpi::tr6(fenv) * _interp6 * powr<4>(hPhi) * rhoPhi, 0.))))))))))))));
        // clang-format on
      }
      return _acc;
    }
    #endif

    static KOKKOS_INLINE_FUNCTION auto constant(const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      const auto _interp1 = ZPi(p0, p);
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
using DiFfRG::ZPi_kernel;