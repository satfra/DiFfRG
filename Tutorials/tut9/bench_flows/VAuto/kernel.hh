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
  class VAuto_kernel
  {
    public:
    using Regulator = REG;

    static constexpr bool has_pre = true;

    #if NT_VAUTO_VERDICT == 2   // Pure: the Complex -> Re projection is exact
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& f0, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      double fenv[(DiFfRG::vauto::nenv) > 0 ? (DiFfRG::vauto::nenv) : 1];
      const double dr_0 = 0.7071067811865475;
      const double dr_1 = hPhi;
      const double dr_2 = sqrt(rhoPhi);
      const double dr_3 = f0 + 3.141592653589793 * T;
      const double dr_4 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      DiFfRG::vauto::fill(fenv, l1, f0, T, dr_0, dr_1, dr_2, dr_3, dr_4);
      const auto _interp1 = ntRe(DiFfRG::vauto::tr1(fenv));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZPi(-f0, l1);
      const auto _interp5 = ntRe(DiFfRG::vauto::tr2(fenv));
      const auto _interp6 = ZSig(-f0, l1);
      const auto _den1 = powr<-1>(_interp2 + d1V + _interp4 * (powr<2>(f0) + powr<2>(l1)));
      const auto _den2 = powr<-1>(_interp2 + d1V + _interp6 * (powr<2>(f0) + powr<2>(l1)) + 2. * d2V * rhoPhi);
      const auto _cse1 = -_interp2 * etaPi0;
      const auto _cse2 = _cse1 + _interp3;
      return 0.5 * fma(_cse2, _den1 * _interp1, fma(_cse2, _den2 * _interp5, 0.));
    }
    static constexpr std::array<bool, 12> pre_reads{true, true, true, true, true, true, true, false, false, false, true, true};
    static KOKKOS_INLINE_FUNCTION auto pre(const double& l1, const double& f0, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      const double dr_0 = 0.7071067811865475;
      const double dr_1 = hPhi;
      const double dr_2 = sqrt(rhoPhi);
      const double dr_3 = f0 + 3.141592653589793 * T;
      const double dr_4 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ZPi(-f0, l1);
      const auto _interp6 = ZSig(-f0, l1);
      const auto _cse1 = -_interp2 * etaPi0;
      const auto _cse2 = _cse1 + _interp3;
      return DiFfRG::device::make_tuple(dr_0, dr_1, dr_2, dr_3, dr_4, _interp2, _interp4, _interp6, _cse2);
    }
    static KOKKOS_INLINE_FUNCTION auto post(const auto &P_, const double& l1, const double& f0, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto &dr_0 = DiFfRG::device::get<0>(P_);
      [[maybe_unused]] const auto &dr_1 = DiFfRG::device::get<1>(P_);
      [[maybe_unused]] const auto &dr_2 = DiFfRG::device::get<2>(P_);
      [[maybe_unused]] const auto &dr_3 = DiFfRG::device::get<3>(P_);
      [[maybe_unused]] const auto &dr_4 = DiFfRG::device::get<4>(P_);
      [[maybe_unused]] const auto &_interp2 = DiFfRG::device::get<5>(P_);
      [[maybe_unused]] const auto &_interp4 = DiFfRG::device::get<6>(P_);
      [[maybe_unused]] const auto &_interp6 = DiFfRG::device::get<7>(P_);
      [[maybe_unused]] const auto &_cse2 = DiFfRG::device::get<8>(P_);
      double fenv[(DiFfRG::vauto::nenv) > 0 ? (DiFfRG::vauto::nenv) : 1];
      DiFfRG::vauto::fill(fenv, l1, f0, T, dr_0, dr_1, dr_2, dr_3, dr_4);
      const auto _interp1 = ntRe(DiFfRG::vauto::tr1(fenv));
      const auto _interp5 = ntRe(DiFfRG::vauto::tr2(fenv));
      const auto _den1 = powr<-1>(_interp2 + d1V + _interp4 * (powr<2>(f0) + powr<2>(l1)));
      const auto _den2 = powr<-1>(_interp2 + d1V + _interp6 * (powr<2>(f0) + powr<2>(l1)) + 2. * d2V * rhoPhi);
      return 0.5 * fma(_cse2, _den1 * _interp1, fma(_cse2, _den2 * _interp5, 0.));
    }
    #elif NT_VAUTO_VERDICT == 1   // RePart: real value via complex trace(s), re/im split
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& f0, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      double fenv[(DiFfRG::vauto::nenv) > 0 ? (DiFfRG::vauto::nenv) : 1];
      const double dr_0 = 0.7071067811865475;
      const double dr_1 = hPhi;
      const double dr_2 = sqrt(rhoPhi);
      const double dr_3 = f0 + 3.141592653589793 * T;
      const double dr_4 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      DiFfRG::vauto::fill(fenv, l1, f0, T, dr_0, dr_1, dr_2, dr_3, dr_4);
      const auto _interp1 = ntIm(DiFfRG::vauto::tr0(fenv));
      const auto _interp2 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = ntRe(DiFfRG::vauto::tr1(fenv));
      const auto _interp5 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp6 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp7 = ZPi(-f0, l1);
      const auto _interp8 = ntRe(DiFfRG::vauto::tr2(fenv));
      const auto _interp9 = ZSig(-f0, l1);
      const auto _den1 = powr<-1>(_interp5 + d1V + _interp7 * (powr<2>(f0) + powr<2>(l1)));
      const auto _den2 = powr<-1>(_interp5 + d1V + _interp9 * (powr<2>(f0) + powr<2>(l1)) + 2. * d2V * rhoPhi);
      const auto _den3 = powr<-1>(powr<2>(_interp2 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _cse1 = -_interp5 * etaPi0;
      const auto _cse2 = _cse1 + _interp6;
      return 0.5 * fma(_cse2, _den1 * _interp4, fma(_cse2, _den2 * _interp8, fma(-4., _den3 * _interp1 * _interp3 * powr<-1>(l1), 0.)));
    }
    static constexpr std::array<bool, 12> pre_reads{true, true, true, true, true, true, true, false, false, false, true, true};
    static KOKKOS_INLINE_FUNCTION auto pre(const double& l1, const double& f0, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      const double dr_0 = 0.7071067811865475;
      const double dr_1 = hPhi;
      const double dr_2 = sqrt(rhoPhi);
      const double dr_3 = f0 + 3.141592653589793 * T;
      const double dr_4 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp5 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp6 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp7 = ZPi(-f0, l1);
      const auto _interp9 = ZSig(-f0, l1);
      const auto _den3 = powr<-1>(powr<2>(_interp2 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _cse1 = -_interp5 * etaPi0;
      const auto _cse2 = _cse1 + _interp6;
      return DiFfRG::device::make_tuple(dr_0, dr_1, dr_2, dr_3, dr_4, _interp3, _interp5, _interp7, _interp9, _den3, _cse2);
    }
    static KOKKOS_INLINE_FUNCTION auto post(const auto &P_, const double& l1, const double& f0, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto &dr_0 = DiFfRG::device::get<0>(P_);
      [[maybe_unused]] const auto &dr_1 = DiFfRG::device::get<1>(P_);
      [[maybe_unused]] const auto &dr_2 = DiFfRG::device::get<2>(P_);
      [[maybe_unused]] const auto &dr_3 = DiFfRG::device::get<3>(P_);
      [[maybe_unused]] const auto &dr_4 = DiFfRG::device::get<4>(P_);
      [[maybe_unused]] const auto &_interp3 = DiFfRG::device::get<5>(P_);
      [[maybe_unused]] const auto &_interp5 = DiFfRG::device::get<6>(P_);
      [[maybe_unused]] const auto &_interp7 = DiFfRG::device::get<7>(P_);
      [[maybe_unused]] const auto &_interp9 = DiFfRG::device::get<8>(P_);
      [[maybe_unused]] const auto &_den3 = DiFfRG::device::get<9>(P_);
      [[maybe_unused]] const auto &_cse2 = DiFfRG::device::get<10>(P_);
      double fenv[(DiFfRG::vauto::nenv) > 0 ? (DiFfRG::vauto::nenv) : 1];
      DiFfRG::vauto::fill(fenv, l1, f0, T, dr_0, dr_1, dr_2, dr_3, dr_4);
      const auto _interp1 = ntIm(DiFfRG::vauto::tr0(fenv));
      const auto _interp4 = ntRe(DiFfRG::vauto::tr1(fenv));
      const auto _interp8 = ntRe(DiFfRG::vauto::tr2(fenv));
      const auto _den1 = powr<-1>(_interp5 + d1V + _interp7 * (powr<2>(f0) + powr<2>(l1)));
      const auto _den2 = powr<-1>(_interp5 + d1V + _interp9 * (powr<2>(f0) + powr<2>(l1)) + 2. * d2V * rhoPhi);
      return 0.5 * fma(_cse2, _den1 * _interp4, fma(_cse2, _den2 * _interp8, fma(-4., _den3 * _interp1 * _interp3 * powr<-1>(l1), 0.)));
    }
    #else                              // the imaginary part survives: genuinely complex
    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& f0, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      double fenv[(DiFfRG::vauto::nenv) > 0 ? (DiFfRG::vauto::nenv) : 1];
      const double dr_0 = 0.7071067811865475;
      const double dr_1 = hPhi;
      const double dr_2 = sqrt(rhoPhi);
      const double dr_3 = f0 + 3.141592653589793 * T;
      const double dr_4 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      DiFfRG::vauto::fill(fenv, l1, f0, T, dr_0, dr_1, dr_2, dr_3, dr_4);
      const auto _interp1 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp4 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp5 = ZPi(-f0, l1);
      const auto _interp6 = ZSig(-f0, l1);
      const auto _den1 = powr<-1>(_interp3 + d1V + _interp5 * (powr<2>(f0) + powr<2>(l1)));
      const auto _den2 = powr<-1>(_interp3 + d1V + _interp6 * (powr<2>(f0) + powr<2>(l1)) + 2. * d2V * rhoPhi);
      const auto _den3 = powr<-1>(powr<2>(_interp1 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _cse1 = -_interp3 * etaPi0;
      const auto _cse2 = _cse1 + _interp4;
      return 0.5 * fma(_cse2, _den1 * DiFfRG::vauto::tr1(fenv), fma(_cse2, _den2 * DiFfRG::vauto::tr2(fenv), fma(complex<double>(0.,4.), _den3 * DiFfRG::vauto::tr0(fenv) * _interp2 * powr<-1>(l1), 0.)));
    }
    static constexpr std::array<bool, 12> pre_reads{true, true, true, true, true, true, true, false, false, false, true, true};
    static KOKKOS_INLINE_FUNCTION auto pre(const double& l1, const double& f0, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      const double dr_0 = 0.7071067811865475;
      const double dr_1 = hPhi;
      const double dr_2 = sqrt(rhoPhi);
      const double dr_3 = f0 + 3.141592653589793 * T;
      const double dr_4 = 1. + powr<-1>(l1) * RF(powr<2>(k), powr<2>(l1));
      const auto _interp1 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp4 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp5 = ZPi(-f0, l1);
      const auto _interp6 = ZSig(-f0, l1);
      const auto _den3 = powr<-1>(powr<2>(_interp1 + l1) + 0.5 * powr<2>(hPhi) * rhoPhi + powr<2>(f0 + 3.141592653589793 * T));
      const auto _cse1 = -_interp3 * etaPi0;
      const auto _cse2 = _cse1 + _interp4;
      return DiFfRG::device::make_tuple(dr_0, dr_1, dr_2, dr_3, dr_4, _interp2, _interp3, _interp5, _interp6, _den3, _cse2);
    }
    static KOKKOS_INLINE_FUNCTION auto post(const auto &P_, const double& l1, const double& f0, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
    {
      using namespace DiFfRG;
      using namespace DiFfRG::compute;
      using namespace numtracer;
      [[maybe_unused]] const auto &dr_0 = DiFfRG::device::get<0>(P_);
      [[maybe_unused]] const auto &dr_1 = DiFfRG::device::get<1>(P_);
      [[maybe_unused]] const auto &dr_2 = DiFfRG::device::get<2>(P_);
      [[maybe_unused]] const auto &dr_3 = DiFfRG::device::get<3>(P_);
      [[maybe_unused]] const auto &dr_4 = DiFfRG::device::get<4>(P_);
      [[maybe_unused]] const auto &_interp2 = DiFfRG::device::get<5>(P_);
      [[maybe_unused]] const auto &_interp3 = DiFfRG::device::get<6>(P_);
      [[maybe_unused]] const auto &_interp5 = DiFfRG::device::get<7>(P_);
      [[maybe_unused]] const auto &_interp6 = DiFfRG::device::get<8>(P_);
      [[maybe_unused]] const auto &_den3 = DiFfRG::device::get<9>(P_);
      [[maybe_unused]] const auto &_cse2 = DiFfRG::device::get<10>(P_);
      double fenv[(DiFfRG::vauto::nenv) > 0 ? (DiFfRG::vauto::nenv) : 1];
      DiFfRG::vauto::fill(fenv, l1, f0, T, dr_0, dr_1, dr_2, dr_3, dr_4);
      const auto _den1 = powr<-1>(_interp3 + d1V + _interp5 * (powr<2>(f0) + powr<2>(l1)));
      const auto _den2 = powr<-1>(_interp3 + d1V + _interp6 * (powr<2>(f0) + powr<2>(l1)) + 2. * d2V * rhoPhi);
      return 0.5 * fma(_cse2, _den1 * DiFfRG::vauto::tr1(fenv), fma(_cse2, _den2 * DiFfRG::vauto::tr2(fenv), fma(complex<double>(0.,4.), _den3 * DiFfRG::vauto::tr0(fenv) * _interp2 * powr<-1>(l1), 0.)));
    }
    #endif

    static KOKKOS_INLINE_FUNCTION auto constant(const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const auto& d1V, const auto& d2V, const auto& d3V, const auto& ZPi, const auto& ZSig)
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
using DiFfRG::VAuto_kernel;