#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG {
  template<typename _Regulator>
  class hPhi0_kernel
  {
    public:
    using Regulator = _Regulator;

    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& l0, const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const auto& etaQ, const auto& etaPhi, const auto& hPhi, const auto& d1V, const auto& d2V, const auto& d3V, const double& rhoPhi)
    {
      using namespace DiFfRG;using namespace DiFfRG::compute;
      const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp3 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp4 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp5 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp6 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _den1 = powr<-2>(_interp1 + d1V + powr<2>(l0) + powr<2>(l1));
      const auto _den2 = powr<-2>(_interp1 + d1V + powr<2>(l0) + powr<2>(l1) + 2. * d2V * rhoPhi);
      const auto _den3 = powr<-2>(powr<2>(_interp5 + l1) + powr<2>(hPhi) * powr<-1>(Nf) * rhoPhi + powr<2>(l0 + complex<double>(0.,1.) * muq + 3.141592653589793 * T));
      const auto _den4 = powr<-1>(_interp4 + d1V + powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + powr<2>(l0 + complex<double>(0.,1.) * muq - p0f + 3.141592653589793 * T));
      const auto _den5 = powr<-1>(_interp4 + d1V + powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + 2. * d2V * rhoPhi + powr<2>(l0 + complex<double>(0.,1.) * muq - p0f + 3.141592653589793 * T));
      const auto _den6 = powr<-1>(Nf * (powr<2>(_interp3) + powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + 2. * _interp3 * sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)) + powr<2>(l0 - p0f)) + powr<2>(hPhi) * rhoPhi);
      const auto _cse1 = -_interp2;
      const auto _cse2 = _interp1 * etaPhi;
      const auto _cse3 = _cse1 + _cse2;
      const auto _cse4 = powr<-1>(Nf);
      const auto _cse5 = _interp5 + l1;
      const auto _cse6 = -_interp5 * etaQ;
      const auto _cse7 = _cse6 + _interp6;
      return real(fma(_cse4, _cse5 * _cse7 * _den3 * _den5 * powr<3>(hPhi), fma(-0.5, _cse3 * _den2 * _den6 * powr<3>(hPhi), fma(_cse5, _cse7 * _den3 * _den4 * powr<3>(hPhi) * (_cse4 - Nf), fma(0.5, _cse3 * _den1 * _den6 * powr<3>(hPhi) * (-1. + powr<2>(Nf)), 0.)))));
    }

    static KOKKOS_INLINE_FUNCTION auto constant(const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const auto& etaQ, const auto& etaPhi, const auto& hPhi, const auto& d1V, const auto& d2V, const auto& d3V, const double& rhoPhi)
    {
      using namespace DiFfRG;using namespace DiFfRG::compute;
      return 0.5 * fma(etaPhi, hPhi, fma(2., etaQ * hPhi, 0.));
    }
    private:
    static KOKKOS_INLINE_FUNCTION auto RB(const auto& k2, const auto& p2)
    {
      return Regulator::RB(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto RF(const auto& k2, const auto& p2)
    {
      return Regulator::RF(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto RBdot(const auto& k2, const auto& p2)
    {
      return Regulator::RBdot(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto RFdot(const auto& k2, const auto& p2)
    {
      return Regulator::RFdot(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto dq2RB(const auto& k2, const auto& p2)
    {
      return Regulator::dq2RB(k2, p2);
    }

    static KOKKOS_INLINE_FUNCTION auto dq2RF(const auto& k2, const auto& p2)
    {
      return Regulator::dq2RF(k2, p2);
    }
  };
} using DiFfRG::hPhi0_kernel;