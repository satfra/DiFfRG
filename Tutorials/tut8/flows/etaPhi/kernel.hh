#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG {
  template<typename _Regulator>
  class etaPhi_kernel
  {
    public:
    using Regulator = _Regulator;

    static KOKKOS_INLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& l0, const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const auto& etaQ, const auto& etaPhi, const auto& hPhi, const auto& d1V, const auto& d2V, const auto& d3V, const double& rhoPhi)
    {
      using namespace DiFfRG;using namespace DiFfRG::compute;
      const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
      const auto _interp2 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
      const auto _interp4 = RF(powr<2>(k), powr<2>(l1));
      const auto _interp5 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _interp6 = RFdot(powr<2>(k), powr<2>(l1));
      const auto _den1 = powr<-2>(_interp1 + d1V + powr<2>(l0) + powr<2>(l1));
      const auto _den2 = powr<-1>(_interp1 + d1V + powr<2>(l0) + powr<2>(l1));
      const auto _den3 = powr<-2>(_interp1 + d1V + powr<2>(l0) + powr<2>(l1) + 2. * d2V * rhoPhi);
      const auto _den4 = powr<-1>(_interp1 + d1V + powr<2>(l0) + powr<2>(l1) + 2. * d2V * rhoPhi);
      const auto _den5 = powr<-1>(_interp2 + d1V + powr<2>(l0) + powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
      const auto _den6 = powr<-1>(_interp2 + d1V + powr<2>(l0) + powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + 2. * d2V * rhoPhi);
      const auto _den7 = powr<-2>(powr<2>(_interp4 + l1) + powr<-1>(Nf) * (powr<2>(hPhi) * rhoPhi + Nf * powr<2>(l0 + complex<double>(0.,-1.) * muq + 3.141592653589793 * T)));
      const auto _den8 = powr<-2>(powr<2>(_interp4 + l1) + powr<-1>(Nf) * (powr<2>(hPhi) * rhoPhi + Nf * powr<2>(l0 + complex<double>(0.,1.) * muq + 3.141592653589793 * T)));
      const auto _den9 = powr<-1>(powr<2>(_interp4) + powr<2>(l1) + 2. * _interp4 * sqrt(powr<2>(l1)) + powr<2>(hPhi) * powr<-1>(Nf) * rhoPhi + powr<2>(l0 + complex<double>(0.,-1.) * muq + 3.141592653589793 * T));
      const auto _den10 = powr<-1>(powr<2>(_interp4) + powr<2>(l1) + 2. * _interp4 * sqrt(powr<2>(l1)) + powr<2>(hPhi) * powr<-1>(Nf) * rhoPhi + powr<2>(l0 + complex<double>(0.,1.) * muq + 3.141592653589793 * T));
      const auto _den11 = Nf * powr<-1>(powr<2>(hPhi) * rhoPhi + Nf * (powr<2>(_interp5) + powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + 2. * _interp5 * sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)) + powr<2>(l0 + complex<double>(0.,-1.) * muq + 3.141592653589793 * T)));
      const auto _den12 = Nf * powr<-1>(powr<2>(hPhi) * rhoPhi + Nf * (powr<2>(_interp5) + powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p) + 2. * _interp5 * sqrt(powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p)) + powr<2>(l0 + complex<double>(0.,1.) * muq + 3.141592653589793 * T)));
      const auto _cse1 = powr<-2>(p);
      const auto _cse2 = -_interp3;
      const auto _cse3 = _interp1 * etaPhi;
      const auto _cse4 = _cse2 + _cse3;
      const auto _cse5 = -2. * cos1 * l1;
      const auto _cse6 = _cse5 + p;
      const auto _cse7 = _cse6 * p;
      const auto _cse8 = -_interp1;
      const auto _cse9 = _cse7 + _cse8 + _interp2;
      const auto _cse10 = 3.141592653589793 * T;
      const auto _cse11 = _cse10 + l0;
      const auto _cse12 = -2. * cos1 * l1 * p;
      const auto _cse13 = powr<-1>(_cse1) + _cse12 + powr<2>(l1);
      const auto _cse14 = powr<-1>(Nf);
      // clang-format off
      return real(fma(-2., _cse1 * _cse4 * _cse9 * _den2 * _den3 * _den5 * powr<2>(d2V) * rhoPhi, fma(-2., _cse1 * _cse4 * _cse9 * _den1 * _den4 * _den6 * powr<2>(d2V) * rhoPhi, fma(2., _cse1 * _den8 * (_interp6 - _interp4 * etaQ) * powr<2>(hPhi) * Nc * (_den10 * sqrt(powr<-1>(powr<2>(l1))) * (_interp4 * l1 * (-powr<2>(_cse11) + powr<2>(_interp4) + 2. * _interp4 * l1 + powr<2>(l1) + complex<double>(0.,-2.) * _cse11 * muq + powr<2>(muq) - _cse14 * powr<2>(hPhi) * rhoPhi) + _cse14 * sqrt(powr<2>(l1)) * (powr<2>(_interp4) * l1 * Nf + powr<3>(l1) * Nf + 2. * _interp4 * (powr<2>(_cse11) * Nf + powr<2>(l1) * Nf + complex<double>(0.,2.) * _cse11 * muq * Nf - powr<2>(muq) * Nf + powr<2>(hPhi) * rhoPhi) + l1 * (-powr<2>(complex<double>(0.,-1.) * _cse11 + muq) * Nf + powr<2>(hPhi) * rhoPhi))) - sqrt(powr<-1>(_cse13)) * _den12 * powr<-1>(l1) * (_interp5 * l1 * (l1 - cos1 * p) * (-powr<2>(_cse11) + powr<2>(_interp4) + 2. * _interp4 * l1 + powr<2>(l1) + complex<double>(0.,-2.) * _cse11 * muq + powr<2>(muq) - _cse14 * powr<2>(hPhi) * rhoPhi) - sqrt(_cse13) * l1 * (-powr<3>(l1) + powr<2>(_interp4) * (-l1 + cos1 * p) + cos1 * p * (powr<2>(complex<double>(0.,-1.) * _cse11 + muq) - _cse14 * powr<2>(hPhi) * rhoPhi) + 2. * _interp4 * (-powr<2>(_cse11) - powr<2>(l1) + complex<double>(0.,-2.) * _cse11 * muq + powr<2>(muq) + cos1 * l1 * p - _cse14 * powr<2>(hPhi) * rhoPhi) + l1 * (powr<2>(complex<double>(0.,-1.) * _cse11 + muq) + cos1 * l1 * p - _cse14 * powr<2>(hPhi) * rhoPhi)))), fma(2., _cse1 * _den7 * (_interp6 - _interp4 * etaQ) * powr<2>(hPhi) * Nc * (_den9 * sqrt(powr<-1>(powr<2>(l1))) * (_interp4 * l1 * (-powr<2>(_cse11) + powr<2>(_interp4) + 2. * _interp4 * l1 + powr<2>(l1) + complex<double>(0.,2.) * _cse11 * muq + powr<2>(muq) - _cse14 * powr<2>(hPhi) * rhoPhi) + sqrt(powr<2>(l1)) * (powr<2>(_interp4) * l1 + powr<3>(l1) - l1 * powr<2>(complex<double>(0.,1.) * _cse11 + muq) + _cse14 * powr<2>(hPhi) * l1 * rhoPhi - 2. * _interp4 * (-powr<2>(_cse11) - powr<2>(l1) + complex<double>(0.,2.) * _cse11 * muq + powr<2>(muq) - _cse14 * powr<2>(hPhi) * rhoPhi))) - sqrt(powr<-1>(_cse13)) * _den11 * powr<-1>(l1) * (_interp5 * l1 * (l1 - cos1 * p) * (-powr<2>(_cse11) + powr<2>(_interp4) + 2. * _interp4 * l1 + powr<2>(l1) + complex<double>(0.,2.) * _cse11 * muq + powr<2>(muq) - _cse14 * powr<2>(hPhi) * rhoPhi) - sqrt(_cse13) * l1 * (-powr<3>(l1) + powr<2>(_interp4) * (-l1 + cos1 * p) + cos1 * p * (powr<2>(complex<double>(0.,1.) * _cse11 + muq) - _cse14 * powr<2>(hPhi) * rhoPhi) + 2. * _interp4 * (-powr<2>(_cse11) - powr<2>(l1) + complex<double>(0.,2.) * _cse11 * muq + powr<2>(muq) + cos1 * l1 * p - _cse14 * powr<2>(hPhi) * rhoPhi) + l1 * (powr<2>(complex<double>(0.,1.) * _cse11 + muq) + cos1 * l1 * p - _cse14 * powr<2>(hPhi) * rhoPhi)))), 0.)))));
      // clang-format on
    }

    static KOKKOS_INLINE_FUNCTION auto constant(const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const auto& etaQ, const auto& etaPhi, const auto& hPhi, const auto& d1V, const auto& d2V, const auto& d3V, const double& rhoPhi)
    {
      using namespace DiFfRG;using namespace DiFfRG::compute;
      return 0.;
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
} using DiFfRG::etaPhi_kernel;