#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG {
template<typename _Regulator>
class etaPhi_kernel
{
public: using Regulator = _Regulator;

static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& l0, const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const auto& etaQ, const auto& etaPhi, const auto& hPhi, const auto& d1V, const auto& d2V, const auto& d3V, const double& rhoPhi)
{
using namespace DiFfRG;using namespace DiFfRG::compute;const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
const auto _interp2 = RB(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
const auto _interp4 = RF(powr<2>(k), powr<2>(l1));
const auto _interp5 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
const auto _interp6 = RFdot(powr<2>(k), powr<2>(l1));
const auto _cse1 = powr<2>(l0);
const auto _cse2 = powr<2>(l1);
const auto _cse3 = powr<2>(d2V);
const auto _cse17 = powr<2>(p);
const auto _cse4 = powr<-1>(_cse17);
const auto _cse5 = _cse1 + _cse2 + _interp1 + d1V;
const auto _cse6 = -_interp3;
const auto _cse7 = _interp1 * etaPhi;
const auto _cse8 = _cse6 + _cse7;
const auto _cse9 = -2. * cos1 * l1;
const auto _cse10 = _cse9 + p;
const auto _cse11 = _cse10 * p;
const auto _cse12 = -_interp1;
const auto _cse13 = _cse11 + _cse12 + _interp2;
const auto _cse14 = 2. * d2V * rhoPhi;
const auto _cse15 = _cse1 + _cse14 + _cse2 + _interp1 + d1V;
const auto _cse16 = -2. * cos1 * l1 * p;
const auto _cse18 = powr<2>(hPhi);
const auto _cse19 = 3.141592653589793 * T;
const auto _cse20 = _cse19 + l0;
const auto _cse21 = powr<2>(muq);
const auto _cse22 = -_cse21;
const auto _cse23 = powr<2>(_interp4);
const auto _cse24 = powr<-1>(Nf);
const auto _cse25 = _cse18 * _cse24 * rhoPhi;
const auto _cse26 = complex<double>(0.,-2.) * _cse20 * muq;
  return real(fma(-2., _cse13 * powr<-2>(_cse15) * _cse3 * _cse4 * powr<-1>(_cse5) * _cse8 * powr<-1>(_cse1 + _cse16 + _cse17 + _cse2 + _interp2 + d1V) * rhoPhi, fma(-2., _cse13 * powr<-1>(_cse15) * _cse3 * _cse4 * powr<-2>(_cse5) * _cse8 * powr<-1>(_cse1 + _cse14 + _cse16 + _cse17 + _cse2 + _interp2 + d1V) * rhoPhi, fma(4., _cse18 * _cse4 * (_interp6 - _interp4 * etaQ) * powr<-1>(l1) * powr<-2>(_cse2 + powr<2>(_cse20) + _cse22 + _cse23 + _cse25 + _cse26 + 2. * _interp4 * l1) * Nc * (sqrt(powr<-1>(_cse2)) * powr<-1>(_cse2 + powr<2>(_cse20) + _cse22 + _cse23 + _cse25 + _cse26 + 2. * sqrt(_cse2) * _interp4) * l1 * (_interp4 * l1 * (_cse2 - powr<2>(_cse20) + _cse21 + _cse23 + 2. * _interp4 * l1 + complex<double>(0.,2.) * _cse20 * muq - _cse18 * _cse24 * rhoPhi) + sqrt(_cse2) * (_cse23 * l1 + powr<3>(l1) - l1 * powr<2>(complex<double>(0.,1.) * _cse20 + muq) + _cse18 * _cse24 * l1 * rhoPhi - 2. * _interp4 * (-_cse2 - powr<2>(_cse20) + _cse21 + complex<double>(0.,2.) * _cse20 * muq - _cse18 * _cse24 * rhoPhi))) + sqrt(powr<-1>(_cse16 + _cse17 + _cse2)) * powr<-1>(_cse16 + _cse17 + _cse2 + powr<2>(_cse20) + _cse22 + _cse25 + _cse26 + 2. * sqrt(_cse16 + _cse17 + _cse2) * _interp5 + powr<2>(_interp5)) * (-_interp5 * l1 * (l1 - cos1 * p) * (_cse2 - powr<2>(_cse20) + _cse21 + _cse23 + 2. * _interp4 * l1 + complex<double>(0.,2.) * _cse20 * muq - _cse18 * _cse24 * rhoPhi) + sqrt(_cse16 + _cse17 + _cse2) * l1 * (-powr<3>(l1) + _cse23 * (-l1 + cos1 * p) + cos1 * p * (powr<2>(complex<double>(0.,1.) * _cse20 + muq) - _cse18 * _cse24 * rhoPhi) + 2. * _interp4 * (-_cse2 - powr<2>(_cse20) + _cse21 + complex<double>(0.,2.) * _cse20 * muq + cos1 * l1 * p - _cse18 * _cse24 * rhoPhi) + l1 * (powr<2>(complex<double>(0.,1.) * _cse20 + muq) + cos1 * l1 * p - _cse18 * _cse24 * rhoPhi)))), 0.))));
}

static KOKKOS_FORCEINLINE_FUNCTION auto constant(const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const auto& etaQ, const auto& etaPhi, const auto& hPhi, const auto& d1V, const auto& d2V, const auto& d3V, const double& rhoPhi)
{
using namespace DiFfRG;using namespace DiFfRG::compute;
  return 0.;
}private: static KOKKOS_FORCEINLINE_FUNCTION auto RB(const auto& k2, const auto& p2)
{
return Regulator::RB(k2, p2);
}

static KOKKOS_FORCEINLINE_FUNCTION auto RF(const auto& k2, const auto& p2)
{
return Regulator::RF(k2, p2);
}

static KOKKOS_FORCEINLINE_FUNCTION auto RBdot(const auto& k2, const auto& p2)
{
return Regulator::RBdot(k2, p2);
}

static KOKKOS_FORCEINLINE_FUNCTION auto RFdot(const auto& k2, const auto& p2)
{
return Regulator::RFdot(k2, p2);
}

static KOKKOS_FORCEINLINE_FUNCTION auto dq2RB(const auto& k2, const auto& p2)
{
return Regulator::dq2RB(k2, p2);
}

static KOKKOS_FORCEINLINE_FUNCTION auto dq2RF(const auto& k2, const auto& p2)
{
return Regulator::dq2RF(k2, p2);
}
};
} using DiFfRG::etaPhi_kernel;