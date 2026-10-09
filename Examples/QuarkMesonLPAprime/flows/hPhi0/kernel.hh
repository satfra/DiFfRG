#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG {
template<typename _Regulator>
class hPhi0_kernel
{
public: using Regulator = _Regulator;

static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double& l1, const double& cos1, const double& l0, const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const auto& etaQ, const auto& etaPhi, const auto& hPhi, const auto& d1V, const auto& d2V, const auto& d3V, const double& rhoPhi)
{
using namespace DiFfRG;using namespace DiFfRG::compute;const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
const auto _interp2 = RBdot(powr<2>(k), powr<2>(l1));
const auto _interp3 = RF(powr<2>(k), powr<2>(l1) - 2. * cos1 * l1 * p + powr<2>(p));
const auto _interp4 = RF(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
const auto _interp5 = RFdot(powr<2>(k), powr<2>(l1) + 2. * cos1 * l1 * p + powr<2>(p));
const auto _cse1 = powr<2>(l0);
const auto _cse2 = powr<2>(l1);
const auto _cse3 = powr<2>(p);
const auto _cse4 = powr<3>(hPhi);
const auto _cse5 = -_interp2;
const auto _cse6 = _interp1 * etaPhi;
const auto _cse7 = _cse5 + _cse6;
const auto _cse8 = _cse1 * Nf;
const auto _cse9 = _cse2 * Nf;
const auto _cse10 = -2. * cos1 * l1 * Nf * p;
const auto _cse11 = _cse3 * Nf;
const auto _cse12 = -2. * l0 * Nf * p0f;
const auto _cse13 = powr<2>(p0f);
const auto _cse14 = _cse13 * Nf;
const auto _cse15 = -2. * cos1 * l1 * p;
const auto _cse16 = _cse15 + _cse2 + _cse3;
const auto _cse17 = sqrt(_cse16);
const auto _cse18 = 2. * _cse17 * _interp3 * Nf;
const auto _cse19 = powr<2>(_interp3);
const auto _cse20 = _cse19 * Nf;
const auto _cse21 = powr<2>(hPhi);
const auto _cse22 = _cse21 * rhoPhi;
const auto _cse23 = _cse10 + _cse11 + _cse12 + _cse14 + _cse18 + _cse20 + _cse22 + _cse8 + _cse9;
const auto _cse24 = powr<-1>(_cse23);
const auto _cse25 = powr<2>(Nf);
const auto _cse26 = -1. + _cse25;
const auto _cse27 = _cse1 + _cse2 + _interp1 + d1V;
  return real(fma(0.1666666666666667, _cse24 * _cse26 * powr<-2>(_cse27) * _cse4 * _cse7 * Nc, fma(0.3333333333333333, _cse26 * powr<-1>(_cse27) * _cse4 * (-_interp5 + _interp4 * etaQ) * Nc * Nf * (_interp4 + sqrt(_cse2 + _cse3 + 2. * cos1 * l1 * p)) * powr<-2>(_cse11 + _cse14 + _cse22 + _cse8 + _cse9 + powr<2>(_interp4) * Nf + 2. * cos1 * l1 * Nf * p + 2. * _interp4 * Nf * sqrt(_cse2 + _cse3 + 2. * cos1 * l1 * p) + 2. * l0 * Nf * p0f), fma(-0.3333333333333333, _cse24 * _cse4 * _cse7 * Nc * Nf * powr<-2>(_cse1 + _cse2 + _interp1 + d1V + 2. * d2V * rhoPhi), fma(-0.6666666666666666, _cse25 * _cse4 * (-_interp5 + _interp4 * etaQ) * Nc * (_interp4 + sqrt(_cse2 + _cse3 + 2. * cos1 * l1 * p)) * powr<-2>(_cse11 + _cse14 + _cse22 + _cse8 + _cse9 + powr<2>(_interp4) * Nf + 2. * cos1 * l1 * Nf * p + 2. * _interp4 * Nf * sqrt(_cse2 + _cse3 + 2. * cos1 * l1 * p) + 2. * l0 * Nf * p0f) * powr<-1>(_cse1 + _cse2 + _interp1 + d1V + 2. * d2V * rhoPhi), 0.)))));
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
} using DiFfRG::hPhi0_kernel;