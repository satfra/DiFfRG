#pragma once

#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG {
template<typename _Regulator>
class V_kernel
{
public: using Regulator = _Regulator;

static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double& l1, const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const auto& etaQ, const auto& etaPhi, const auto& hPhi, const auto& d1V, const auto& d2V, const auto& d3V, const double& rhoPhi)
{
using namespace DiFfRG;using namespace DiFfRG::compute;const auto _interp1 = RB(powr<2>(k), powr<2>(l1));
const auto _interp2 = CothFiniteT(sqrt(d1V + powr<2>(l1) + RB(powr<2>(k), powr<2>(l1))), T);
const auto _interp3 = RBdot(powr<2>(k), powr<2>(l1));
const auto _interp4 = CothFiniteT(sqrt(d1V + powr<2>(l1) + 2. * d2V * rhoPhi + RB(powr<2>(k), powr<2>(l1))), T);
const auto _interp5 = RF(powr<2>(k), powr<2>(l1));
const auto _interp6 = RFdot(powr<2>(k), powr<2>(l1));
const auto _interp7 = TanhFiniteT(muq - sqrt(powr<2>(l1) + powr<2>(hPhi) * powr<-1>(Nf) * rhoPhi + 2. * l1 * RF(powr<2>(k), powr<2>(l1)) + powr<2>(RF(powr<2>(k), powr<2>(l1)))), T);
const auto _interp8 = TanhFiniteT(muq + sqrt(powr<2>(l1) + powr<2>(hPhi) * powr<-1>(Nf) * rhoPhi + 2. * l1 * RF(powr<2>(k), powr<2>(l1)) + powr<2>(RF(powr<2>(k), powr<2>(l1)))), T);
const auto _cse1 = powr<2>(l1);
  return 0.25 * fma(-1., _interp2 * _interp3 * sqrt(powr<-1>(_cse1 + _interp1 + d1V)), fma(_interp1, _interp2 * sqrt(powr<-1>(_cse1 + _interp1 + d1V)) * etaPhi, fma(_interp2, _interp3 * sqrt(powr<-1>(_cse1 + _interp1 + d1V)) * powr<2>(Nf), fma(-1., _interp1 * _interp2 * sqrt(powr<-1>(_cse1 + _interp1 + d1V)) * etaPhi * powr<2>(Nf), fma(_interp3, _interp4 * sqrt(powr<-1>(_cse1 + _interp1 + d1V + 2. * d2V * rhoPhi)), fma(-1., _interp1 * _interp4 * etaPhi * sqrt(powr<-1>(_cse1 + _interp1 + d1V + 2. * d2V * rhoPhi)), fma(4., _interp5 * _interp6 * _interp7 * Nc * Nf * sqrt(powr<-1>(powr<-1>(Nf) * (_cse1 * Nf + powr<2>(_interp5) * Nf + 2. * _interp5 * l1 * Nf + powr<2>(hPhi) * rhoPhi))), fma(-4., _interp5 * _interp6 * _interp8 * Nc * Nf * sqrt(powr<-1>(powr<-1>(Nf) * (_cse1 * Nf + powr<2>(_interp5) * Nf + 2. * _interp5 * l1 * Nf + powr<2>(hPhi) * rhoPhi))), fma(-4., powr<2>(_interp5) * _interp7 * etaQ * Nc * Nf * sqrt(powr<-1>(powr<-1>(Nf) * (_cse1 * Nf + powr<2>(_interp5) * Nf + 2. * _interp5 * l1 * Nf + powr<2>(hPhi) * rhoPhi))), fma(4., powr<2>(_interp5) * _interp8 * etaQ * Nc * Nf * sqrt(powr<-1>(powr<-1>(Nf) * (_cse1 * Nf + powr<2>(_interp5) * Nf + 2. * _interp5 * l1 * Nf + powr<2>(hPhi) * rhoPhi))), fma(4., _interp6 * _interp7 * l1 * Nc * Nf * sqrt(powr<-1>(powr<-1>(Nf) * (_cse1 * Nf + powr<2>(_interp5) * Nf + 2. * _interp5 * l1 * Nf + powr<2>(hPhi) * rhoPhi))), fma(-4., _interp6 * _interp8 * l1 * Nc * Nf * sqrt(powr<-1>(powr<-1>(Nf) * (_cse1 * Nf + powr<2>(_interp5) * Nf + 2. * _interp5 * l1 * Nf + powr<2>(hPhi) * rhoPhi))), fma(-4., _interp5 * _interp7 * etaQ * l1 * Nc * Nf * sqrt(powr<-1>(powr<-1>(Nf) * (_cse1 * Nf + powr<2>(_interp5) * Nf + 2. * _interp5 * l1 * Nf + powr<2>(hPhi) * rhoPhi))), fma(4., _interp5 * _interp8 * etaQ * l1 * Nc * Nf * sqrt(powr<-1>(powr<-1>(Nf) * (_cse1 * Nf + powr<2>(_interp5) * Nf + 2. * _interp5 * l1 * Nf + powr<2>(hPhi) * rhoPhi))), 0.))))))))))))));
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
} using DiFfRG::V_kernel;