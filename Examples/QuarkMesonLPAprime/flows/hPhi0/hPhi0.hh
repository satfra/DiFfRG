#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/interpolation.hh"

namespace DiFfRG { template<typename> class hPhi0_kernel;

class hPhi0_integrator
{
public:   hPhi0_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& json)
;


using Regulator = DiFfRG::PolynomialExpRegulator<>;

Integrator_fT_p2_1ang<4, double, hPhi0_kernel<Regulator>, DiFfRG::TBB_exec> integrator;

Integrator_fT_p2_1ang<4, autodiff::real, hPhi0_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD;

 void get(double& dest, const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const double& etaQ, const double& etaPhi, const double& hPhi, const double& d1V, const double& d2V, const double& d3V, const double& rhoPhi)
;

template<typename IT, typename ...T>
 void get(IT& dest, const device::tuple<T...>& args)
{
device::apply([&](const auto...t){get(dest, t...);}, args);
}

 void get(autodiff::real& dest, const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const autodiff::real& etaQ, const autodiff::real& etaPhi, const autodiff::real& hPhi, const autodiff::real& d1V, const autodiff::real& d2V, const autodiff::real& d3V, const double& rhoPhi)
;private: DiFfRG::QuadratureProvider& quadrature_provider;
};
}
using DiFfRG::hPhi0_integrator;