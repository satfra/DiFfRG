#include "../kernel.hh"

#include "../Zc.hh"

Zc_integrator::Zc_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &config)
    : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}