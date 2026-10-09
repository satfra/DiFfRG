#include "../kernel.hh"

#include "../ZA.hh"

ZA_integrator::ZA_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &config)
    : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}