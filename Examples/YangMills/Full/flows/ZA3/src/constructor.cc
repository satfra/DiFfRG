#include "../kernel.hh"

#include "../ZA3.hh"

ZA3_integrator::ZA3_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &config)
    : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}