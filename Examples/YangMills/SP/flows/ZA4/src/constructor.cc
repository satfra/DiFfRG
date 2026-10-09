#include "../kernel.hh"

#include "../ZA4.hh"

ZA4_integrator::ZA4_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &config)
    : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}