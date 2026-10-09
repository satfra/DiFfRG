#include "../kernel.hh"

#include "../ZAcbc.hh"

ZAcbc_integrator::ZAcbc_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &config)
    : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}