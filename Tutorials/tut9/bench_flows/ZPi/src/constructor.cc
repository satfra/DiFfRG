#include "../kernel.hh"

#include "../ZPi.hh"

ZPi_integrator::ZPi_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config) : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}