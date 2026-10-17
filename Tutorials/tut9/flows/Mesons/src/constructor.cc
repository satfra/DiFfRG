#include "../kernel.hh"

#include "../Mesons.hh"

Mesons_integrator::Mesons_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config) : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}