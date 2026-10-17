#include "../kernel.hh"

#include "../ZSig.hh"

ZSig_integrator::ZSig_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config) : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}