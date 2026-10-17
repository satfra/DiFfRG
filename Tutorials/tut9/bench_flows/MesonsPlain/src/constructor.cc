#include "../kernel.hh"

#include "../MesonsPlain.hh"

MesonsPlain_integrator::MesonsPlain_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config) : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}