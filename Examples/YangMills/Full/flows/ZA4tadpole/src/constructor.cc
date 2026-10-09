#include "../kernel.hh"

#include "../ZA4tadpole.hh"

ZA4tadpole_integrator::ZA4tadpole_integrator(DiFfRG::QuadratureProvider &quadrature_provider,
                                             const DiFfRG::ConfigTree &config)
    : integrator(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}