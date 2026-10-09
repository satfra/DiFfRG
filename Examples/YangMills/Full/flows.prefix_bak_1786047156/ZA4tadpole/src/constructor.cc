#include "../kernel.hh"

#include "../ZA4tadpole.hh"

ZA4tadpole_integrator::ZA4tadpole_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json)
    : integrator(quadrature_provider, json), quadrature_provider(quadrature_provider)
{
}