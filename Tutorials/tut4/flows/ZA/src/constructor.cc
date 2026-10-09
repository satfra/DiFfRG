#include "../kernel.hh"

#include "../ZA.hh"

ZA_integrator::ZA_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json)
    : integrator(quadrature_provider, json), quadrature_provider(quadrature_provider)
{
}