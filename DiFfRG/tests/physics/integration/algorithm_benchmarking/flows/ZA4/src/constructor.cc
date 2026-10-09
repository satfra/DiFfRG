#include "../kernel.hh"

#include "../ZA4.hh"

ZA4_integrator::ZA4_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json)
    : integrator(quadrature_provider, json), quadrature_provider(quadrature_provider)
{
}