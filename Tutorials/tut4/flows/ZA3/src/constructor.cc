#include "../kernel.hh"

#include "../ZA3.hh"

ZA3_integrator::ZA3_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json)
    : integrator(quadrature_provider, json), quadrature_provider(quadrature_provider)
{
}