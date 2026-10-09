#include "../kernel.hh"

#include "../ZA4SP.hh"

ZA4SP_integrator::ZA4SP_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json)
    : integrator(quadrature_provider, json), quadrature_provider(quadrature_provider)
{
}