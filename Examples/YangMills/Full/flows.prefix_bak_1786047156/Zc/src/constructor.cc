#include "../kernel.hh"

#include "../Zc.hh"

Zc_integrator::Zc_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json)
    : integrator(quadrature_provider, json), quadrature_provider(quadrature_provider)
{
}