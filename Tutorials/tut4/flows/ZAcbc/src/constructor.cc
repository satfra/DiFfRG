#include "../kernel.hh"

#include "../ZAcbc.hh"

ZAcbc_integrator::ZAcbc_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json)
    : integrator(quadrature_provider, json), quadrature_provider(quadrature_provider)
{
}