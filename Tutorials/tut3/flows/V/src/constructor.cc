#include "../kernel.hh"

#include "../V.hh"

V_integrator::V_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json)
    : integrator(quadrature_provider, json), integrator_AD(quadrature_provider, json), quadrature_provider(quadrature_provider)
{
}