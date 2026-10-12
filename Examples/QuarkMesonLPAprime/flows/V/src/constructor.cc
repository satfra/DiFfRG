#include "../kernel.hh"

#include "../V.hh"

V_integrator::V_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config) : integrator(quadrature_provider, config), integrator_AD(quadrature_provider, config), integrator_AD2(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}