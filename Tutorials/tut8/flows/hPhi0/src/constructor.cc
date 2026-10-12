#include "../kernel.hh"

#include "../hPhi0.hh"

hPhi0_integrator::hPhi0_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config) : integrator(quadrature_provider, config), integrator_AD(quadrature_provider, config), integrator_AD2(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}