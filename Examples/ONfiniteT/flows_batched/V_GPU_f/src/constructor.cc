#include "../kernel.hh"

#include "../V_GPU_f.hh"

V_GPU_f_integrator::V_GPU_f_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config) : integrator(quadrature_provider, config), integrator_AD(quadrature_provider, config), integrator_AD2(quadrature_provider, config), quadrature_provider(quadrature_provider)
{
}