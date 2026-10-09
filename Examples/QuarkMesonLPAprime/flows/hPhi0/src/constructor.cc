#include "../kernel.hh"

#include "../hPhi0.hh"

  hPhi0_integrator::hPhi0_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& json) : integrator(quadrature_provider, json), integrator_AD(quadrature_provider, json), quadrature_provider(quadrature_provider) 
{}
