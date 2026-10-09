#include "../kernel.hh"

#include "../etaQ.hh"

  etaQ_integrator::etaQ_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& json) : integrator(quadrature_provider, json), integrator_AD(quadrature_provider, json), quadrature_provider(quadrature_provider) 
{}
