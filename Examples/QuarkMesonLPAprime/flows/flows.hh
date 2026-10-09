#pragma once

#include "DiFfRG/common/utils.hh"
#include "DiFfRG/physics/integration.hh"
#include "./etaPhi/etaPhi.hh"
#include "./etaQ/etaQ.hh"
#include "./hPhi0/hPhi0.hh"
#include "./V/V.hh"

class LPA_QM_flows
{
public:   LPA_QM_flows(const DiFfRG::ConfigTree& json)
;

 void set_k(const double k)
;

 void set_T(const double T)
;

 void set_typical_E(const double E)
;

 void set_x_extent(const double x_extent)
;

DiFfRG::QuadratureProvider quadrature_provider;

etaPhi_integrator etaPhi;

etaQ_integrator etaQ;

hPhi0_integrator hPhi0;

V_integrator V;
};