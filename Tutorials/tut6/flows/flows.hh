#pragma once

#include "DiFfRG/common/utils.hh"
#include "DiFfRG/physics/integration.hh"
#include "./V/V.hh"
#include "./V_GPU/V_GPU.hh"
#include "./V_pion/V_pion.hh"
#include "./V_pion_GPU/V_pion_GPU.hh"
#include "./V_sigma/V_sigma.hh"
#include "./V_sigma_GPU/V_sigma_GPU.hh"

class ONFlows
{
  public:
  ONFlows(const DiFfRG::ConfigTree& config)
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

  V_integrator V;

  V_GPU_integrator V_GPU;

  V_pion_integrator V_pion;

  V_pion_GPU_integrator V_pion_GPU;

  V_sigma_integrator V_sigma;

  V_sigma_GPU_integrator V_sigma_GPU;
};