#pragma once

#include "DiFfRG/common/utils.hh"
#include "DiFfRG/physics/integration.hh"
#include "./MesonsPlain/MesonsPlain.hh"
#include "./VAuto/VAuto.hh"
#include "./ZPi/ZPi.hh"
#include "./ZSig/ZSig.hh"

class QMBenchFlows
{
  public:
  QMBenchFlows(const DiFfRG::ConfigTree& config)
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

  MesonsPlain_integrator MesonsPlain;

  VAuto_integrator VAuto;

  ZPi_integrator ZPi;

  ZSig_integrator ZSig;
};