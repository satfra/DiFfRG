#pragma once

#include "./ZA4/ZA4.hh"
#include "DiFfRG/common/utils.hh"
#include "DiFfRG/physics/integration.hh"

class YangMillsFlows
{
public:
  YangMillsFlows(const DiFfRG::ConfigTree &json);

  void set_k(const double k);

  void set_T(const double T);

  void set_typical_E(const double E);

  void set_x_extent(const double x_extent);

  DiFfRG::QuadratureProvider quadrature_provider;

  ZA4_integrator ZA4;
};