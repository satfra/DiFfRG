#pragma once

#include "./ZA/ZA.hh"
#include "./ZA3/ZA3.hh"
#include "./ZA4SP/ZA4SP.hh"
#include "./ZA4tadpole/ZA4tadpole.hh"
#include "./ZAcbc/ZAcbc.hh"
#include "./Zc/Zc.hh"
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

  ZA_integrator ZA;

  ZA3_integrator ZA3;

  ZA4SP_integrator ZA4SP;

  ZA4tadpole_integrator ZA4tadpole;

  ZAcbc_integrator ZAcbc;

  Zc_integrator Zc;
};