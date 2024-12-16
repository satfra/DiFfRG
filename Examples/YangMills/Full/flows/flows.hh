#pragma once

#include "AA/ZA.hh"
#include "AA/m2A.hh"
#include "AAA/ZA3.hh"
#include "AAAA/ZA4SP.hh"
#include "AAAA/ZA4tadpole.hh"
#include "Acbc/ZAcbc.hh"
#include "cbc/Zc.hh"

#include "def.hh"
#include <DiFfRG/physics/flow_equations.hh>

class YangMillsFlowEquations : public FlowEquations
{
public:
  YangMillsFlowEquations(const JSONValue &json);

private:
  const std::array<uint, 1> grid_size_int;
  const std::array<uint, 2> grid_sizes_angle_int;
  const std::array<uint, 3> grid_sizes_3D_int;
  const std::array<uint, 4> grid_sizes_4D_int;

public:
  ::DiFfRG::QuadratureProvider quadrature_provider;
  ::DiFfRG::Flows::ZA_integrator ZA_integrator;
  ::DiFfRG::Flows::m2A_integrator m2A_integrator;
  ::DiFfRG::Flows::Zc_integrator Zc_integrator;
  ::DiFfRG::Flows::ZA3_integrator ZA3_integrator;
  ::DiFfRG::Flows::ZAcbc_integrator ZAcbc_integrator;
  ::DiFfRG::Flows::ZA4tadpole_integrator ZA4tadpole_integrator;
  ::DiFfRG::Flows::ZA4SP_integrator ZA4SP_integrator;
};