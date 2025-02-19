#pragma once

#include "PhiPhi/etaPhi.hh"
#include "Phiqbq/hPhi0.hh"
#include "V/V.hh"
#include "qbq/etaQ.hh"

#include "def.hh"
#include <DiFfRG/physics/flow_equations.hh>

class QuarkMesonFlowEquations : public FlowEquationsFiniteT
{
public:
  QuarkMesonFlowEquations(const JSONValue &json);

private:
  const std::array<uint, 1> grid_size_int;
  const std::array<uint, 2> grid_sizes_angle_int;
  const std::array<uint, 3> grid_sizes_3D_int;
  const std::array<uint, 4> grid_sizes_4D_int;

  const std::array<uint, 2> grid_sizes_int_fT;
  const std::array<uint, 3> grid_sizes_angle_int_fT;
  const std::array<uint, 4> grid_sizes_4D_int_fT;

  const std::array<uint, 2> grid_sizes_2D_cartesian_int;
  const std::array<uint, 3> grid_sizes_3D_cartesian_int;

public:
  QuadratureProvider quadrature_provider;
  Flows::V_integrator V_integrator;
  Flows::etaQ_integrator etaQ_integrator;
  Flows::etaPhi_integrator etaPhi_integrator;
  Flows::hPhi0_integrator hPhi0_integrator;
};