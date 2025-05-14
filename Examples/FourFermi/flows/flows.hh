#pragma once

#include "qbqqbq/lambda1.hh"
#include "qbqqbq/lambda10.hh"
#include "qbqqbq/lambda2.hh"
#include "qbqqbq/lambda3.hh"
#include "qbqqbq/lambda4.hh"
#include "qbqqbq/lambda5.hh"
#include "qbqqbq/lambda6.hh"
#include "qbqqbq/lambda7.hh"
#include "qbqqbq/lambda8.hh"
#include "qbqqbq/lambda9.hh"

#include "def.hh"
#include <DiFfRG/physics/flow_equations.hh>

class FourFermiFlowEquations : public FlowEquationsFiniteT
{
public:
  FourFermiFlowEquations(const JSONValue &json);

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
  Flows::lambda1_integrator lambda1_integrator;
  Flows::lambda2_integrator lambda2_integrator;
  Flows::lambda3_integrator lambda3_integrator;
  Flows::lambda4_integrator lambda4_integrator;
  Flows::lambda5_integrator lambda5_integrator;
  Flows::lambda6_integrator lambda6_integrator;
  Flows::lambda7_integrator lambda7_integrator;
  Flows::lambda8_integrator lambda8_integrator;
  Flows::lambda9_integrator lambda9_integrator;
  Flows::lambda10_integrator lambda10_integrator;
};