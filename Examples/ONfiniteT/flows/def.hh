#pragma once

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/integration.hh>
#include <DiFfRG/physics/integration_finiteT.hh>
#include <DiFfRG/physics/interpolation.hh>
#include <DiFfRG/physics/regulators.hh>
#include <DiFfRG/physics/thermodynamics.hh>

using namespace ::DiFfRG;

struct reg_opts {
  static constexpr uint order = 6;
};
#define __REGULATOR__ ::DiFfRG::PolynomialExpRegulator<reg_opts>