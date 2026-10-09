#pragma once

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/integration.hh>
#include <DiFfRG/physics/interpolation.hh>
#include <DiFfRG/physics/regulators.hh>
#include <DiFfRG/physics/thermodynamics.hh>

using namespace ::DiFfRG;

#define __REGULATOR__ ::DiFfRG::PolynomialExpRegulator<>
