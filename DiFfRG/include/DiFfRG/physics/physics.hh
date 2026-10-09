#pragma once

/**
 * @file physics.hh
 * @brief Convenience header for the physics helpers: regulators (regulators.hh), thermal distribution functions
 * (thermodynamics.hh), finite-temperature Litim threshold functions (threshold_functions/litim.hh) and the simple
 * radial loop integral (loop_integrals.hh). Momentum integrators and interpolators have their own headers,
 * DiFfRG/physics/integration.hh and DiFfRG/physics/interpolation.hh.
 */

#include <DiFfRG/physics/threshold_functions/litim.hh>

#include <DiFfRG/physics/loop_integrals.hh>
#include <DiFfRG/physics/regulators.hh>
#include <DiFfRG/physics/thermodynamics.hh>
