// Smoke tests: one TEST_CASE per (model × coordinate × flux-source) variant.
// Each runs the LSM physical-scale setup (Λ=0.65, m²=-0.1, λ=71.6, N=2,
// T=0.05) from t=0 to their configured target time on the Example grid.
// The rho-coordinate KT variants now reach final_time=4.0 and are kept as
// ordinary positive regressions alongside the CG reference.

#include "kt_3D_runners.hh"

#include <catch2/catch_all.hpp>

namespace
{
  using namespace on_kt_3D;
}

// --- σ-coords --------------------------------------------------------------

TEST_CASE("LSM σ-coords analytic Litim T=0 reaches t=3.6",
          "[smoke][lsm-physical][sigma][analytic]")
{
  kt_regression::ensure_logger();
  run_flow_to_time<LSM_sigma_analytic>(default_sigma_grid(), smoke_target_time);
}

TEST_CASE("LSM σ-coords integrator Litim T=0 reaches t=3.6",
          "[smoke][lsm-physical][sigma][integrator-litim]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time<LSM_sigma_integrator_Litim>(default_sigma_grid(), smoke_target_time);
}

TEST_CASE("LSM σ-coords integrator PolyExp T=0.05 reaches t=3.6",
          "[smoke][lsm-physical][sigma][integrator-polyexp]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time<LSM_sigma_integrator_PolyExp>(default_sigma_grid(), smoke_target_time);
}

// --- ρ-coords --------------------------------------------------------------

TEST_CASE("LSM ρ-coords analytic Litim T=0 reaches final time",
          "[smoke][lsm-physical][rho][analytic]")
{
  kt_regression::ensure_logger();
  run_flow_to_time<LSM_rho_analytic>(default_rho_grid(), final_time);
}

TEST_CASE("LSM ρ-coords integrator Litim T=0 reaches final time",
          "[smoke][lsm-physical][rho][integrator-litim]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time<LSM_rho_integrator_Litim>(default_rho_grid(), final_time);
}

TEST_CASE("LSM ρ-coords integrator PolyExp T=0.05 reaches final time",
          "[smoke][lsm-physical][rho][integrator-polyexp]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time<LSM_rho_integrator_PolyExp>(default_rho_grid(), final_time);
}

// --- CG --------------------------------------------------------------------

TEST_CASE("LSM CG integrator PolyExp T=0.05 reaches final time",
          "[smoke][lsm-physical][cg][integrator-polyexp]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_cg<LSM_CG>(default_rho_grid(), final_time);
}
