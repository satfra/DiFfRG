// CG-vs-KT discriminator hunt.
//
// Historical context (smoke.cc and Examples/ONfiniteT*) for the LSM-physical
// integrator-based PolyExp T=0.05 setup:
//   - Examples/ONfiniteT/CG reaches t=4 in ~1.7s
//   - Earlier Examples/ONfiniteT/KT runs failed at t≈0.008
//   - The KT variants in this test file now reach their configured target
//     times and are kept as positive regressions.
//
// Differences between the Example's CG and KT configurations:
//   (a) Discretisation:  CG fe_order=4  vs  KT FV (fe_order=0) with MinMod
//   (b) Numerical flux:  LLFFlux (CG)   vs  KT central + wave-speed
//   (c) Boundary:        FlowBoundaries vs  FVDefaultBoundaries / OriginOdd
//   (d) Initial grid:    adaptive (fine near ρ=0)  vs  uniform
//   (e) AMR:             on  vs  off
//   (f) IDA abs_tol:     1e-13  vs  1e-7
//
// (e) and (f) are not controlled in our test harness (no AMR; abs_tol via
// JSON, default 1e-7). This file isolates (d) and partial (a) by:
//   1. Running CG on the EXAMPLE KT uniform grid with default tolerance —
//      does CG still pass? Tests robustness of CG to grid choice.
//   2. Running CG on the same grid with the CG-Example tight tolerance —
//      does tightening abs_tol matter?
//   3. Running KT on the adaptive non-uniform CG-style grid — does KT
//      pass when given the same grid CG uses?
//   4. Running KT on a 4×-finer uniform grid (600 cells × [0, 0.015]) —
//      does uniform refinement help?
//
// References:
//   - arXiv:1903.09503 (Grossi & Wink, LDG predecessor): the IR flat-region
//     wave speed grows exponentially fast; explicit/CFL-style stepping
//     becomes infeasible. They circumvent this with SUNDIALS BDF (implicit).
//     We already use IDA (implicit, BDF-class); so the failure isn't pure
//     CFL — it's stiffness of the IDA-Jacobian, dominated by KT's λ_max
//     term that CG doesn't have. The tests below probe whether grid choice
//     can compensate. These variants now record the currently regressed
//     behavior: each reaches final_time=4.0.
//   - arXiv:2412.16053 (Zorbach et al., 2D-field-space KT): demonstrates
//     KT for multi-invariant problems but doesn't directly address the
//     IR-flat-region failure on the kind of test we run here.

#include "kt_3D_kernels.hh" // for the diagnostic includes
#include "kt_3D_runners.hh"
#include "kt_3D_setup.hh"

// Limiter variants beyond MinMod (default) — for the kinks-off tests below.
#include <DiFfRG/discretization/FV/limiter/central_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/van_albada_limiter.hh>

#include <catch2/catch_all.hpp>

namespace
{
  using namespace on_kt_3D;
} // namespace

// ============================================================================
// Test 1: CG on KT-Example's uniform ρ-grid with KT's default abs_tol=1e-7.
// Hypothesis: CG should still reach t=4 (it's not relying on tight tol).
// If yes → tolerance isn't the discriminator.
// If no  → tolerance matters more than we thought.
// ============================================================================

TEST_CASE("CG-vs-KT: CG on KT-Example uniform grid + default tolerance", "[cg-vs-kt][cg-uniform-loose-tol]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_cg<LSM_CG>(default_rho_grid(), final_time, /*fe_order=*/4,
                              /*abs_tol=*/1.0e-7, /*rel_tol=*/1.0e-7);
}

// ============================================================================
// Test 2: CG on KT-Example's uniform ρ-grid with the CG-Example's tight
// tolerance abs_tol=1e-13 (rel_tol=1e-7).
// Hypothesis: should also reach t=4. If yes, the CG advantage transfers to
// any grid.
// ============================================================================

TEST_CASE("CG-vs-KT: CG on KT-Example uniform grid + tight tolerance", "[cg-vs-kt][cg-uniform-tight-tol]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_cg<LSM_CG>(default_rho_grid(), final_time, /*fe_order=*/4,
                              /*abs_tol=*/1.0e-13, /*rel_tol=*/1.0e-7);
}

// ============================================================================
// Test 3: KT on the CG-Example's adaptive non-uniform grid (fine near ρ=0,
// coarser farther out — same as `Examples/ONfiniteT/parameter.json`'s
// `0:1e-4:5e-3, 5e-3:5e-4:6e-3, 6e-3:1e-3:1.5e-2`).
// Regression: KT reaches final_time=4.0 on the adaptive grid, so this keeps
// grid-adapted KT behavior under ordinary passing-test semantics.
// ============================================================================

TEST_CASE("CG-vs-KT: KT on CG-Example adaptive non-uniform grid", "[cg-vs-kt][kt-adaptive-grid]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_kt_adaptive<LSM_rho_integrator_PolyExp>(example_cg_subranges(), final_time);
}

// ============================================================================
// Test 4: KT on a 4× uniformly-finer ρ-grid (600 cells × [0, 0.015]).
// Regression: KT reaches final_time=4.0 on the uniformly refined grid.
// ============================================================================

TEST_CASE("CG-vs-KT: KT on 4×-finer uniform ρ-grid (600 cells × [0, 0.015])", "[cg-vs-kt][kt-uniform-4x-finer]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  GridSettings fine_grid{600, 0.0, max_rho};
  run_flow_to_time<LSM_rho_integrator_PolyExp>(fine_grid, final_time);
}

// ============================================================================
// Test 5 (bonus): CG on the adaptive non-uniform grid — reproduces what the
// CG Example does exactly (modulo AMR). Should pass to t=4. If CG passes
// here AND on Test 1/2 (uniform), then CG is robust to grid choice; if CG
// only passes on adaptive but fails on uniform, then grid adaptivity is
// also needed for CG.
// ============================================================================

TEST_CASE("CG-vs-KT: CG on CG-Example adaptive non-uniform grid", "[cg-vs-kt][cg-adaptive-grid]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_cg_adaptive<LSM_CG>(example_cg_subranges(), final_time);
}

// ============================================================================
// Test 9-10: Large-N "all-pion" model — the sigma (radial) mode is replaced
// by another pion, so all N modes have m²_pion = u and the total flux is
//   F = N · V_pion(u).
// This eliminates the derivative-dependent diffusion flux entirely: the PDE
// becomes purely hyperbolic (first-order advection) — the setting KT was
// designed for. If KT reaches t=4 here but fails on the full O(N) split,
// then the derivative-dependent diffusion flux (or its interaction with
// KT's per-face reconstruction) is the discriminator.
// ============================================================================

TEST_CASE("CG-vs-KT: KT-ρ large-N all-pion (purely hyperbolic, no diffusion flux)", "[cg-vs-kt][kt-largeN]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time<LSM_rho_largeN_PolyExp>(default_rho_grid(), final_time);
}

TEST_CASE("CG-vs-KT: CG-ρ large-N all-pion (reference)", "[cg-vs-kt][cg-largeN]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();
  run_flow_to_time_cg<LSM_CG_largeN>(default_rho_grid(), final_time);
}

// ============================================================================
// Tests 6-7: limiter-kink diagnostics.
//
// Default file-level Reconstructor is MinMod, which is C^0 but NOT C^1 in u
// (sign-change AND magnitude-tie kinks per cell-face). The smooth-limiter
// variants below isolate the limiter kink while retaining the full speed derivative.
// The reconstruction is u_plus = u_center + 0.5·limiter(slope_L, slope_R)·dx.
// Swap MinMod for a smoother limiter:
//   VanAlbada: C^1 TVD modulo the (rarely-crossed) sign-change boundary
//   Central:   C^∞ but NOT TVD (diagnostic baseline)
// Both already implemented in DiFfRG/include/DiFfRG/discretization/FV/limiter/.
//
// Regression: VanAlbada reaches final_time=4.0 with a smooth limiter.
// ============================================================================

TEST_CASE("CG-vs-KT: KT-ρ uniform grid, VanAlbada limiter + differentiated wave speed", "[cg-vs-kt][kt-vanalbada]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  // Custom limiter requires a one-off run with explicit Assembler.
  // We can't go through run_flow_to_time (which uses the file-level MinMod).
  using ReconstructorVA = def::TVDReconstructor<dim, def::VanAlbadaLimiter, double>;
  using AssemblerVA = FV::KurganovTadmor::Assembler<Discretization, LSM_rho_integrator_PolyExp, ReconstructorVA,
                                                    FV::KurganovTadmor::MaxEigenvalueWaveSpeed>;

  const auto grid = default_rho_grid();
  const ConfigTree json = make_json();
  LSM_rho_integrator_PolyExp model(json, grid);
  Mesh mesh(make_mesh_config(grid));
  Discretization discretization(mesh, json);
  AssemblerVA assembler(discretization, model, json);

  auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "on_kt_3D", "output");
  OutputSession<Discretization> data_out(data_out_path, json);
  auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
  ImplicitTimeStepper time_stepper(json, assembler, data_out, *adaptor);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  time_stepper.run(state, 0.0, final_time);
}

TEST_CASE("CG-vs-KT: KT-ρ uniform grid, Central limiter + differentiated wave speed (NOT TVD)",
          "[cg-vs-kt][kt-central]")
{
  kt_regression::ensure_logger();
  kt_regression::ensure_diffrg_initialized();

  using ReconstructorC = def::TVDReconstructor<dim, def::CentralLimiter, double>;
  using AssemblerC = FV::KurganovTadmor::Assembler<Discretization, LSM_rho_integrator_PolyExp, ReconstructorC,
                                                   FV::KurganovTadmor::MaxEigenvalueWaveSpeed>;

  const auto grid = default_rho_grid();
  const ConfigTree json = make_json();
  LSM_rho_integrator_PolyExp model(json, grid);
  Mesh mesh(make_mesh_config(grid));
  Discretization discretization(mesh, json);
  AssemblerC assembler(discretization, model, json);

  auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, "on_kt_3D", "output");
  OutputSession<Discretization> data_out(data_out_path, json);
  auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
  ImplicitTimeStepper time_stepper(json, assembler, data_out, *adaptor);

  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  time_stepper.run(state, 0.0, final_time);
}
