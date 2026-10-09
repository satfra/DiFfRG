#define CATCH_CONFIG_MAIN
#include "dg_test_model.hh"

using namespace dg_test;

TEST_CASE("DG residuals match a plain mesh_loop assembly", "[discretization][dg]")
{
  DiFfRG::Init();
  const int fe_order = GENERATE(1, 2);
  SECTION("DG values only, 1D, LLF")
  {
    Setup<1, false, false, false, false> s(fe_order, "0:0.1:1");
    require_residual_matches_reference(s);
  }
  SECTION("DG with derivatives, 1D, hessians, LLF")
  {
    Setup<1, true, false, true, false> s(fe_order, "0:0.1:1");
    require_residual_matches_reference(s);
  }
  SECTION("DG values only, 2D with hanging faces, custom numflux")
  {
    Setup<2, false, false, false, false, NumFlux::custom> s(fe_order, "0:0.25:1", true);
    require_residual_matches_reference(s);
  }
  SECTION("DG with derivatives, 2D with hanging faces, hessians, LLF")
  {
    Setup<2, true, false, true, false> s(fe_order, "0:0.25:1", true);
    require_residual_matches_reference(s);
  }
}

TEST_CASE("DG jacobians match finite differences of the residual", "[discretization][dg]")
{
  DiFfRG::Init();
  const int fe_order = GENERATE(1, 2);
  SECTION("DG values only, 1D, per-point model, extractor, custom numflux")
  {
    Setup<1, false, false, false, true, NumFlux::custom> s(fe_order, "0:0.1:1");
    require_jacobian_matches_finite_differences(s);
  }
  SECTION("DG with derivatives, 1D, batched model, hessians, extractor, LLF")
  {
    Setup<1, true, true, true, true> s(fe_order, "0:0.1:1");
    require_jacobian_matches_finite_differences(s);
  }
  SECTION("DG with derivatives, 2D with hanging faces, hessians, LLF")
  {
    Setup<2, true, false, true, false> s(fe_order, "0:0.25:1", true);
    require_jacobian_matches_finite_differences(s);
  }
}
