#define CATCH_CONFIG_MAIN
#include "ldg_test_model.hh"

using namespace ldg_test;

TEST_CASE("LDG residual matches a plain mesh_loop assembly", "[discretization][ldg]")
{
  DiFfRG::Init();
  const int fe_order = GENERATE(1, 2);
  SECTION("one level, constant level jacobian")
  {
    Setup<Model2<true, false>> s(fe_order);
    require_residual_matches_reference(s);
  }
  SECTION("one level, assembled level jacobian")
  {
    Setup<Model2<false, false>> s(fe_order);
    require_residual_matches_reference(s);
  }
  SECTION("two levels")
  {
    Setup<Model3<true>> s(fe_order);
    require_residual_matches_reference(s);
  }
}

// With a constant level jacobian the level is built as J_gu * u; every FE component has to reach it, not just
// the first. The assembled level (non-constant path) is the reference.
TEST_CASE("LDG level from a constant jacobian matches the assembled level", "[discretization][ldg]")
{
  DiFfRG::Init();
  const int fe_order = GENERATE(1, 2);
  Setup<Model2<true, false>> constant(fe_order);
  Setup<Model2<false, false>> assembled(fe_order);
  const auto r_c = constant.residual(constant.u), r_a = assembled.residual(assembled.u);
  auto deviation = r_c;
  deviation -= r_a;
  REQUIRE(max_abs(r_a) > 0.);
  INFO("worst deviation " << max_abs(deviation) << ", scale " << max_abs(r_a));
  REQUIRE(max_abs(deviation) <= 1e-11 * max_abs(r_a));
}

TEST_CASE("LDG jacobian matches finite differences of the residual", "[discretization][ldg]")
{
  DiFfRG::Init();
  const int fe_order = GENERATE(1, 2);
  SECTION("constant level jacobian")
  {
    Setup<Model2<true, false>> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
  SECTION("assembled level jacobian")
  {
    Setup<Model2<false, false>> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
  // The extractor enters the volume terms and, through the LLF flux, the face terms; weight != 1.
  SECTION("constant level jacobian, extractor")
  {
    Setup<Model2<true, true>> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
  SECTION("assembled level jacobian, extractor")
  {
    Setup<Model2<false, true>> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
  // Level 2 is nonlinear in level 1, so its jacobian needs level 1 at the current state, and depends on both
  // level-1 components, so the chain rule through level 1 has more than one term per entry. The jacobian is
  // the first call on a fresh assembler: nothing has built level 1 at this state before.
  SECTION("two levels, constant first level")
  {
    Setup<Model3<true>> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
  SECTION("two levels, assembled first level")
  {
    Setup<Model3<false>> s(fe_order);
    require_jacobian_matches_finite_differences(s);
  }
}

// A level that is not constant is rebuilt on every jacobian call; the composite level jacobians must be rebuilt,
// not accumulated, so a repeated call at the same state gives the same matrix.
TEST_CASE("Repeated LDG jacobians at the same state agree", "[discretization][ldg]")
{
  DiFfRG::Init();
  Setup<Model3<true>> s(1);
  const auto &first_assembled = s.jacobian();
  typename decltype(s)::SparseMatrixType first(s.assembler.get_sparsity_pattern_jacobian());
  first.copy_from(first_assembled);
  const auto &second = s.jacobian();
  const auto [worst, scale] = matrix_deviation(first, second);
  REQUIRE(scale > 0.);
  INFO("worst deviation " << worst << ", scale " << scale);
  REQUIRE(worst <= 1e-12 * scale);
}
