#define CATCH_CONFIG_MAIN
#include "ldg_test_model.hh"

#include <tbb/global_control.h>

#include <optional>

using namespace ldg_test;

namespace
{
  template <typename A, typename B> void require_same_assembly(A &a, B &b)
  {
    const auto r_a = a.residual(a.u), r_b = b.residual(b.u);
    auto deviation = r_a;
    deviation -= r_b;
    REQUIRE(max_abs(r_b) > 0.);
    REQUIRE(max_abs(deviation) <= 1e-12 * max_abs(r_b));
    // The first jacobian may rebuild the sparsity pattern (a new extractor cell), so the copy is made after it.
    const auto &J_b_assembled = b.jacobian();
    typename B::SparseMatrixType J_b(b.assembler.get_sparsity_pattern_jacobian());
    J_b.copy_from(J_b_assembled);
    const auto [worst, scale] = matrix_deviation(J_b, a.jacobian());
    REQUIRE(scale > 0.);
    REQUIRE(worst <= 1e-12 * scale);
  }
} // namespace

TEST_CASE("A model's own evaluate_batch matches the per-point default under LDG", "[discretization][ldg]")
{
  DiFfRG::Init();
  SECTION("one level, extractor")
  {
    Setup<Model2<true, true, true>> batched(2);
    Setup<Model2<true, true, false>> per_point(2);
    require_same_assembly(batched, per_point);
  }
  SECTION("two levels")
  {
    Setup<Model3<false, true>> batched(2);
    Setup<Model3<false, false>> per_point(2);
    require_same_assembly(batched, per_point);
  }
}

// Thousands of cells, so that the cells of one color really are assembled on many threads at once. The
// result must not depend on the thread count at all.
TEST_CASE("LDG assembly does not depend on the thread count", "[discretization][ldg]")
{
  DiFfRG::Init();
  Setup<Model3<false, true>> s(2, "0:0.00025:1");
  const auto r_all = s.residual(s.u);
  const auto &J_all_assembled = s.jacobian();
  typename decltype(s)::SparseMatrixType J_all(s.assembler.get_sparsity_pattern_jacobian());
  J_all.copy_from(J_all_assembled);
  std::optional<tbb::global_control> serial;
  serial.emplace(tbb::global_control::max_allowed_parallelism, 1);
  const auto r_one = s.residual(s.u);
  const auto &J_one_assembled = s.jacobian();
  typename decltype(s)::SparseMatrixType J_one(s.assembler.get_sparsity_pattern_jacobian());
  J_one.copy_from(J_one_assembled);
  serial.reset();

  auto deviation = r_all;
  deviation -= r_one;
  REQUIRE(max_abs(r_all) > 0.);
  REQUIRE(max_abs(deviation) == 0.);
  // Entry by entry over the blocks: an el() loop over 24000^2 entries would take minutes.
  double worst = 0., scale = 0.;
  for (uint b = 0; b < J_all.n_block_rows() * J_all.n_block_cols(); ++b) {
    const auto &a = J_all.block(b / J_all.n_block_cols(), b % J_all.n_block_cols());
    const auto &o = J_one.block(b / J_all.n_block_cols(), b % J_all.n_block_cols());
    for (auto it = a.begin(), jt = o.begin(); it != a.end(); ++it, ++jt) {
      scale = std::max(scale, std::abs(it->value()));
      worst = std::max(worst, std::abs(it->value() - jt->value()));
    }
  }
  REQUIRE(scale > 0.);
  REQUIRE(worst == 0.);
}
