#define CATCH_CONFIG_MAIN
#include "dg_test_model.hh"

#include <tbb/global_control.h>

#include <optional>

using namespace dg_test;

namespace
{
  template <typename A, typename B> void require_same_assembly(A &a, B &b)
  {
    const auto r_a = a.residual(a.u), r_b = b.residual(b.u);
    auto deviation = r_a;
    deviation -= r_b;
    REQUIRE(max_abs(r_b) > 0.);
    REQUIRE(max_abs(deviation) <= 1e-12 * max_abs(r_b));
    const auto [worst, scale] = matrix_deviation(b.jacobian(), a.jacobian());
    REQUIRE(scale > 0.);
    REQUIRE(worst <= 1e-12 * scale);
  }

  // An input the assembler does not gather is absent at compile time, in the batch and in the per-point tuple, so a
  // model cannot read zeros by mistake.
  template <bool derivatives, bool hessians>
  using TestBatch = PointBatch<1, double, 2, std::array<double, 0>, Vector<double>, derivatives, hessians>;
  template <typename Batch>
  using PointTuple = decltype(std::declval<const Batch &>().tie(std::declval<typename Batch::State &>()));
  template <typename Batch> constexpr bool has_derivative_column = requires(const Batch &b) { b.derivatives(0, 0); };
  template <typename Batch> constexpr bool has_hessian_column = requires(const Batch &b) { b.hessians(0, 0, 0); };

  static_assert(!has_derivative_column<TestBatch<false, false>> && !has_hessian_column<TestBatch<false, false>>);
  static_assert(has_derivative_column<TestBatch<true, false>> && !has_hessian_column<TestBatch<true, false>>);
  static_assert(has_derivative_column<TestBatch<true, true>> && has_hessian_column<TestBatch<true, true>>);
  static_assert(!tuple_has<"fe_derivatives", PointTuple<TestBatch<false, false>>> &&
                !tuple_has<"fe_hessians", PointTuple<TestBatch<false, false>>>);
  static_assert(tuple_has<"fe_derivatives", PointTuple<TestBatch<true, false>>> &&
                !tuple_has<"fe_hessians", PointTuple<TestBatch<true, false>>>);
  static_assert(tuple_has<"fe_derivatives", PointTuple<TestBatch<true, true>>> &&
                tuple_has<"fe_hessians", PointTuple<TestBatch<true, true>>>);
} // namespace

TEST_CASE("A model's own evaluate_batch matches the per-point default under DG", "[discretization][dg]")
{
  DiFfRG::Init();
  Setup<1, true, true, true, true> batched(2, "0:0.1:1");
  Setup<1, true, false, true, true> per_point(2, "0:0.1:1");
  require_same_assembly(batched, per_point);
}

TEST_CASE("The batched LLF numflux matches the per-point LLF numflux", "[discretization][dg]")
{
  DiFfRG::Init();
  Setup<1, true, false, true, true, NumFlux::llf> batched(2, "0:0.1:1");
  Setup<1, true, false, true, true, NumFlux::llf_per_point> per_point(2, "0:0.1:1");
  require_same_assembly(batched, per_point);
}

// Thousands of cells, so that the cells of one color really are assembled on many threads at once. The
// result must not depend on the thread count at all.
TEST_CASE("DG assembly does not depend on the thread count", "[discretization][dg]")
{
  DiFfRG::Init();
  Setup<1, true, true, false, false> s(2, "0:0.00025:1");
  const auto r_all = s.residual(s.u);
  typename decltype(s)::SparseMatrixType J_all(s.assembler.get_sparsity_pattern_jacobian());
  J_all.copy_from(s.jacobian());
  std::optional<tbb::global_control> serial;
  serial.emplace(tbb::global_control::max_allowed_parallelism, 1);
  const auto r_one = s.residual(s.u);
  const auto &J_one = s.jacobian();
  serial.reset();

  auto deviation = r_all;
  deviation -= r_one;
  REQUIRE(max_abs(r_all) > 0.);
  REQUIRE(max_abs(deviation) == 0.);
  const auto [worst, scale] = matrix_deviation(J_all, J_one);
  REQUIRE(scale > 0.);
  REQUIRE(worst == 0.);
}
