#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_1ang.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

#include <vector>

using namespace DiFfRG;

namespace
{
  // A summand with finite extent in the frequency: a super-exponential cut at |q0| ~ R.
  template <typename ctype> KOKKOS_FORCEINLINE_FUNCTION ctype confined(const ctype x, const ctype q0)
  {
    using Kokkos::exp;
    const ctype u = q0 / ctype(2);
    return exp(-powr<4>(u * u)) * (ctype(1) + x * x) / (ctype(1) + q0 * q0);
  }
  // ...and one with a genuine algebraic tail, which only the Gaussian rule can do.
  template <typename ctype> KOKKOS_FORCEINLINE_FUNCTION ctype unbounded(const ctype x, const ctype q0)
  {
    return (ctype(2) + x) / (ctype(1) + q0 * q0);
  }

  /// The unsplit reference: one entry point, the sum of both halves.
  struct WholeKernel {
    template <typename... T>
    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double x, const double q0, const T &...)
    {
      return confined(x, q0) + unbounded(x, q0);
    }
    template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION double constant(const T &...) { return 0.; }
  };

  /// The same integrand, offered as two entry points. `kernel` stays the whole thing, exactly as
  /// NumTracer emits it, so a consumer without the split machinery is unaffected.
  struct SplitKernel {
    static constexpr bool matsubara_split = true;
    template <typename... T>
    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double x, const double q0, const T &...)
    {
      return confined(x, q0) + unbounded(x, q0);
    }
    template <typename... T>
    static KOKKOS_FORCEINLINE_FUNCTION double kernel_finite_extent(const double x, const double q0, const T &...)
    {
      return confined(x, q0);
    }
    template <typename... T>
    static KOKKOS_FORCEINLINE_FUNCTION double kernel_tail(const double x, const double q0, const T &...)
    {
      return unbounded(x, q0);
    }
    template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION double constant(const T &...) { return 0.; }
  };
} // namespace

TEMPLATE_TEST_CASE("Split Matsubara kernel reproduces the unsplit one", "[integration][quadrature][matsubara-split]",
                   KokkosHost_exec, GPU_exec)
{
  DiFfRG::Init();
  using ExecutionSpace = TestType;

  const std::array<size_t, 1> grid_size{{32}};
  const std::array<double, 1> ext_min{{-1.}};
  const std::array<double, 1> ext_max{{1.}};
  const std::array<QuadratureType, 1> quad_type{{QuadratureType::legendre}};

  const double T = GENERATE(0.05, 0.3, 1.0);

  QuadratureProvider qp;
  QuadratureIntegrator_fT<2, double, WholeKernel, ExecutionSpace> whole(qp, grid_size, ext_min, ext_max, quad_type, T);
  QuadratureIntegrator_fT<2, double, SplitKernel, ExecutionSpace> split(qp, grid_size, ext_min, ext_max, quad_type, T);

  double ref = 0., got = 0.;
  whole.get(ref);
  split.get(got);

  CAPTURE(T, whole.get_matsubara_size(), split.get_matsubara_size(), split.uses_exact_matsubara_sum());
  // No frequency cutoff has been set, so the split half runs on the SAME Gaussian rule as the tail
  // half: the two must then agree to roundoff, which is what proves the concatenated axis and the
  // per-node dispatch are wired up right, independently of any quadrature question.
  REQUIRE_THAT(got, Catch::Matchers::WithinRel(ref, 1e-13));

  SECTION("map() agrees too -- the concatenated axis has to survive the cache + team reduction")
  {
    // get() reduces inside one parallel_reduce; map() writes every node into a cache and sums it in
    // a second, team-parallel pass whose strides come from grid_size. Doubling the frequency axis
    // is exactly the kind of change that can go wrong only in the second path, so it gets its own
    // check rather than riding on get().
    const size_t rsize = 16;
    LinearCoordinates1D<double> coordinates(rsize, 0., 1.);
    std::vector<double> ref_m(rsize, 0.), got_m(rsize, 0.);
    whole.map(ref_m.data(), coordinates);
    split.map(got_m.data(), coordinates);
    double worst = 0.;
    for (size_t i = 0; i < rsize; ++i)
      worst = std::max(worst, std::abs(ref_m[i] - got_m[i]) / std::max(std::abs(ref_m[i]), 1e-300));
    CAPTURE(T, worst, ref_m[0], got_m[0], split.get_matsubara_size());
    REQUIRE(worst < 1e-12);
  }

  SECTION("...and still agrees once the finite-extent half moves to the exact sum")
  {
    // R = 12 is far outside the confined half's support (exp(-(R/2)^8) is dead), so switching that
    // half to the exact sum must not move the answer beyond the Gaussian rule's own error.
    split.set_frequency_cutoff(12.);
    split.set_allow_exact_matsubara_sum(true);
    double exact = 0.;
    split.get(exact);
    CAPTURE(split.get_matsubara_size(), split.uses_exact_matsubara_sum());
    REQUIRE_THAT(exact, Catch::Matchers::WithinRel(ref, 1e-7));
  }
}


// The layer the tests above skip: the measure adapter (Transform_fT_p2_1ang) has to forward BOTH
// the split trait and the two entry points, and the wrapper has to keep feeding the frequency
// cutoff. This is the shape every real flow uses.
namespace
{
  struct WholeKernel2 {
    template <typename... T>
    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double q, const double c, const double q0, const T &...)
    {
      return confined(c, q0) + unbounded(c, q0);
    }
    template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION double constant(const T &...) { return 0.; }
  };
  struct SplitKernel2 {
    static constexpr bool matsubara_split = true;
    template <typename... T>
    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double q, const double c, const double q0, const T &...)
    {
      return confined(c, q0) + unbounded(c, q0);
    }
    template <typename... T>
    static KOKKOS_FORCEINLINE_FUNCTION double kernel_finite_extent(const double q, const double c, const double q0,
                                                                  const T &...)
    {
      return confined(c, q0);
    }
    template <typename... T>
    static KOKKOS_FORCEINLINE_FUNCTION double kernel_tail(const double q, const double c, const double q0, const T &...)
    {
      return unbounded(c, q0);
    }
    template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION double constant(const T &...) { return 0.; }
  };
} // namespace

TEMPLATE_TEST_CASE("Split survives the measure adapter and the p2_1ang wrapper",
                   "[integration][quadrature][matsubara-split]", KokkosHost_exec, GPU_exec)
{
  DiFfRG::Init();
  using ExecutionSpace = TestType;

  const double T = GENERATE(0.05, 0.3);
  const std::array<size_t, 2> grid_size{{16, 8}};

  QuadratureProvider qp;
  Integrator_fT_p2_1ang<4, double, WholeKernel2, ExecutionSpace> whole(qp, grid_size, 1.52, T, 1.);
  Integrator_fT_p2_1ang<4, double, SplitKernel2, ExecutionSpace> split(qp, grid_size, 1.52, T, 1.);

  whole.set_k(3.);
  split.set_k(3.);

  double ref = 0.;
  whole.get(ref, 1.0, 3.0);

  SECTION("both halves on the same rule: an identity, so it must hold to roundoff")
  {
    // Isolates the plumbing from the quadrature question. The wrapper has already fed a frequency
    // cutoff via set_k(), so the exact sum has to be turned off explicitly to get this comparison.
    split.set_allow_exact_matsubara_sum(false);
    double got = 0.;
    split.get(got, 1.0, 3.0);
    CAPTURE(T, ref, got, split.get_matsubara_size(), split.uses_exact_matsubara_sum());
    REQUIRE_THAT(got, Catch::Matchers::WithinRel(ref, 1e-12));
  }

  SECTION("finite-extent half on the exact sum: agrees to the Gaussian rule's own error")
  {
    // Now the two rules genuinely differ, and the exact sum is the accurate one -- so this checks
    // that the cutoff reached the integrator and that the answer did not MOVE, not that it is
    // bit-identical. The gap is the GAUSSIAN rule's error (~1e-6 here): this integrand's confined
    // half has a super-exponential cut, which is exactly the shape a rule built for an algebraic
    // tail resolves worst, so the tolerance is loose on purpose.
    double got = 0.;
    split.get(got, 1.0, 3.0);
    CAPTURE(T, ref, got, split.get_matsubara_size(), split.uses_exact_matsubara_sum());
    REQUIRE(split.uses_exact_matsubara_sum());
    REQUIRE_THAT(got, Catch::Matchers::WithinRel(ref, 1e-5));
  }
}
