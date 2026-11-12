#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT.hh>

#include <cmath>
#include <iomanip>
#include <sstream>

using namespace DiFfRG;

namespace
{
  /**
   * @brief A summand with two scales, one light and one heavy: `1 / (((q0 + s)^2 + a^2) ((q0 + s)^2 + b^2))`.
   *
   * `s = 0` sums over the bosonic frequencies, `s = pi T` over the fermionic ones. This is the structure of a
   * finite-T loop with a light and a heavy line, e.g. a pion and a sigma, or a regulated mode near convexity
   * restoration next to a massive one: the light pole carries thermal content that only the sum over the lowest
   * modes resolves, the heavy one an algebraic tail out to `b`.
   */
  struct TwoScale {
    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double q0, const double a, const double b, const double s)
    {
      const double Q2 = powr<2>(q0 + s);
      return 1. / ((Q2 + powr<2>(a)) * (Q2 + powr<2>(b)));
    }

    static KOKKOS_FORCEINLINE_FUNCTION double constant(const double, const double, const double) { return 0.; }
  };

  /// `T sum_n 1 / (w_n^2 + m^2)`, bosonic or fermionic.
  double single_pole_sum(const double m, const double T, const bool fermionic)
  {
    return fermionic ? std::tanh(m / (2. * T)) / (2. * m) : 1. / (std::tanh(m / (2. * T)) * 2. * m);
  }

  /// The exact sum of TwoScale, by partial fractions.
  double two_scale_sum(const double a, const double b, const double T, const bool fermionic)
  {
    return (single_pole_sum(a, T, fermionic) - single_pole_sum(b, T, fermionic)) / (powr<2>(b) - powr<2>(a));
  }
} // namespace

/**
 * @brief The frequency rule must stay accurate when the summand has a scale below T and one far above it.
 *
 * The model reports the heavy scale through set_typical_E, as the docs ask for the reach, and the light one is k.
 * While the Monien rule sized for the heavy scale fits max_matsubara_size, the sum is computed and all is well. Past
 * that budget the selection hands over to the T = 0 integral, which discards the thermal content of the light pole:
 * at a ~ T that is an O(1) error, not a small one. Nothing in the rule choice compares k (or any light scale) with T.
 *
 * Grid: a/T from far below to far above 1, b/T inside the budget (50) and beyond it (300, 3000; the shipped budget
 * of 128 nodes reaches b/T ~ 187). Where every scale is far above T (a/T = 50) the integral is right and is also
 * the cheap choice; that it stays the choice there is checked as well, so a fix cannot buy accuracy with nodes.
 * Before the hybrid rule (exact low modes, integrated tail) the over-budget cases with a/T <= 5 were off by 1% to
 * 275%.
 */
TEMPLATE_TEST_CASE("Matsubara rule on a summand with a light and a heavy scale", "[integration][quadrature][matsubara]",
                   KokkosHost_exec, GPU_exec)
{
  DiFfRG::Init();
  using ExecutionSpace = TestType;
  using Integrator = Integrator_fT<1, double, TwoScale, ExecutionSpace>;

  const double T = GENERATE(1e-3, 1e-2);
  const bool fermionic = GENERATE(false, true);
  const double a_over_T = GENERATE(0.05, 0.5, 2., 5., 50.);
  const double b_over_T = GENERATE(50., 300., 3000.);

  // The light scale must lie below the heavy one (and a = b is 0/0 in the reference). A plain return, not SKIP():
  // Catch2 reports a test case with any skipped branch as skipped, which ctest counts as a failure.
  if (a_over_T >= b_over_T) return;
  const double a = a_over_T * T, b = b_over_T * T, s = fermionic ? M_PI * T : 0.;

  QuadratureProvider qp;
  Integrator integrator(qp, T, a);
  integrator.set_T(T);
  integrator.set_k(a); // the light scale: k is what the rule has to resolve below T
  integrator.set_typical_E(b);

  double got = 0.;
  integrator.get(got, a, b, s);
  const double ref = two_scale_sum(a, b, T, fermionic);
  const double rel_err = std::abs(got / ref - 1.);
  const size_t nodes = integrator.get_matsubara_size();

  std::ostringstream where;
  where << (fermionic ? "fermionic" : "bosonic") << ", T = " << T << ", a/T = " << a_over_T << ", b/T = " << b_over_T
        << ": rel. error " << std::scientific << std::setprecision(2) << rel_err << ", " << nodes << " frequency nodes";
  INFO(where.str());
  std::cout << where.str() << std::endl;

  CHECK(rel_err < 1e-8);

  // Cost: with every scale far above T the integral is exact to ~exp(-a/T) and costs only the vacuum rule's nodes.
  // A fix for the light-scale case must keep choosing it here.
  constexpr size_t vacuum_nodes = 64; // the default /integration/vacuum_quad_size
  if (a_over_T >= 50. && b_over_T > 200.) CHECK(nodes <= vacuum_nodes);

  // The budget holds everywhere: at most max_matsubara_size nodes plus the zero mode.
  CHECK(nodes <= size_t(qp.max_matsubara_size()) + 1);
  // The hybrid rule is what handles the light scale beyond the budget, and only there.
  CHECK(integrator.uses_hybrid_matsubara_rule() == (b_over_T > 200. && a_over_T < 30.));
}
