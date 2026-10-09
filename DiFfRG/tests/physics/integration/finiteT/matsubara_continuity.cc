#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/integrator_fT_p2_1ang.hh>
#include <DiFfRG/physics/regulators.hh>

#include <algorithm>
#include <vector>

using namespace DiFfRG;

namespace
{
  /**
   * @brief The 4D-regulated bosonic summand, the shape every finite-T flow integrates.
   *
   * `dtR_B(k^2, Q^2) / (Q^2 + m^2 + R_B(k^2, Q^2))^2` with `Q^2 = q^2 + q0^2`. It carries NO
   * `matsubara_finite_extent` trait on purpose: that puts it on the Monien/vacuum path, which is
   * the one the handover lives on, and the one every kernel takes whose generator does not derive
   * the trait.
   *
   * The summand nevertheless *is* of finite extent, which is what lets this test build its own
   * reference: a second integrator with the exact sum forced on computes the sum itself.
   */
  struct RegulatedKernel {
    using Regulator = PolynomialExpRegulator<>;

    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double q, const double /*cos*/, const double q0,
                                                     const double k, const double m2)
    {
      const double k2 = powr<2>(k);
      const double Q2 = powr<2>(q) + powr<2>(q0);
      return Regulator::RBdot(k2, Q2) / powr<2>(Q2 + m2 + Regulator::RB(k2, Q2));
    }

    static KOKKOS_FORCEINLINE_FUNCTION double constant(const double, const double) { return 0.; }
  };

  /**
   * @brief A summand with a genuine algebraic tail in the frequency, and a scale BELOW k.
   *
   * The regulator argument carries only the spatial momentum, as it does for a 3D-regulated quark
   * loop, so in `q0` this is a double pole at `E = sqrt(q^2 + (mass_fraction k)^2)` with a
   * `1/q0^4` tail and no cutoff at all. That is the other analytic structure a finite-T flow
   * contains, and it fails differently: the compact case loses *aliasing* at the handover, this
   * one loses the *thermal content* of its lightest scale, `~4 exp(-E_min/T)`.
   *
   * `mass_fraction` sets that lightest scale, and the step at the handover is exponential in it:
   * 1.6e-9 at 0.1, 1.9e-7 at 0.075, 1.1e-3 at the 0.03 that is about as light as a regulated
   * propagator can plausibly get. The real QCD_Nf2/avatars_3Dquark kernels sit at 2.2e-6, so 0.075
   * puts this proxy in the right regime -- but the magnitude is a property of the summand, not of
   * the quadrature, which is why what is asserted below is the BOUND and not the value.
   */
  struct TailKernel {
    using Regulator = PolynomialExpRegulator<>;
    static constexpr double mass_fraction = 0.075;

    static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double q, const double /*cos*/, const double q0,
                                                     const double k, const double /*m2*/)
    {
      const double k2 = powr<2>(k), q2 = powr<2>(q);
      const double E2 = q2 + powr<2>(mass_fraction * k);
      return Regulator::RBdot(k2, q2) / powr<2>(powr<2>(q0) + E2);
    }

    static KOKKOS_FORCEINLINE_FUNCTION double constant(const double, const double) { return 0.; }
  };

  /// `x_extent` for PolynomialExpRegulator<8> at the default x_extent_tolerance; see optimize_x_extent().
  constexpr double x_extent = 1.5209;
} // namespace

/**
 * @brief The frequency rule must not put a step into the right-hand side of the flow.
 *
 * Along a flow at fixed T the rule changes as k/T crosses whatever the sizing logic decides on,
 * and every such change is a jump in the computed value unless the two rules agree. This walks k
 * finely across that band and compares the shipped choice against the exact Matsubara sum on the
 * same kernel -- the summand has finite extent, so forcing the exact sum on gives the truth, and
 * the test verifies that by pushing the extent margin out until the answer stops moving.
 *
 * This gate FAILS on the pre-2026-08 selection logic, which handed over to the vacuum rule at
 * typical_E/T = 30 on the argument that the thermal content there is ~4e-13. That law is the
 * aliasing law of a POLE, and a 4D-regulated summand has none (q^2 + R_B >= k^2 exactly); the
 * measured gap is 3.1e-4. See Part 13 of tests/common/quadrature/matsubara_convergence.cc.
 */
TEMPLATE_TEST_CASE("Matsubara rule selection is continuous in k", "[integration][quadrature][matsubara]", KokkosHost_exec,
                   GPU_exec)
{
  DiFfRG::Init();
  using ExecutionSpace = TestType;
  using Integrator = Integrator_fT_p2_1ang<4, double, RegulatedKernel, ExecutionSpace>;

  const double T = 0.05;
  const double m2 = GENERATE(0., -0.9); // in units of k^2; -0.9 is the hardest configuration
  const std::array<size_t, 2> grid{{96, 16}};

  QuadratureProvider qp;
  Integrator shipped(qp, grid, x_extent, T);

  // The reference needs its own, deliberately over-provisioned provider. The exact sum is only
  // *chosen* while it is cheaper than the Gaussian rule it is measured against, so both the node
  // ceiling and the reach have to be raised far enough that the comparison never flips -- else the
  // reference quietly becomes the finite-interval integral instead of the sum, which is the very
  // thing under test. reference_always_exact below is what makes that a check rather than a hope.
  QuadratureProvider qp_ref(
      ConfigTree::from_json_string(R"({"integration":{"max_matsubara_size":1024,"matsubara_precision_factor":8}})"));
  Integrator reference(qp_ref, grid, x_extent, T);
  // The reference: enumerate every mode the summand carries. The margin pushes the enumeration
  // past the radius where the summand is merely small rather than zero -- the convergence check
  // below is what makes that a proof and not an assumption.
  reference.set_allow_exact_matsubara_sum(true);
  reference.set_matsubara_extent_margin(1.5);
  Integrator reference_wide(qp_ref, grid, x_extent, T);
  reference_wide.set_allow_exact_matsubara_sum(true);
  reference_wide.set_matsubara_extent_margin(2.);

  shipped.set_T(T);
  reference.set_T(T);
  reference_wide.set_T(T);

  // From below the pre-2026-08 handover (k/T = 30) to past the budget one (k/T = 373), so the
  // sweep crosses whichever one the selection logic under test actually has.
  const double ratio_lo = 20., ratio_hi = 450., step = 1.004;

  std::vector<double> got, ref, ratios;
  std::vector<size_t> sizes;
  double max_reference_drift = 0.;
  bool reference_always_exact = true;

  for (double r = ratio_lo; r <= ratio_hi; r *= step) {
    const double k = r * T;
    shipped.set_k(k);
    reference.set_k(k);
    reference_wide.set_k(k);

    double a = 0., b = 0., c = 0.;
    shipped.get(a, k, m2 * powr<2>(k));
    reference.get(b, k, m2 * powr<2>(k));
    reference_wide.get(c, k, m2 * powr<2>(k));

    // Both references must really BE the exact sum. Past a certain margin the integrator prefers
    // the finite-interval rule over an exact sum that has grown expensive, and a reference that
    // quietly became an integral would be measuring the very thing under test.
    reference_always_exact &= reference.uses_exact_matsubara_sum() && reference_wide.uses_exact_matsubara_sum();
    max_reference_drift = std::max(max_reference_drift, std::abs(b - c) / std::abs(b));
    got.push_back(a);
    ref.push_back(b);
    ratios.push_back(r);
    sizes.push_back(shipped.get_matsubara_size());
  }

  INFO("reference drift between extent margins 1.5 and 2: " << max_reference_drift);
  REQUIRE(reference_always_exact);
  CHECK(max_reference_drift < 1e-13);

  // Falsification guard: a sweep that never changes the rule would pass everything below while
  // testing nothing. Require the frequency axis to actually be re-selected, and to shrink at
  // least once -- that shrink IS the handover to the integral.
  const bool size_changes = std::adjacent_find(sizes.begin(), sizes.end(), std::not_equal_to<>()) != sizes.end();
  const bool size_drops =
      std::adjacent_find(sizes.begin(), sizes.end(), [](size_t a, size_t b) { return b + 8 < a; }) != sizes.end();
  INFO("frequency nodes: " << sizes.front() << " ... " << sizes.back());
  REQUIRE(size_changes);
  REQUIRE(size_drops);

  // 1. Accuracy against the sum itself, everywhere in the band.
  double worst = 0.;
  size_t worst_i = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = std::abs(got[i] - ref[i]) / std::abs(ref[i]);
    if (e > worst) {
      worst = e;
      worst_i = i;
    }
  }
  INFO("worst rule-vs-exact-sum error " << worst << " at k/T = " << ratios[worst_i] << " (m2 = " << m2
                                        << " k^2, nodes = " << sizes[worst_i] << ")");
  CHECK(worst < 1e-7);

  // 2. Continuity: the step in the RESIDUAL between neighbouring k. Taking it against a smooth
  //    reference removes the integrand's own curvature in k, which a difference quotient of the
  //    value alone cannot separate from a rule change: f''/f is O(5) here, so at any usable step
  //    size the smooth background (8e-5) swamps the 3e-4 the old handover introduced.
  double worst_jump = 0.;
  size_t jump_i = 0;
  for (size_t i = 1; i < got.size(); ++i) {
    const double d = std::abs((got[i] / ref[i]) - (got[i - 1] / ref[i - 1]));
    if (d > worst_jump) {
      worst_jump = d;
      jump_i = i;
    }
  }
  INFO("worst residual step " << worst_jump << " at k/T = " << ratios[jump_i]);
  CHECK(worst_jump < 1e-7);
}

/**
 * @brief What the sum -> integral handover costs on a summand with an algebraic tail, and that the
 * node budget really is what bounds it.
 *
 * The test above uses a compactly supported summand, where the handover loses aliasing and the
 * step is 2.6e-8. That understates the mixed case: a 3D-regulated loop has a genuine pole in `q0`,
 * and switching to an integral discards its thermal content, `~4 exp(-E_min/T)`. On the real
 * QCD_Nf2/avatars_3Dquark kernels that is 2.2e-6 (ZA3) and up to 1.1e-5, against 5.8e-4 for the
 * pre-2026-08 handover at k/T = 30.
 *
 * The design's claim is not that this error is small but that it is bounded by
 * `max_matsubara_size`, which the user sets. That is what this checks: raising the ceiling until
 * the handover leaves the swept band takes the step to the noise floor. Measured the same way on
 * the real kernels -- at `max_matsubara_size = 192` every flow drops to 3e-14 ... 1e-12.
 *
 * There is no exact reference here (the summand does not vanish), so the reference is a provider
 * whose ceiling and reach are raised far enough that it never hands over and never clamps.
 */
TEMPLATE_TEST_CASE("Matsubara handover on an algebraic tail is bounded by the node budget",
                   "[integration][quadrature][matsubara]", KokkosHost_exec, GPU_exec)
{
  DiFfRG::Init();
  using ExecutionSpace = TestType;
  using Integrator = Integrator_fT_p2_1ang<4, double, TailKernel, ExecutionSpace>;

  const double T = 0.05;
  const std::array<size_t, 2> grid{{96, 16}};

  QuadratureProvider qp_default;
  QuadratureProvider qp_raised(ConfigTree::from_json_string(R"({"integration":{"max_matsubara_size":512}})"));
  QuadratureProvider qp_ref(
      ConfigTree::from_json_string(R"({"integration":{"max_matsubara_size":4096,"matsubara_precision_factor":8}})"));

  Integrator shipped(qp_default, grid, x_extent, T);
  Integrator raised(qp_raised, grid, x_extent, T);
  Integrator reference(qp_ref, grid, x_extent, T);
  shipped.set_T(T);
  raised.set_T(T);
  reference.set_T(T);

  std::vector<double> e_shipped, e_raised, ratios;
  std::vector<size_t> sizes;
  for (double r = 20.; r <= 450.; r *= 1.004) {
    const double k = r * T;
    shipped.set_k(k);
    raised.set_k(k);
    reference.set_k(k);

    double a = 0., b = 0., c = 0.;
    shipped.get(a, k, 0.);
    raised.get(b, k, 0.);
    reference.get(c, k, 0.);

    e_shipped.push_back(a / c - 1.);
    e_raised.push_back(b / c - 1.);
    ratios.push_back(r);
    sizes.push_back(shipped.get_matsubara_size());
  }

  const auto worst_step = [](const std::vector<double> &e, const std::vector<double> &r) {
    double worst = 0., where = 0.;
    for (size_t i = 1; i < e.size(); ++i)
      if (std::abs(e[i] - e[i - 1]) > worst) {
        worst = std::abs(e[i] - e[i - 1]);
        where = r[i];
      }
    return std::make_pair(worst, where);
  };

  const auto [step_shipped, at_shipped] = worst_step(e_shipped, ratios);
  const auto [step_raised, at_raised] = worst_step(e_raised, ratios);

  // Falsification guard: the shipped sweep must actually cross the handover, or both numbers below
  // are just the noise floor and the comparison says nothing.
  const bool size_drops =
      std::adjacent_find(sizes.begin(), sizes.end(), [](size_t a, size_t b) { return b + 8 < a; }) != sizes.end();
  INFO("frequency nodes: " << sizes.front() << " ... " << sizes.back());
  REQUIRE(size_drops);

  INFO("step at the shipped ceiling: " << step_shipped << " at k/T = " << at_shipped);
  INFO("step with the ceiling raised to 512: " << step_raised << " at k/T = " << at_raised);
  CHECK(step_shipped < 1e-4);
  // The point of the design: the budget bounds the step. Two decades of headroom over the shipped
  // ceiling has to buy at least two orders.
  CHECK(step_raised < step_shipped / 100.);
}
