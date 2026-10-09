#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

#include <random>

using namespace DiFfRG;

namespace
{
  // A summand of finite extent in the frequency (it vanishes for |q0| >= 1), and one with an algebraic tail.
  template <typename ctype> KOKKOS_FORCEINLINE_FUNCTION ctype bump(const ctype q0)
  {
    const ctype s = ctype(1) - q0 * q0;
    return s > ctype(0) ? s * s : ctype(0);
  }

  enum class Kind { plain, even, finite_extent, split };

  /**
   * Every argument slot enters the result differently, so a mixed-up argument shows up. `plain` has a part odd in
   * the frequency, which only the explicit +-frequency sum gets right; `even` declares matsubara_even;
   * `finite_extent` vanishes beyond |q0| = 1 and so may run on the exact sum; `split` offers both halves.
   */
  template <Kind kind> struct PointKernelT {
    static constexpr bool matsubara_even = kind == Kind::even;
    static constexpr bool matsubara_finite_extent = kind == Kind::finite_extent;
    static constexpr bool matsubara_split = kind == Kind::split;

    template <typename A, typename B, typename C>
    static KOKKOS_FORCEINLINE_FUNCTION auto tail(const double x, const double q0, const A &a, const B &b, const C &c)
    {
      if constexpr (kind == Kind::plain)
        return a / (1. + b * (x * x + q0 * q0)) + c * x * q0;
      else
        return a / (1. + b * (x * x + q0 * q0));
    }
    template <typename A, typename B, typename C>
    static KOKKOS_FORCEINLINE_FUNCTION auto confined(const double x, const double q0, const A &a, const B &b,
                                                     const C &c)
    {
      return (c + a * x) * bump(q0) / (1. + b);
    }

    template <typename A, typename B, typename C>
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double x, const double q0, const A &a, const B &b, const C &c)
    {
      if constexpr (kind == Kind::finite_extent)
        return confined(x, q0, a, b, c);
      else if constexpr (kind == Kind::split)
        return tail(x, q0, a, b, c) + confined(x, q0, a, b, c);
      else
        return tail(x, q0, a, b, c);
    }
    template <typename A, typename B, typename C>
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel_tail(const double x, const double q0, const A &a, const B &b,
                                                        const C &c)
    {
      return tail(x, q0, a, b, c);
    }
    template <typename A, typename B, typename C>
    static KOKKOS_FORCEINLINE_FUNCTION auto kernel_finite_extent(const double x, const double q0, const A &a,
                                                                 const B &b, const C &c)
    {
      return confined(x, q0, a, b, c);
    }
    template <typename A, typename B, typename C>
    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const A &a, const B &b, const C &c)
    {
      return c * c + a * b;
    }
  };

  template <typename T> double value_of(const T &v)
  {
    if constexpr (get_type::is_autodiff<T>)
      return double(v[0]);
    else
      return double(v);
  }
  template <typename T> double grad_of(const T &v)
  {
    if constexpr (get_type::is_autodiff<T>)
      return double(v[1]);
    else
      return 0.;
  }

  template <typename NT> NT make_value(std::mt19937 &rng, const double lo, const double hi)
  {
    std::uniform_real_distribution<double> dist(lo, hi);
    if constexpr (get_type::is_autodiff<NT>) {
      NT v = dist(rng);
      v[1] = dist(rng);
      return v;
    } else
      return NT(dist(rng));
  }

  /// map_points with per-point arguments where `vary` says so, against get() at every point; then the gate: one
  /// perturbed point changes that point only.
  template <typename NT, typename Integrator>
  void compare_to_get(Integrator &integrator, const size_t n, const std::array<bool, 3> vary, const double tolerance)
  {
    std::mt19937 rng(7 + n + 2 * vary[0] + 4 * vary[1] + 8 * vary[2]);
    std::array<std::vector<NT>, 3> values;
    for (int k = 0; k < 3; ++k) {
      values[k].resize(vary[k] ? n : 1);
      for (auto &v : values[k])
        v = make_value<NT>(rng, k == 1 ? 0.2 : -1., k == 1 ? 2. : 1.);
    }
    auto arg = [&](const int k) { return vary[k] ? PointArg<NT>(values[k]) : PointArg<NT>(values[k][0]); };

    std::vector<NT> mapped(n);
    integrator.map_points(PointSpan<NT>(mapped), arg(0), arg(1), arg(2));
    for (size_t i = 0; i < n; ++i) {
      NT reference{};
      integrator.get(reference, arg(0)[i], arg(1)[i], arg(2)[i]);
      CAPTURE(i, value_of(mapped[i]), value_of(reference));
      REQUIRE(std::abs(value_of(mapped[i]) - value_of(reference)) <= tolerance * (1. + std::abs(value_of(reference))));
      REQUIRE(std::abs(grad_of(mapped[i]) - grad_of(reference)) <= tolerance * (1. + std::abs(grad_of(reference))));
    }

    if (n > 2 && (vary[0] || vary[1] || vary[2])) {
      const int k = vary[0] ? 0 : (vary[1] ? 1 : 2);
      const size_t j = n / 2;
      values[k][j] += NT(0.25);
      std::vector<NT> perturbed(n);
      integrator.map_points(PointSpan<NT>(perturbed), arg(0), arg(1), arg(2));
      for (size_t i = 0; i < n; ++i)
        if (i == j)
          REQUIRE(value_of(perturbed[i]) != value_of(mapped[i]));
        else
          REQUIRE(value_of(perturbed[i]) == value_of(mapped[i]));
    }
  }

  const std::vector<std::array<bool, 3>> patterns = {
      {false, false, false}, {true, false, false}, {false, false, true}, {true, true, true}};

  /// One integrator per kernel kind, number type and backend; T selects the frequency rule.
  template <Kind kind, typename NT, typename ExecutionSpace>
  void check(QuadratureProvider &provider, const double T, const size_t n, const MapPointsPolicy policy,
             const double tolerance)
  {
    QuadratureIntegrator_fT<2, NT, PointKernelT<kind>, ExecutionSpace> integrator(provider, {{24}}, {{0.}}, {{1.5}},
                                                                                  {{QuadratureType::legendre}}, T);
    integrator.set_k(1.);
    // Lets the finite-extent kinds run on the exact sum where it is cheaper than the Gaussian rule.
    integrator.set_frequency_cutoff(1.);
    if constexpr (requires { integrator.set_map_points_policy(policy); }) integrator.set_map_points_policy(policy);
    CAPTURE(int(kind), T, n, int(policy), integrator.get_matsubara_size(), integrator.uses_exact_matsubara_sum());
    // The low temperature runs the finite-extent kind on the exact sum, so that path is covered too.
    if (kind == Kind::finite_extent && T == 0.02) REQUIRE(integrator.uses_exact_matsubara_sum());
    for (const auto &vary : patterns)
      compare_to_get<NT>(integrator, n, vary, tolerance);
  }

  template <typename NT, typename ExecutionSpace>
  void check_all_kinds(QuadratureProvider &provider, const double T, const size_t n, const MapPointsPolicy policy,
                       const double tolerance)
  {
    check<Kind::plain, NT, ExecutionSpace>(provider, T, n, policy, tolerance);
    check<Kind::even, NT, ExecutionSpace>(provider, T, n, policy, tolerance);
    check<Kind::finite_extent, NT, ExecutionSpace>(provider, T, n, policy, tolerance);
    check<Kind::split, NT, ExecutionSpace>(provider, T, n, policy, tolerance);
  }
} // namespace

TEST_CASE("Finite-T map_points matches a loop of get()", "[integration][quadrature][map_points][finiteT]")
{
  DiFfRG::Init();
  QuadratureProvider provider;

  // T = 0.02: exact sum for the finite-extent kinds; T = 0.3: Monien rule; T = 0: the vacuum rule.
  const double T = GENERATE(0.02, 0.3, 0.);
  const size_t n = GENERATE(1, 37, 700);

  SECTION("TBB")
  {
    // One get() per point: equal up to the last bit of the inlined arithmetic (-ffp-contract=fast).
    check_all_kinds<double, TBB_exec>(provider, T, n, MapPointsPolicy::automatic, 1e-15);
    check_all_kinds<autodiff::real, TBB_exec>(provider, T, n, MapPointsPolicy::automatic, 1e-15);
  }

  auto kokkos_checks = [&](auto execution_space) {
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;
    for (const auto policy : {MapPointsPolicy::thread_per_point, MapPointsPolicy::team_per_point}) {
      check_all_kinds<double, ExecutionSpace>(provider, T, n, policy, 1e-13);
      check_all_kinds<autodiff::real, ExecutionSpace>(provider, T, n, policy, 1e-13);
    }
  };
  SECTION("Kokkos host") { kokkos_checks(KokkosHost_exec()); }
  SECTION("GPU") { kokkos_checks(GPU_exec()); }
}

TEST_CASE("Finite-T map_points rejects a per-point argument of the wrong size",
          "[integration][quadrature][map_points][finiteT]")
{
  DiFfRG::Init();
  QuadratureProvider provider;
  QuadratureIntegrator_fT<2, double, PointKernelT<Kind::plain>, GPU_exec> integrator(provider, {{24}}, {{0.}}, {{1.5}},
                                                                                     {{QuadratureType::legendre}}, 0.3);
  QuadratureIntegrator_fT<2, double, PointKernelT<Kind::plain>, TBB_exec> integrator_tbb(
      provider, {{24}}, {{0.}}, {{1.5}}, {{QuadratureType::legendre}}, 0.3);
  std::vector<double> short_array(4, 1.), dest(8);
  CHECK_THROWS(integrator.map_points(PointSpan<double>(dest), short_array, 1., 1.));
  CHECK_THROWS(integrator_tbb.map_points(PointSpan<double>(dest), short_array, 1., 1.));
}
