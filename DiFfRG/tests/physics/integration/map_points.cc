#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>

#include <random>

using namespace DiFfRG;

namespace
{
  // Every argument slot enters the result differently, so a mixed-up argument shows up.
  template <int dim, typename NT> struct PointKernel {
    using ctype = typename get_type::ctype<NT>;

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const ctype x, const auto &a, const auto &b, const auto &c)
      requires(dim == 1)
    {
      return a / (1. + b * x * x) + c * x;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const ctype x, const ctype y, const auto &a, const auto &b,
                                                   const auto &c)
      requires(dim == 2)
    {
      return a / (1. + b * (x * x + y * y)) + c * x * y * y;
    }

    static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto &a, const auto &b, const auto &c)
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

  /**
   * Run map_points with the argument pattern given by `vary` (which of a, b, c are per-point) and
   * compare every point against a get() with the same arguments. exact = bitwise equality.
   */
  template <int dim, typename OT, typename AT, typename Integrator>
  void compare_to_get(Integrator &integrator, const size_t n, const std::array<bool, 3> vary, const bool exact,
                      const double tolerance)
  {
    std::mt19937 rng(42 + n + 2 * vary[0] + 4 * vary[1] + 8 * vary[2]);
    std::array<std::vector<AT>, 3> values;
    for (int k = 0; k < 3; ++k) {
      values[k].resize(vary[k] ? n : 1);
      for (auto &v : values[k])
        v = make_value<AT>(rng, k == 1 ? 0.2 : -1., k == 1 ? 2. : 1.);
    }
    // Shared arguments as plain values, per-point ones as the std::vector itself.
    auto arg = [&](const int k) { return vary[k] ? PointArg<AT>(values[k]) : PointArg<AT>(values[k][0]); };

    std::vector<OT> mapped(n);
    integrator.map_points(PointSpan<OT>(mapped), arg(0), arg(1), arg(2));

    for (size_t i = 0; i < n; ++i) {
      OT reference{};
      integrator.get(reference, arg(0)[i], arg(1)[i], arg(2)[i]);
      if (exact) {
        REQUIRE(value_of(mapped[i]) == value_of(reference));
        REQUIRE(grad_of(mapped[i]) == grad_of(reference));
      } else {
        const double scale = 1. + std::abs(value_of(reference));
        REQUIRE(std::abs(value_of(mapped[i]) - value_of(reference)) <= tolerance * scale);
        REQUIRE(std::abs(grad_of(mapped[i]) - grad_of(reference)) <= tolerance * (1. + std::abs(grad_of(reference))));
      }
    }

    // Gate check: perturbing one point's argument changes that point, and only that point.
    if (n > 2 && (vary[0] || vary[1] || vary[2])) {
      const int k = vary[0] ? 0 : (vary[1] ? 1 : 2);
      const size_t j = n / 2;
      values[k][j] += AT(0.25);
      std::vector<OT> perturbed(n);
      integrator.map_points(PointSpan<OT>(perturbed), arg(0), arg(1), arg(2));
      for (size_t i = 0; i < n; ++i)
        if (i == j)
          REQUIRE(value_of(perturbed[i]) != value_of(mapped[i]));
        else
          REQUIRE(value_of(perturbed[i]) == value_of(mapped[i]));
    }
  }

  template <int dim, typename NT, typename ExecutionSpace> auto make_integrator(QuadratureProvider &provider)
  {
    using ctype = typename get_type::ctype<NT>;
    std::array<size_t, dim> grid_size;
    std::array<ctype, dim> lo, hi;
    std::array<QuadratureType, dim> types;
    for (int d = 0; d < dim; ++d) {
      grid_size[d] = d == 0 ? 48 : 20;
      lo[d] = -1;
      hi[d] = 1.5;
      types[d] = QuadratureType::legendre;
    }
    return QuadratureIntegrator<dim, NT, PointKernel<dim, NT>, ExecutionSpace>(provider, grid_size, lo, hi, types);
  }

  const std::vector<std::array<bool, 3>> patterns = {{false, false, false}, {true, false, false}, {false, true, false},
                                                     {false, false, true},  {true, true, true},   {true, false, true}};
} // namespace

TEMPLATE_TEST_CASE_SIG("map_points matches a loop of get()", "[integration][quadrature][map_points]", ((int dim), dim),
                       (1), (2))
{
  DiFfRG::Init();
  QuadratureProvider provider;

  const size_t n = GENERATE(1, 37, 3000);

  SECTION("TBB evaluates exactly get()")
  {
    auto integrator_d = make_integrator<dim, double, TBB_exec>(provider);
    auto integrator_ad = make_integrator<dim, autodiff::real, TBB_exec>(provider);
    for (const auto &vary : patterns) {
      compare_to_get<dim, double, double>(integrator_d, n, vary, true, 0.);
      // Same get(), but -ffp-contract=fast may contract the inlined AD arithmetic differently at the
      // two call sites: the gradient can differ in the last bit.
      compare_to_get<dim, autodiff::real, autodiff::real>(integrator_ad, n, vary, false, 1e-15);
    }
  }

  auto kokkos_checks = [&](auto execution_space) {
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;
    auto integrator_d = make_integrator<dim, double, ExecutionSpace>(provider);
    auto integrator_ad = make_integrator<dim, autodiff::real, ExecutionSpace>(provider);
    auto integrator_f = make_integrator<dim, float, ExecutionSpace>(provider);
    for (const auto policy : {MapPointsPolicy::thread_per_point, MapPointsPolicy::team_per_point}) {
      integrator_d.set_map_points_policy(policy);
      integrator_ad.set_map_points_policy(policy);
      integrator_f.set_map_points_policy(policy);
      for (const auto &vary : patterns) {
        compare_to_get<dim, double, double>(integrator_d, n, vary, false, 1e-13);
        compare_to_get<dim, autodiff::real, autodiff::real>(integrator_ad, n, vary, false, 1e-13);
        // The float integrator receives float arguments, get() the double ones it was called with.
        compare_to_get<dim, double, double>(integrator_f, n, vary, false, 1e-5);
      }
    }
  };

  SECTION("Kokkos host") { kokkos_checks(KokkosHost_exec()); }
  SECTION("GPU") { kokkos_checks(GPU_exec()); }
}

TEST_CASE("map_points rejects a per-point argument of the wrong size", "[integration][quadrature][map_points]")
{
  DiFfRG::Init();
  QuadratureProvider provider;
  auto integrator = make_integrator<1, double, GPU_exec>(provider);
  auto integrator_tbb = make_integrator<1, double, TBB_exec>(provider);
  std::vector<double> short_array(4, 1.), long_array(9, 1.), dest(8);
  CHECK_THROWS(integrator.map_points(PointSpan<double>(dest), short_array, 1., 1.));
  CHECK_THROWS(integrator_tbb.map_points(PointSpan<double>(dest), short_array, 1., 1.));
  CHECK_THROWS(integrator.map_points(PointSpan<double>(dest), long_array, 1., 1.));
  CHECK_THROWS(integrator_tbb.map_points(PointSpan<double>(dest), long_array, 1., 1.));
}

TEST_CASE("map_points converts arguments to the kernel's argument type", "[integration][quadrature][map_points]")
{
  DiFfRG::Init();
  QuadratureProvider provider;
  auto integrator = make_integrator<1, autodiff::real, TBB_exec>(provider);
  const size_t n = 5;
  std::vector<double> a(n);
  for (size_t i = 0; i < n; ++i)
    a[i] = 0.1 * i;
  std::vector<autodiff::real> a_ad(a.begin(), a.end()), mapped(n), reference(n);
  // A double column and a double scalar reach AD-typed arguments.
  const auto args = [&](const auto &first) {
    return std::make_tuple(PointArg<autodiff::real>(first), PointArg<autodiff::real>(0.7),
                           PointArg<autodiff::real>(-0.3));
  };
  std::apply([&](const auto &...p) { integrator.map_points(PointSpan<autodiff::real>(mapped), p...); },
             args(PointSpan<const double>(a)));
  std::apply([&](const auto &...p) { integrator.map_points(PointSpan<autodiff::real>(reference), p...); }, args(a_ad));
  for (size_t i = 0; i < n; ++i)
    REQUIRE(value_of(mapped[i]) == value_of(reference[i]));
}
