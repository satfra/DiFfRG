#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "../boilerplate/lattice_kernel.hh"
#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat.hh>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

using namespace DiFfRG;

/**
 * The volume integrals in integrator_lat_{1..4}d.cc pass for any rule that visits the right number
 * of points, so they cannot tell a correct lattice sum from one that double-counts n = 0 and drops
 * n = N/2. This compares against the sum over the whole Brillouin zone, for a kernel that is
 * neither constant nor lattice-periodic, and with an odd q0 part wherever q0 is summed in full.
 */
TEST_CASE("Lattice sums match the full Brillouin zone", "[lattice][integration]")
{
  DiFfRG::Init();

  auto check = [](auto execution_space, auto type, auto dim) {
    using ExecutionSpace = decltype(execution_space);
    using NT = decltype(type);
    constexpr int d = decltype(dim)::value;
    using ctype = typename get_type::ctype<NT>;

    // Plain loops rather than GENERATE: this lambda is instantiated 24 times from one source line.
    for (const unsigned N_t : {2u, 4u, 10u})
      for (const unsigned N_s : {2u, 6u, 12u})
        for (const bool q0_symmetric : {false, true}) {
          const double a_t = 0.7, a_s = 0.4;
          const double c = q0_symmetric ? 0. : 0.05;

          std::array<uint, d == 1 ? 1 : 2> N;
          std::array<ctype, d == 1 ? 1 : 2> a;
          N[0] = N_t;
          a[0] = ctype(a_t);
          if constexpr (d > 1) {
            N[1] = N_s;
            a[1] = ctype(a_s);
          }
          IntegratorLat<d, NT, LatticeTestKernel<d, NT>, ExecutionSpace> integrator(N, a, q0_symmetric);

          NT result{};
          integrator.get(result, c, a_s);
          // The reference uses the spacings the integrator actually sees, i.e. rounded to ctype.
          const double reference =
              lattice_brute_force_sum(d, N_t, N_s, double(ctype(a_t)), double(ctype(a_s)), c);

          // Forward error bound of summing n positive terms; the rule this replaces was off by O(1/N).
          const double tolerance = std::max<double>(
              8., double(integrator.quadrature_volume())) * std::numeric_limits<ctype>::epsilon();
          INFO("d = " << d << ", N_t = " << N_t << ", N_s = " << N_s << ", q0_symmetric = " << q0_symmetric);
          INFO("result = " << double(result) << ", reference = " << reference);
          CHECK(std::abs(double(result) - reference) <= tolerance * std::abs(reference));
        }
  };

  auto all_dims = [&](auto execution_space, auto type) {
    check(execution_space, type, std::integral_constant<int, 1>{});
    check(execution_space, type, std::integral_constant<int, 2>{});
    check(execution_space, type, std::integral_constant<int, 3>{});
    check(execution_space, type, std::integral_constant<int, 4>{});
  };

  SECTION("TBB")
  {
    all_dims(TBB_exec(), double{});
    all_dims(TBB_exec(), float{});
  }
  SECTION("Kokkos host")
  {
    all_dims(KokkosHost_exec(), double{});
    all_dims(KokkosHost_exec(), float{});
  }
  SECTION("GPU")
  {
    all_dims(GPU_exec(), double{});
    all_dims(GPU_exec(), float{});
  }
}

TEST_CASE("Lattice integrators construct from the config", "[lattice][integration]")
{
  DiFfRG::Init();

  const ConfigTree config(json::value{
      {"integration", {{"lattice", {{"N_t", 6}, {"N_s", 4}, {"a_t", 0.7}, {"a_s", 0.4}, {"q0_symmetric", true}}}}}});
  QuadratureProvider quadrature_provider;

  auto check = [&](auto execution_space, auto dim) {
    using ExecutionSpace = decltype(execution_space);
    constexpr int d = decltype(dim)::value;
    using Integrator = IntegratorLat<d, double, LatticeTestKernel<d, double>, ExecutionSpace>;

    Integrator from_config(quadrature_provider, config);
    std::array<uint, d == 1 ? 1 : 2> N;
    std::array<double, d == 1 ? 1 : 2> a;
    N[0] = 6;
    a[0] = 0.7;
    if constexpr (d > 1) {
      N[1] = 4;
      a[1] = 0.4;
    }
    Integrator from_arrays(N, a, true);

    double x = 0., y = 0.;
    from_config.get(x, 0., 0.4);
    from_arrays.get(y, 0., 0.4);
    INFO("d = " << d);
    CHECK(std::memcmp(&x, &y, sizeof(double)) == 0);
    CHECK(from_config.quadrature_volume() == from_arrays.quadrature_volume());
  };

  check(TBB_exec(), std::integral_constant<int, 1>{});
  check(TBB_exec(), std::integral_constant<int, 4>{});
  check(GPU_exec(), std::integral_constant<int, 2>{});
  check(GPU_exec(), std::integral_constant<int, 3>{});
}
