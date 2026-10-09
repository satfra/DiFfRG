#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "../boilerplate/lattice_kernel.hh"
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/lattice/integrator_lat.hh>
#include <DiFfRG/physics/integration/map_completion.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

using namespace DiFfRG;

/**
 * The lattice integrators' map() under MapScheduler. As for the quadrature integrators, the contract
 * is *bitwise* equality with a serial run at any rank count: map_dist() skips the scheduler and is
 * the reference. Registered under mpirun at several rank counts by setup_mpi_test.
 */
namespace
{
  template <typename T> bool bitwise_equal(const std::vector<T> &a, const std::vector<T> &b)
  {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
  }

  template <int d, typename ctype> auto make_integrator_args()
  {
    std::array<uint, d == 1 ? 1 : 2> N;
    std::array<ctype, d == 1 ? 1 : 2> a;
    N[0] = 6;
    a[0] = ctype(0.7);
    if constexpr (d > 1) {
      N[1] = 4;
      a[1] = ctype(0.4);
    }
    return std::make_pair(N, a);
  }
} // namespace

TEST_CASE("Lattice map() is split across ranks and stays bitwise exact", "[integration][lattice][mpi]")
{
  DiFfRG::Init();

  const uint my_rank = DiFfRG::MPI::rank(MPI_COMM_WORLD);
  const uint n_ranks = DiFfRG::MPI::size(MPI_COMM_WORLD);
  const double c = 0.05, a_s = 0.4;

  auto check = [&](auto execution_space, auto type, auto dim) {
    using ExecutionSpace = decltype(execution_space);
    using NT = decltype(type);
    constexpr int d = decltype(dim)::value;
    using ctype = typename get_type::ctype<NT>;

    const auto [N, a] = make_integrator_args<d, ctype>();
    IntegratorLat<d, NT, LatticeTestKernel<d, NT>, ExecutionSpace> integrator(N, a, false);

    // 7 x 5 = 35 points: not a multiple of any rank count the suite runs at, and not square, so a
    // row/column mix-up cannot pass by symmetry.
    const CoordinatePackND coordinates(LinearCoordinates1D<ctype>(7, 0., 1.), LinearCoordinates1D<ctype>(5, 2., 3.));
    const size_t G = coordinates.size();

    std::vector<NT> reference(G, NT(0));
    integrator.map_dist(reference.data(), coordinates, c, a_s);
    flush_maps();

    // Small enough that the split is limited by the rank count, not the cost score.
    MapScheduler::instance().set_quantum(1.);

    std::vector<NT> distributed(G, NT(0));
    integrator.map(distributed.data(), coordinates, c, a_s);
    flush_maps();

    std::vector<NT> deferred(G, NT(0));
    {
      DeferredMaps defer;
      integrator.map(deferred.data(), coordinates, c, a_s);
    }

    MapScheduler::instance().set_quantum(0.);

    INFO("d = " << d << ", rank " << my_rank << " of " << n_ranks);
    CHECK(bitwise_equal(reference, distributed));
    CHECK(bitwise_equal(reference, deferred));

    // Each entry belongs to its own grid point: compare with get() at that point's position.
    for (size_t i = 0; i < G; ++i) {
      const auto p = coordinates.forward(coordinates.from_linear_index(i));
      NT expected{};
      integrator.get(expected, p[0], p[1], c, a_s);
      INFO("grid point " << i);
      CHECK(std::abs(double(expected) - double(distributed[i])) <=
            1e2 * std::numeric_limits<ctype>::epsilon() * std::abs(double(expected)));
    }
  };

  auto all_dims = [&](auto execution_space, auto type) {
    check(execution_space, type, std::integral_constant<int, 1>{});
    check(execution_space, type, std::integral_constant<int, 2>{});
    check(execution_space, type, std::integral_constant<int, 3>{});
    check(execution_space, type, std::integral_constant<int, 4>{});
  };

  SECTION("TBB") { all_dims(TBB_exec(), double{}); }
  SECTION("Kokkos host") { all_dims(KokkosHost_exec(), double{}); }
  SECTION("GPU")
  {
    all_dims(GPU_exec(), double{});
    all_dims(GPU_exec(), float{});
  }
}
