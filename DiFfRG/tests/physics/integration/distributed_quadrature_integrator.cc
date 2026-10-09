#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "boilerplate/poly_integrand.hh"
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/polynomials.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/map_completion.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>
#include <DiFfRG/physics/interpolation.hh>

#include <cstring>

using namespace DiFfRG;

/**
 * The contract of the MPI path is *bitwise* equality with a serial run, at any rank count. That is
 * not a stylistic preference: the flow kernels feed a stiff DAE, where a last-bit change in a
 * residual changes the accepted step sequence and therefore the whole trajectory. So this test
 * compares raw bit patterns, never a tolerance.
 *
 * `map()` goes through MapScheduler, `map_dist()` does not, which gives us the serial reference
 * inside the same process and the same execution space.
 */
namespace
{
  template <typename T> bool bitwise_equal(const std::vector<T> &a, const std::vector<T> &b)
  {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
  }
} // namespace

TEST_CASE("Test distributed integration", "[integration][quadrature][mpi]")
{
  DiFfRG::Init();

  constexpr int dim = 2;

  const uint n_ranks = DiFfRG::MPI::size(MPI_COMM_WORLD);
  const uint my_rank = DiFfRG::MPI::rank(MPI_COMM_WORLD);

  auto check = [&](auto execution_space, auto type) {
    using NT = std::decay_t<decltype(type)>;
    using ctype = typename get_type::ctype<NT>;
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;

    constexpr size_t size = 16;

    std::array<ctype, dim> ext_min;
    std::array<ctype, dim> ext_max;
    std::array<size_t, dim> grid_size;
    std::array<QuadratureType, dim> quad_type;

    // Every rank must integrate the *same* thing, so no randomness here: a random extent drawn per
    // rank would make the ranks disagree and the comparison meaningless.
    for (uint d = 0; d < dim; ++d) {
      ext_min[d] = ctype(-1.5);
      ext_max[d] = ctype(1.25);
      grid_size[d] = size;
      quad_type[d] = QuadratureType::legendre;
    }

    QuadratureProvider quadrature_provider;
    QuadratureIntegrator<dim, NT, PolyIntegrand<dim, NT>, ExecutionSpace> integrator(quadrature_provider, grid_size,
                                                                                     ext_min, ext_max, quad_type);

    std::array<NT, 4 * dim> coeffs{};
    for (uint d = 0; d < dim; ++d)
      coeffs[4 * d] = 1;

    constexpr uint rsize = 32;
    LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

    // Serial reference, in the same process and the same execution space: the overloads that take
    // an explicit destination without going through MapScheduler.
    std::vector<NT> reference(rsize, NT(0));
    if constexpr (std::is_same_v<ExecutionSpace, TBB_exec>) {
      auto space = ExecutionSpace();
      std::apply([&](auto... c) { integrator.map(space, reference.data(), coordinates, c...); }, coeffs);
    } else {
      std::apply([&](auto... c) { integrator.map_dist(reference.data(), coordinates, c...).fence(); }, coeffs);
    }
    flush_maps();

    // A quantum this small forces the scheduler to split even this toy grid, so the test exercises
    // the partitioning rather than falling back to "one rank owns everything".
    MapScheduler::instance().set_quantum(1.);

    std::vector<NT> distributed(rsize, NT(0));
    std::apply([&](auto... c) { integrator.map(distributed.data(), coordinates, c...).fence(); }, coeffs);
    flush_maps();

    INFO("rank " << my_rank << " of " << n_ranks);
    CHECK(bitwise_equal(reference, distributed));

    // Sanity: the values are actually the integral, not zeros that happen to match.
    NT volume = 1;
    for (uint d = 0; d < dim; ++d)
      volume *= (ext_max[d] - ext_min[d]);
    for (uint i = 0; i < rsize; ++i) {
      const auto expected = coordinates.forward(i) + volume;
      using Kokkos::abs;
      CHECK(abs(expected - distributed[i]) <= 10 * expected_precision<ctype>::value * abs(expected));
    }
  };

  SECTION("TBB integrals")
  {
    check(TBB_exec(), (double)0);
    check(TBB_exec(), (float)0);
  };
  SECTION("Threads integrals")
  {
    check(KokkosHost_exec(), (double)0);
    check(KokkosHost_exec(), (float)0);
  };
  SECTION("GPU integrals")
  {
    check(GPU_exec(), (double)0);
    check(GPU_exec(), (float)0);
  };
}

TEST_CASE("Test the map schedule itself", "[integration][quadrature][mpi]")
{
  DiFfRG::Init();

  auto &scheduler = MapScheduler::instance();
  const uint n_ranks = scheduler.n_ranks();

  SECTION("A cheap flow is assigned whole, not chopped into slivers")
  {
    scheduler.set_quantum(5e4);

    // G * Q = 64 * 192, well under the quantum: this must land on a single rank.
    std::vector<double> sink(64, 0.);
    const MapSlice slice = scheduler.schedule(1000, sink.data(), sizeof(double), 64, 192, true, map_target<GPU_exec>());
    CHECK((slice.count == 0 || slice.count == 64));

    // Exactly one rank may own it.
    const int owners = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, slice.count > 0 ? 1 : 0);
    CHECK(owners == 1);

    flush_maps(); // completes (and clears) the plan
  }

  SECTION("An expensive flow is spread, and the slices tile the grid exactly")
  {
    scheduler.set_quantum(5e4);

    // G * Q = 64 * 6912 = 442368, i.e. r = 8 before the rank-count clamp.
    std::vector<double> sink(64, 0.);
    const MapSlice slice = scheduler.schedule(1001, sink.data(), sizeof(double), 64, 6912, true, map_target<GPU_exec>());

    const int covered = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, static_cast<int>(slice.count));
    CHECK(covered == 64);

    const int owners = DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, slice.count > 0 ? 1 : 0);
    CHECK(owners == static_cast<int>(std::min<uint>(n_ranks, 8)));

    flush_maps();
  }

  SECTION("A coordinate system flagged unsplittable is assigned whole")
  {
    scheduler.set_quantum(1.);

    std::vector<double> sink(64, 0.);
    const MapSlice slice = scheduler.schedule(1002, sink.data(), sizeof(double), 64, 6912, false, map_target<GPU_exec>());
    CHECK((slice.count == 0 || slice.count == 64));

    flush_maps();
  }
}

namespace
{
  /**
   * A kernel whose *external* grid is two-dimensional, which is what the split used to refuse to
   * touch. The quadrature stays 1D so the arity works out: map() passes the two external positions
   * ahead of the caller's arguments, to both constant() and kernel().
   */
  template <typename T> struct External2DIntegrand {
    using ctype = typename get_type::ctype<T>;
    static KOKKOS_INLINE_FUNCTION T kernel(const ctype q, const ctype /*px*/, const ctype /*py*/, const T m)
    {
      using DiFfRG::powr;
      return m * (1 + powr<2>(q));
    }
    static KOKKOS_INLINE_FUNCTION T constant(const ctype px, const ctype py, const T /*m*/)
    {
      // Distinct per grid point and cheap to predict, so a slice landing at the wrong offset shows
      // up as a value from the wrong point rather than as a plausible number.
      return T(px) + T(100) * T(py);
    }
  };
} // namespace

TEST_CASE("A two-dimensional external grid is split, and stays bitwise exact",
          "[integration][quadrature][mpi]")
{
  DiFfRG::Init();

  // Before SubCoordinates became a linear index window this could not happen at all: schedule() was
  // told `splittable = (Coordinates::dim == 1)`, so a 2D external grid went whole to one rank. The
  // window it would have produced -- a per-axis box derived from both ends of the linear range --
  // is a different set of points, so the check below is what that restriction was protecting.
  const uint my_rank = DiFfRG::MPI::rank(MPI_COMM_WORLD);
  const uint n_ranks = DiFfRG::MPI::size(MPI_COMM_WORLD);

  auto check = [&](auto execution_space, auto type) {
    using NT = std::decay_t<decltype(type)>;
    using ctype = typename get_type::ctype<NT>;
    using ExecutionSpace = std::decay_t<decltype(execution_space)>;

    constexpr int dim = 1;
    const std::array<size_t, dim> grid_size{{16}};
    const std::array<ctype, dim> ext_min{{ctype(-1.5)}};
    const std::array<ctype, dim> ext_max{{ctype(1.25)}};
    const std::array<QuadratureType, dim> quad_type{{QuadratureType::legendre}};

    QuadratureProvider quadrature_provider;
    QuadratureIntegrator<dim, NT, External2DIntegrand<NT>, ExecutionSpace> integrator(
        quadrature_provider, grid_size, ext_min, ext_max, quad_type);

    // 7 x 5 = 35: deliberately not a multiple of any rank count the suite runs at, and not a square,
    // so a row/column mix-up cannot pass by symmetry.
    const CoordinatePackND coordinates(LinearCoordinates1D<ctype>(7, 0., 1.),
                                       LinearCoordinates1D<ctype>(5, 2., 3.));
    STATIC_REQUIRE(std::decay_t<decltype(coordinates)>::dim == 2);
    const size_t G = coordinates.size();
    REQUIRE(G == 35u);

    // Serial reference in the same process and space: map_dist()/the space overload skip the
    // scheduler entirely.
    std::vector<NT> reference(G, NT(0));
    if constexpr (std::is_same_v<ExecutionSpace, TBB_exec>) {
      auto space = ExecutionSpace();
      integrator.map(space, reference.data(), coordinates, NT(1));
    } else {
      integrator.map_dist(reference.data(), coordinates, NT(1)).fence();
    }
    flush_maps();

    // Small enough that the split is limited by the rank count, not the cost score.
    MapScheduler::instance().set_quantum(1.);

    std::vector<NT> distributed(G, NT(0));
    integrator.map(distributed.data(), coordinates, NT(1)).fence();
    flush_maps();

    INFO("rank " << my_rank << " of " << n_ranks);
    CHECK(bitwise_equal(reference, distributed));

    // And each value belongs to the grid point it is stored at, not to a shifted one: constant()
    // encodes the external position, while the kernel does not depend on it, so every entry is
    // "its own position plus the same integral". Recovering that integral from entry 0 and
    // predicting the rest is what catches a slice landing at the wrong offset.
    auto position = [&](const size_t i) { return coordinates.forward(coordinates.from_linear_index(i)); };
    auto encoded = [](const auto &p) { return NT(p[0]) + NT(100) * NT(p[1]); };

    const NT integral = distributed[0] - encoded(position(0));
    for (size_t i = 0; i < G; ++i) {
      const NT expected = encoded(position(i)) + integral;
      INFO("grid point " << i);
      using Kokkos::abs;
      CHECK(abs(expected - distributed[i]) <= 10 * expected_precision<ctype>::value * abs(expected));
    }
  };

  SECTION("TBB") { check(TBB_exec(), (double)0); }
  SECTION("Threads") { check(KokkosHost_exec(), (double)0); }
  SECTION("GPU") { check(GPU_exec(), (double)0); }
}

TEST_CASE("Interleaved device and host maps in one deferral scope", "[integration][quadrature][mpi]")
{
  DiFfRG::Init();

  // The gotcha this guards: a host map() is synchronous, so writing GPU, CPU, GPU used to block the
  // second launch until the host kernel finished. Host maps inside a DeferredMaps scope are now
  // queued and run at flush time, which makes call order irrelevant -- but only if the results are
  // still exactly right, and still land in the right place.
  constexpr int dim = 2;
  using NT = double;
  using ctype = get_type::ctype<NT>;

  std::array<ctype, dim> ext_min, ext_max;
  std::array<size_t, dim> grid_size;
  std::array<QuadratureType, dim> quad_type;
  for (uint d = 0; d < dim; ++d) {
    ext_min[d] = ctype(-1.5);
    ext_max[d] = ctype(1.25);
    grid_size[d] = 16;
    quad_type[d] = QuadratureType::legendre;
  }

  QuadratureProvider qp;
  QuadratureIntegrator<dim, NT, PolyIntegrand<dim, NT>, GPU_exec> gpu_a(qp, grid_size, ext_min, ext_max, quad_type);
  QuadratureIntegrator<dim, NT, PolyIntegrand<dim, NT>, TBB_exec> cpu(qp, grid_size, ext_min, ext_max, quad_type);
  QuadratureIntegrator<dim, NT, PolyIntegrand<dim, NT>, GPU_exec> gpu_b(qp, grid_size, ext_min, ext_max, quad_type);

  std::array<NT, 4 * dim> coeffs{};
  for (uint d = 0; d < dim; ++d)
    coeffs[4 * d] = 1;

  constexpr uint rsize = 32;
  LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);

  // Reference: each map on its own, fully landed, no deferral anywhere.
  std::vector<NT> ref_a(rsize, NT(0)), ref_c(rsize, NT(0)), ref_b(rsize, NT(0));
  std::apply([&](auto... c) { gpu_a.map(ref_a.data(), coordinates, c...); }, coeffs);
  flush_maps();
  std::apply([&](auto... c) { cpu.map(ref_c.data(), coordinates, c...); }, coeffs);
  flush_maps();
  std::apply([&](auto... c) { gpu_b.map(ref_b.data(), coordinates, c...); }, coeffs);
  flush_maps();

  // The interleaved block: the CPU map in the middle is queued, so it must not have written
  // anything yet when the scope is still open, and must be correct once it closes.
  std::vector<NT> got_a(rsize, NT(0)), got_c(rsize, NT(0)), got_b(rsize, NT(0));
  {
    DeferredMaps defer;
    std::apply([&](auto... c) { gpu_a.map(got_a.data(), coordinates, c...); }, coeffs);
    std::apply([&](auto... c) { cpu.map(got_c.data(), coordinates, c...); }, coeffs);
    std::apply([&](auto... c) { gpu_b.map(got_b.data(), coordinates, c...); }, coeffs);

    // Deferred, so still untouched -- unless this build has no device at all, in which case there
    // is nothing to overlap with and the map runs inline by design.
    if constexpr (internal::has_device_backend) {
      bool all_zero = true;
      for (uint i = 0; i < rsize; ++i)
        all_zero &= got_c[i] == NT(0);
      CHECK(all_zero);
    }
  } // one fence, host queue drained first, then everything lands

  CHECK(bitwise_equal(ref_a, got_a));
  CHECK(bitwise_equal(ref_c, got_c));
  CHECK(bitwise_equal(ref_b, got_b));

  // And the middle one really is the integral, not a copy of a neighbour.
  NT volume = 1;
  for (uint d = 0; d < dim; ++d)
    volume *= (ext_max[d] - ext_min[d]);
  for (uint i = 0; i < rsize; ++i) {
    const auto expected = coordinates.forward(i) + volume;
    using Kokkos::abs;
    CHECK(abs(expected - got_c[i]) <= 10 * expected_precision<ctype>::value * abs(expected));
  }
}

TEST_CASE("A queued host map is dropped, not run, when the scope unwinds", "[integration][quadrature][mpi]")
{
  DiFfRG::Init();

  // ~DeferredMaps discards on the exception path because the destinations may already be gone. A
  // queued job is exactly as abandoned as an unlanded copy, so it must not run either.
  constexpr int dim = 2;
  using NT = double;
  using ctype = get_type::ctype<NT>;

  std::array<ctype, dim> ext_min, ext_max;
  std::array<size_t, dim> grid_size;
  std::array<QuadratureType, dim> quad_type;
  for (uint d = 0; d < dim; ++d) {
    ext_min[d] = ctype(-1.5);
    ext_max[d] = ctype(1.25);
    grid_size[d] = 16;
    quad_type[d] = QuadratureType::legendre;
  }

  QuadratureProvider qp;
  QuadratureIntegrator<dim, NT, PolyIntegrand<dim, NT>, TBB_exec> cpu(qp, grid_size, ext_min, ext_max, quad_type);
  std::array<NT, 4 * dim> coeffs{};
  for (uint d = 0; d < dim; ++d)
    coeffs[4 * d] = 1;

  constexpr uint rsize = 32;
  LinearCoordinates1D<ctype> coordinates(rsize, 0., 1.);
  std::vector<NT> sink(rsize, NT(0));

  // Single rank only, deliberately. Unwinding out of a DeferredMaps scope poisons the scheduler by
  // design -- the ranks have diverged, so the next collective must abort with a diagnosis rather
  // than hang. There is no un-poison, and there should not be one: provoking that state under
  // mpiexec would correctly take the whole test binary down with it.
  if constexpr (internal::has_device_backend) {
    if (MapScheduler::instance().n_ranks() == 1) {
      try {
        DeferredMaps defer;
        std::apply([&](auto... c) { cpu.map(sink.data(), coordinates, c...); }, coeffs);
        throw std::runtime_error("unwind");
      } catch (const std::runtime_error &) {
      }

      for (uint i = 0; i < rsize; ++i)
        CHECK(sink[i] == NT(0));
    }
  }
}
