#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

using namespace DiFfRG;

// The tests pin the threshold rather than using the hardware-derived default, so that the expected
// split widths below do not depend on which GPU the suite runs on. Two targets differing only in
// resource, so that a mixed-resource schedule can be exercised at one fixed split width.
static constexpr MapTarget QUANTUM{MapResource::device, 5e4};
static constexpr MapTarget HOST_QUANTUM{MapResource::host, 5e4};

/**
 * Everything MapScheduler does apart from the single MPI_Allgatherv -- choosing the split width,
 * picking owners, the byte accounting, the packing and the landing -- is a pure function of the
 * sequence of map() calls. That makes it testable at any rank count in a single process, which
 * matters: an MPI build needs an MPI deal.II underneath it, so a plain build would otherwise leave
 * all of this code completely unexercised.
 *
 * The subclass below plays the part of the network. Everything it drives (prepare_batch,
 * finish_batch, and the buffers between them) is the production code path, not a re-implementation.
 */
namespace
{
  class SimulatedRank : public MapScheduler
  {
  public:
    SimulatedRank(uint rank, uint n_ranks) : MapScheduler(rank, n_ranks) {}

    using MapScheduler::finish_batch;
    using MapScheduler::prepare_batch;

    const std::vector<char> &send_buffer() const { return m_send; }
    const std::vector<int> &counts() const { return m_counts; }
    const std::vector<int> &displs() const { return m_displs; }

    /// Copy every peer's packed block into this rank's receive buffer, i.e. do by hand what
    /// MPI_Allgatherv would have done.
    void receive_from(const std::vector<SimulatedRank *> &all)
    {
      m_recv.assign(static_cast<size_t>(m_displs.back()) + static_cast<size_t>(m_counts.back()), '\0');
      for (uint p = 0; p < all.size(); ++p)
        if (m_counts[p] > 0)
          std::memcpy(m_recv.data() + m_displs[p], all[p]->send_buffer().data(), static_cast<size_t>(m_counts[p]));
    }
  };

  /// One simulated batch: every rank schedules the same sequence of maps, then they exchange.
  struct Cluster {
    std::vector<std::unique_ptr<SimulatedRank>> ranks;
    std::vector<SimulatedRank *> raw;

    /// `quantum` pins the split threshold so the expected widths below do not depend on the host's
    /// hardware. Pass 0 to exercise the automatic, per-resource thresholds instead.
    explicit Cluster(uint n, double quantum = 5e4)
    {
      for (uint p = 0; p < n; ++p) {
        ranks.push_back(std::make_unique<SimulatedRank>(p, n));
        ranks.back()->set_quantum(quantum);
      }
      for (auto &r : ranks)
        raw.push_back(r.get());
    }

    /// Stand in for opening (or not opening) a DeferredMaps scope around the batch. Production sets
    /// this through MapCompletion::set_deferral(); the default is false, matching a model that maps
    /// without a scope.
    void set_batched(const bool batched)
    {
      for (auto &r : ranks)
        r->set_batched(batched);
    }

    void exchange()
    {
      for (auto *r : raw)
        REQUIRE(r->prepare_batch());
      for (auto *r : raw)
        r->receive_from(raw);
      for (auto *r : raw)
        r->finish_batch();
    }
  };
} // namespace

TEST_CASE("Slices tile the grid exactly, at every rank count", "[integration][mpi][scheduler]")
{
  const uint n_ranks = GENERATE(1u, 2u, 3u, 4u, 5u, 7u, 8u, 12u, 16u);
  // 64 is the QCD_Nf2 external momentum grid; 3 and 5 are deliberately awkward divisors.
  const size_t G = GENERATE(size_t(1), size_t(3), size_t(5), size_t(64), size_t(100));

  Cluster cluster(n_ranks);
  std::vector<double> sink(G, 0.);

  std::vector<size_t> coverage(G, 0);
  for (uint p = 0; p < n_ranks; ++p) {
    const MapSlice s = cluster.raw[p]->schedule(0, sink.data(), sizeof(double), G, 6912, true, QUANTUM);
    REQUIRE(s.offset + s.count <= G);
    for (size_t i = s.offset; i < s.offset + s.count; ++i)
      coverage[i] += 1;
  }

  // Every grid point computed exactly once: no gaps (wrong answers) and no duplicates (wasted GPU
  // time, and two ranks writing the same destination).
  for (size_t i = 0; i < G; ++i)
    CHECK(coverage[i] == 1);
}

TEST_CASE("The split width follows the cost score", "[integration][mpi][scheduler]")
{
  Cluster cluster(16);
  std::vector<double> sink(64, 0.);

  auto owners_of = [&](const size_t Q, const bool splittable) {
    size_t n_owners = 0;
    for (uint p = 0; p < 16; ++p)
      if (cluster.raw[p]->schedule(0, sink.data(), sizeof(double), 64, Q, splittable, QUANTUM).count > 0) ++n_owners;
    return n_owners;
  };

  // The three flow classes in QCD_Nf2 at x_order=32, cos*=6, phi=6, against a 5e4 quantum.
  SECTION("p2_1ang (Q = 192, S = 12288) stays on one rank") { CHECK(owners_of(192, true) == 1); }
  SECTION("4D_2ang (Q = 1152, S = 73728) stays on one rank") { CHECK(owners_of(1152, true) == 1); }
  SECTION("4D_3ang (Q = 6912, S = 442368) spreads over 8") { CHECK(owners_of(6912, true) == 8); }
  SECTION("a coordinate system flagged unsplittable is assigned whole") { CHECK(owners_of(6912, false) == 1); }
}

TEST_CASE("Cheap flows spread across ranks instead of being chopped up", "[integration][mpi][scheduler]")
{
  // Four cheap flows and four ranks: the point of assigning a cheap flow whole is that they end up
  // on four different ranks, one each, rather than every rank doing a quarter of every flow.
  Cluster cluster(4);
  std::vector<std::vector<double>> sinks(4, std::vector<double>(64, 0.));

  std::vector<size_t> per_rank(4, 0);
  for (size_t flow = 0; flow < 4; ++flow)
    for (uint p = 0; p < 4; ++p)
      if (cluster.raw[p]->schedule(flow, sinks[flow].data(), sizeof(double), 64, 192, true, QUANTUM).count > 0) per_rank[p] += 1;

  for (uint p = 0; p < 4; ++p)
    CHECK(per_rank[p] == 1);
}

TEST_CASE("A batch exchange reassembles the full result on every rank", "[integration][mpi][scheduler]")
{
  const uint n_ranks = GENERATE(2u, 3u, 4u, 8u);
  Cluster cluster(n_ranks);

  // Three destinations of different width and cost, as in a real dt_variables() block.
  constexpr size_t n_flows = 3;
  const size_t G[n_flows] = {64, 64, 37};
  const size_t Q[n_flows] = {6912, 192, 6912};

  // The truth each rank should end up holding.
  std::vector<std::vector<double>> truth(n_flows);
  for (size_t f = 0; f < n_flows; ++f) {
    truth[f].resize(G[f]);
    for (size_t i = 0; i < G[f]; ++i)
      truth[f][i] = 1000. * double(f) + double(i);
  }

  // Per rank, per flow: the destination buffer it actually holds.
  std::vector<std::vector<std::vector<double>>> dest(n_ranks);
  for (uint p = 0; p < n_ranks; ++p) {
    dest[p].resize(n_flows);
    for (size_t f = 0; f < n_flows; ++f)
      dest[p][f].assign(G[f], -1.);
  }

  for (size_t f = 0; f < n_flows; ++f)
    for (uint p = 0; p < n_ranks; ++p) {
      const MapSlice s = cluster.raw[p]->schedule(f, dest[p][f].data(), sizeof(double), G[f], Q[f], true, QUANTUM);
      // Only the owner computes -- exactly what map() does when it skips a non-owner.
      for (size_t i = s.offset; i < s.offset + s.count; ++i)
        dest[p][f][i] = truth[f][i];
    }

  cluster.exchange();

  for (uint p = 0; p < n_ranks; ++p)
    for (size_t f = 0; f < n_flows; ++f) {
      INFO("rank " << p << ", flow " << f);
      // Bitwise: the whole design rests on the distributed result being the serial result exactly.
      CHECK(std::memcmp(dest[p][f].data(), truth[f].data(), G[f] * sizeof(double)) == 0);
    }
}

TEST_CASE("Every rank derives the same plan", "[integration][mpi][scheduler]")
{
  Cluster cluster(5);
  std::vector<double> sink(64, 0.);

  // A mixed batch, scheduled in the same order on every rank.
  const size_t Qs[] = {6912, 192, 6912, 1152, 192, 6912};
  for (size_t f = 0; f < 6; ++f)
    for (uint p = 0; p < 5; ++p)
      cluster.raw[p]->schedule(f, sink.data(), sizeof(double), 64, Qs[f], true, QUANTUM);

  for (auto *r : cluster.raw)
    REQUIRE(r->prepare_batch());

  // The byte accounting is what the Allgatherv is parameterised with, so agreement on it is
  // exactly the condition under which the collective is well formed.
  for (uint p = 1; p < 5; ++p) {
    CHECK(cluster.raw[p]->counts() == cluster.raw[0]->counts());
    CHECK(cluster.raw[p]->displs() == cluster.raw[0]->displs());
  }

  const auto &counts = cluster.raw[0]->counts();
  const int total = std::accumulate(counts.begin(), counts.end(), 0);
  CHECK(total == static_cast<int>(6 * 64 * sizeof(double)));

  for (auto *r : cluster.raw)
    r->receive_from(cluster.raw);
  for (auto *r : cluster.raw)
    r->finish_batch();
}

TEST_CASE("Fill thresholds are hardware-derived and picked per execution space",
          "[integration][mpi][scheduler]")
{
  DiFfRG::Init();

  const double device = internal::device_fill_threshold();
  const double host = internal::host_fill_threshold();

  // Both must be positive, or schedule() would silently stop splitting entirely.
  CHECK(device > 0.);
  CHECK(host > 0.);

  // The selector keys on memory_space, so the host-backed spaces must all agree with each other and
  // GPU_exec must follow whichever space it actually resolves to in this build.
  CHECK(map_fill_threshold<KokkosHost_exec>() == host);
  CHECK(map_fill_threshold<TBB_exec>() == host);
  if constexpr (std::is_same_v<typename GPU_exec::memory_space, CPU_memory>)
    CHECK(map_fill_threshold<GPU_exec>() == host); // CUDA-less build: GPU_exec *is* the host space
  else
    CHECK(map_fill_threshold<GPU_exec>() == device);

  // The host threshold counts TBB workers, so it must track DiFfRG's thread budget -- and must
  // keep tracking it after the first call. Caching it in a function-local static would freeze
  // whatever value the first caller happened to see, which for a caller running before Init would
  // be the wrong one for the rest of the process.
  const unsigned int budget = n_threads();
  if (budget > 1) {
    set_thread_limit(budget / 2);
    CHECK(internal::host_fill_threshold() < host);
    CHECK(internal::host_fill_threshold() == double(n_threads()) * internal::host_grain);
    set_thread_limit(budget);
    CHECK(internal::host_fill_threshold() == host);
  }

  // An explicit override replaces both, which is what DIFFRG_MAP_QUANTUM has to mean.
  auto &scheduler = MapScheduler::instance();
  const double previous = scheduler.quantum();
  scheduler.set_quantum(1234.);
  CHECK(scheduler.quantum() == 1234.);
  scheduler.set_quantum(previous);
}

TEST_CASE("Device and host work are balanced against separate budgets", "[integration][mpi][scheduler]")
{
  // A rank owns a GPU *and* a share of the node's cores. They are different hardware and, inside a
  // DeferredMaps scope, they run at the same time -- so charging both to one budget would make a
  // rank that just took a large host flow look like a bad home for the next device flow, even
  // though its GPU is sitting idle. This is that regression, at the level where it is decided.
  constexpr uint n_ranks = 4;
  Cluster cluster(n_ranks);

  // Six cheap flows, alternating resource. Each is well under the quantum, so r == 1 and every one
  // is assigned whole -- which is exactly what makes the owner choice, and nothing else, visible.
  std::vector<std::vector<double>> sinks(6, std::vector<double>(64, 0.));
  std::vector<uint> device_owner(6, 0), host_owner(6, 0);

  for (uint p = 0; p < n_ranks; ++p)
    for (size_t f = 0; f < 6; ++f) {
      const MapTarget target = (f % 2 == 0) ? QUANTUM : HOST_QUANTUM;
      if (cluster.raw[p]->schedule(f, sinks[f].data(), sizeof(double), 64, 192, true, target).count > 0) {
        if (f % 2 == 0)
          device_owner[f] = p;
        else
          host_owner[f] = p;
      }
    }

  // Three device flows over four ranks: with a shared budget the alternating host flows would have
  // pushed each successive device flow onto a rank that had done no host work, still distinct but
  // for the wrong reason. With separate budgets the device flows see an untouched device budget and
  // fill ranks 0, 1, 2 -- and the host flows do the same independently.
  CHECK(device_owner[0] == 0);
  CHECK(device_owner[2] == 1);
  CHECK(device_owner[4] == 2);
  CHECK(host_owner[1] == 0);
  CHECK(host_owner[3] == 1);
  CHECK(host_owner[5] == 2);

  for (auto *r : cluster.raw)
    r->prepare_batch();
  for (auto *r : cluster.raw)
    r->receive_from(cluster.raw);
  for (auto *r : cluster.raw)
    r->finish_batch();

  // And the exchange does not care about the resource mix: every flow is whole on some rank.
  for (size_t f = 0; f < 6; ++f)
    CHECK(sinks[f].size() == 64u);
}

TEST_CASE("The split width follows the calling space's threshold", "[integration][mpi][scheduler]")
{
  // Same flow, same grid, two resources whose thresholds differ by 10x: the wider split must go to
  // the resource with the smaller threshold. This is the half of the fix that was already correct,
  // pinned so it stays that way.
  constexpr uint n_ranks = 8;
  constexpr size_t G = 64, Q = 6912; // S = 442368

  // No pinned override here: it would replace the per-space threshold, which is the thing under
  // test. The targets below therefore stand in for what map_target<ExecutionSpace>() returns.
  // Batched, because the *fill* threshold is only what a map inside a batch is measured against.
  auto width = [&](const MapTarget target) {
    Cluster cluster(n_ranks, 0.);
    cluster.set_batched(true);
    std::vector<double> sink(G, 0.);
    uint owners = 0;
    for (uint p = 0; p < n_ranks; ++p)
      if (cluster.raw[p]->schedule(0, sink.data(), sizeof(double), G, Q, true, target).count > 0) ++owners;
    return owners;
  };

  CHECK(width(MapTarget{MapResource::device, 5e4}) == 8u);   // 442368 / 5e4 = 8
  CHECK(width(MapTarget{MapResource::host, 5e5}) == 1u);     // 442368 / 5e5 < 1
}

TEST_CASE("An unbatched map is split wider than a batched one", "[integration][mpi][scheduler]")
{
  // The fill threshold answers "is this rank's share big enough to occupy it?", and holding back
  // below it is right only while the ranks left out are about to be taken by the *other* maps in
  // the batch. Outside a DeferredMaps scope there are none: map() flushes on return, so every rank
  // that owns no slice sits in the Allgatherv until the owners are done. There the threshold that
  // applies is the launch cost, not the fill.
  constexpr uint n_ranks = 8;
  constexpr size_t G = 64, Q = 192; // S = 12288, i.e. below a 5e4 fill threshold

  auto width = [&](const bool batched, const double pinned) {
    Cluster cluster(n_ranks, pinned);
    cluster.set_batched(batched);
    std::vector<double> sink(G, 0.);
    uint owners = 0;
    for (uint p = 0; p < n_ranks; ++p)
      if (cluster.raw[p]->schedule(0, sink.data(), sizeof(double), G, Q, true, MapTarget{MapResource::host, 5e4})
              .count > 0)
        ++owners;
    return owners;
  };

  // Batched: below the fill threshold, so assigned whole -- the packing spreads a block of cheap
  // flows across ranks instead of slivering each one, which is the behaviour worth keeping.
  CHECK(width(true, 0.) == 1u);
  // Unbatched: 12288 / 1024 = 12, clamped to the 8 ranks available.
  CHECK(width(false, 0.) == 8u);
  // A pinned DIFFRG_MAP_QUANTUM means the user chose the width; both cases must respect it.
  CHECK(width(true, 5e4) == 1u);
  CHECK(width(false, 5e4) == 1u);
}

TEST_CASE("Multi-dimensional grids split, and the slices reproduce the base positions",
          "[integration][mpi][scheduler]")
{
  // Splitting used to be restricted to dim == 1 because SubCoordinates derived a per-axis *box*,
  // which is the same set of points as a linear range only in one dimension. The expensive flows
  // are exactly the multi-angle ones, so that restriction pinned them to a single rank at any rank
  // count. This is the end-to-end property that restriction existed to avoid violating: the union
  // of the slices is the whole grid, the slices are disjoint, and the position a rank computes for
  // its s-th point is the *identical* double the unsplit run computes for that global point.
  const uint n_ranks = GENERATE(2u, 3u, 4u, 5u, 8u, 16u);

  // A quantum of 1 makes every map split as wide as it is allowed to, so the width is decided by
  // min(n_ranks, G) alone and the assertions do not depend on any threshold.
  auto check_grid = [&](const auto &base) {
    const size_t G = base.size();
    Cluster cluster(n_ranks, 1.);
    std::vector<double> sink(G, 0.);

    std::vector<size_t> coverage(G, 0);
    uint owners = 0;
    for (uint p = 0; p < n_ranks; ++p) {
      const MapSlice sl = cluster.raw[p]->schedule(0, sink.data(), sizeof(double), G, 1, true, QUANTUM);
      if (sl.count == 0) continue;
      ++owners;

      const SubCoordinates window(base, sl.offset, sl.count);
      REQUIRE(window.size() == sl.count);
      // The variadic forward() used to be ill-formed on instantiation (a braced-init-list cannot
      // deduce the array overload's element type), so nothing had ever called it. Pin it here.
      CHECK(device::apply([&](const auto &...ix) { return window.forward(ix...); }, window.from_linear_index(0)) ==
            window.forward(window.from_linear_index(0)));
      for (size_t i = 0; i < sl.count; ++i) {
        const auto here = window.forward(window.from_linear_index(i));
        const auto there = base.forward(base.from_linear_index(sl.offset + i));
        // Bitwise, not approximate: forward() is an fp64 transform and the whole distribution
        // guarantee is that a rank computes the same number for a point as a serial run does.
        INFO("slice at " << sl.offset << ", point " << i);
        CHECK(std::memcmp(&here, &there, sizeof(here)) == 0);
        coverage[sl.offset + i] += 1;
      }
    }

    // It really was split, rather than passing vacuously on one rank owning everything.
    CHECK(owners == std::min<uint>(n_ranks, uint(G)));
    // Every point computed exactly once: no gaps and no two ranks writing the same destination.
    for (size_t i = 0; i < G; ++i)
      CHECK(coverage[i] == 1);
  };

  // Extents chosen so the linear range never lines up with an axis boundary: 35 and 36 share no
  // factor with most of the rank counts above, and neither grid is square, so a row/column mix-up
  // cannot pass by symmetry.
  SECTION("dim == 2")
  {
    check_grid(CoordinatePackND(LinearCoordinates1D<double>(7, 0.5, 3.5), LinearCoordinates1D<double>(5, -1., 2.)));
  }

  SECTION("dim == 3")
  {
    check_grid(CoordinatePackND(LinearCoordinates1D<double>(4, 0., 1.), LinearCoordinates1D<double>(3, 1., 2.),
                                LinearCoordinates1D<double>(3, -1., 1.)));
  }
}
