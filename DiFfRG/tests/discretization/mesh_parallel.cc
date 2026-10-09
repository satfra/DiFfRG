#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

#include <algorithm>
#include <vector>

// These tests pin down the invariants the whole replicated-mesh design rests on.
//
// parallel::shared::Triangulation only exists in an MPI-enabled deal.II, so the partitioned
// cases are guarded. The file must still contribute at least one test case in a serial build:
// catch_discover_tests registers a <target>_NOT_BUILT placeholder that is replaced only once
// test cases are discovered, so a binary with none leaves that placeholder failing forever.
// The serial case below is the ordering half of the same invariant, and is worth having in
// its own right.

namespace
{
  using DiFfRG::RectangularMesh;
  using DiFfRG::Config::ConfigurationMesh;
  using DiFfRG::Config::GridAxis;

  /// 32 cells on [0, 1]; small enough that every rank count up to 4 gets a non-trivial share.
  ConfigurationMesh<1> make_config()
  {
    const std::vector<GridAxis> x_grid{GridAxis(0.0, 1.0 / 32.0, 1.0)};
    return ConfigurationMesh<1>(0u, x_grid);
  }
} // namespace

TEST_CASE("active_cell_index covers every cell exactly once", "[discretization][mesh]")
{
  const RectangularMesh<1> mesh(make_config());
  const auto &tria = mesh.get_triangulation();

  // The caches indexed by active_cell_index() assume it is a bijection onto [0, n_active_cells).
  // deal.II guarantees this, but it is cheap to hold it to that, and this is the serial
  // counterpart of the rank-independence check below.
  std::vector<int> seen(tria.n_active_cells(), 0);
  for (const auto &cell : tria.active_cell_iterators()) {
    REQUIRE(cell->active_cell_index() < tria.n_active_cells());
    seen[cell->active_cell_index()] += 1;
  }
  for (const auto count : seen) REQUIRE(count == 1);
}

#ifdef DEAL_II_WITH_MPI

namespace
{
  template <uint dim> using ParallelMesh = RectangularMesh<dim, dealii::parallel::shared::Triangulation<dim>>;

  /// Cell barycentres in ascending active_cell_index order.
  template <typename Mesh> std::vector<dealii::Point<Mesh::dim>> centers_by_index(const Mesh &mesh)
  {
    const auto &tria = mesh.get_triangulation();
    std::vector<dealii::Point<Mesh::dim>> out(tria.n_active_cells());
    for (const auto &cell : tria.active_cell_iterators())
      out[cell->active_cell_index()] = cell->center();
    return out;
  }
} // namespace

TEST_CASE("A partitioned mesh is geometrically identical to the serial one", "[discretization][mesh][mpi]")
{
  DiFfRG::Init();
  const auto config = make_config();
  const RectangularMesh<1> serial(config);
  const ParallelMesh<1> parallel(config);

  REQUIRE(parallel.get_triangulation().n_active_cells() == serial.get_triangulation().n_active_cells());

  // Not merely the same set of cells: the same cells in the same order. The KT topology and
  // reconstruction caches are dense arrays indexed by active_cell_index(), so a reordering
  // would silently corrupt them rather than fail loudly.
  const auto s = centers_by_index(serial);
  const auto p = centers_by_index(parallel);
  REQUIRE(s.size() == p.size());
  for (std::size_t i = 0; i < s.size(); ++i)
    REQUIRE(s[i].distance(p[i]) == Catch::Approx(0.).margin(1e-14));
}

TEST_CASE("active_cell_index is global and rank-independent", "[discretization][mesh][mpi]")
{
  DiFfRG::Init();
  const ParallelMesh<1> mesh(make_config());
  const auto &tria = mesh.get_triangulation();

  // Every rank sees every cell: this is what `allow_artificial_cells = false` buys, and what the
  // caches, the EoM search and the ghosted-read design all depend on.
  const auto n_cells = tria.n_active_cells();
  REQUIRE(n_cells == DiFfRG::MPI::max_reduce(MPI_COMM_WORLD, static_cast<int>(n_cells)));
  REQUIRE(n_cells == static_cast<decltype(n_cells)>(
                         DiFfRG::MPI::min_reduce(MPI_COMM_WORLD, static_cast<int>(n_cells))));

  // And every rank agrees on which cell each index denotes -- checked by reducing the
  // barycentres rather than trusting the ordering.
  auto centers = centers_by_index(mesh);
  std::vector<double> x(centers.size());
  std::transform(centers.begin(), centers.end(), x.begin(), [](const auto &c) { return c[0]; });
  std::vector<double> hi = x, lo = x;
  DiFfRG::MPI::max_reduce(MPI_COMM_WORLD, hi.data(), static_cast<int>(hi.size()));
  DiFfRG::MPI::min_reduce(MPI_COMM_WORLD, lo.data(), static_cast<int>(lo.size()));
  for (std::size_t i = 0; i < x.size(); ++i) REQUIRE(hi[i] == Catch::Approx(lo[i]).margin(1e-14));
}

TEST_CASE("Ownership partitions the cells exactly once", "[discretization][mesh][mpi]")
{
  DiFfRG::Init();
  const ParallelMesh<1> mesh(make_config());
  const auto &tria = mesh.get_triangulation();

  std::vector<int> owned(tria.n_active_cells(), 0);
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned()) owned[cell->active_cell_index()] = 1;

  // Summing the per-rank ownership flags must give exactly one owner per cell: no cell dropped
  // (which would silently lose its contribution to the residual) and none counted twice.
  DiFfRG::MPI::sum_reduce(MPI_COMM_WORLD, owned.data(), static_cast<int>(owned.size()));
  for (std::size_t i = 0; i < owned.size(); ++i) REQUIRE(owned[i] == 1);
}

#endif // DEAL_II_WITH_MPI
