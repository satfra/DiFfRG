#pragma once

// external libraries
#include <deal.II/base/config.h>
#include <deal.II/grid/tria.h>
#ifdef DEAL_II_WITH_MPI
#include <deal.II/distributed/shared_tria.h>
#endif

// standard library
#include <type_traits>

// DiFfRG
#include <DiFfRG/common/linear_algebra.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/mesh/configuration_mesh.hh>

namespace DiFfRG
{
  using namespace dealii;

  struct RectangularMeshOptions {
    bool origin_cell_centered = false;
  };

  namespace internal
  {
    /**
     * @brief Holds the triangulation, constructing it the way its type requires.
     *
     * A serial dealii::Triangulation is default-constructible; a parallel one is not -- it needs
     * a communicator at construction. Neither is movable, so this cannot be a factory function
     * returning by value; a holder with a default member initializer is the way to give each
     * type its own construction without duplicating RectangularMesh.
     */
    template <typename Tria> struct TriangulationHolder {
      Tria triangulation;
    };

#ifdef DEAL_II_WITH_MPI
    template <int dim> struct TriangulationHolder<dealii::parallel::shared::Triangulation<dim>> {
      /**
       * `allow_artificial_cells = false` is load-bearing, not a default worth changing.
       *
       * With artificial cells, active_cell_iterators() still enumerates them but
       * cell->get_dof_indices() throws on one. That breaks the Kurganov-Tadmor topology and
       * reconstruction caches outright, since they are dense arrays over every active cell.
       * Keeping every cell real is also what makes active_cell_index() a global, deterministic
       * index that every rank agrees on -- the property those caches are built on.
       */
      dealii::parallel::shared::Triangulation<dim> triangulation{MPI_COMM_WORLD, dealii::Triangulation<dim>::none,
                                                                 /*allow_artificial_cells=*/false};
    };
#endif
  } // namespace internal

  /**
   * @brief Class to manage the discretization mesh, also called grid and triangluation, on which we simulate.
   * This class only builds cartesian, regular grids, however cell density in all directions can be chosen
   * independently.
   *
   * @tparam dim_ dimensionality of the spatial discretization.
   * @tparam TriangulationType_ the triangulation to build on. Defaults to
   * DefaultTriangulation<dim_>, i.e. `parallel::shared::Triangulation` in an MPI+PETSc build and
   * `dealii::Triangulation` otherwise, so that a plain `RectangularMesh<dim>` follows the build
   * configuration. A partitioned triangulation splits cell ownership across ranks while keeping
   * the mesh itself replicated, which is what lets the assemblers restrict themselves to
   * locally-owned cells without any of the caches or the EoM search having to change.
   *
   * Use RectangularMeshSerial<dim> / RectangularMeshParallel<dim> below to pin one explicitly.
   */
  template <uint dim_, typename TriangulationType_ = DefaultTriangulation<dim_>>
  class RectangularMesh : protected internal::TriangulationHolder<TriangulationType_>
  {
  public:
    static constexpr uint dim = dim_;
    using TriangulationType = TriangulationType_;
    static constexpr bool is_rectangular = true;

    /// Whether cells carry subdomain ids, i.e. whether `cell->is_locally_owned()` is selective.
    static constexpr bool is_parallel = !std::is_same_v<TriangulationType, dealii::Triangulation<dim_>>;

    /**
     * @brief Construct a new RectangularMesh object.
     * @param config ConfigTree object containing the parameters for the mesh.
     */
    [[deprecated("Please use RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config) instead")]]
    RectangularMesh(const ConfigTree &config);

    /**
     * @brief Construct a new RectangularMesh object from ConfigTree configuration.
     * @param config ConfigTree object containing the parameters for the mesh.
     * @param options Options controlling rectangular mesh construction.
     *
     * All configured axes must start at zero when origin centering is enabled.
     */
    [[deprecated("Please use RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config, "
                 "RectangularMeshOptions options) instead")]]
    RectangularMesh(const ConfigTree &config, RectangularMeshOptions options);

    RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config);

    /**
     * @brief Construct a new RectangularMesh object.
     * @param mesh_config Configuration of the rectangular mesh.
     * @param options Options controlling rectangular mesh construction.
     *
     * All configured axes must start at zero when origin centering is enabled.
     */
    RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config, RectangularMeshOptions options);

    TriangulationType &get_triangulation() { return this->triangulation; }
    const TriangulationType &get_triangulation() const { return this->triangulation; }

  protected:
    virtual void make_grid();

    const Config::ConfigurationMesh<dim> mesh_config;
    const RectangularMeshOptions options;
  };

  /**
   * @brief A rectangular mesh pinned to a serial triangulation, whatever the build configuration.
   *
   * Name this when a run must stay serial in an MPI build: LDG, which cannot be distributed at
   * all, and any test compared against a stored reference, which must produce the same numbers in
   * a serial and an MPI build tree.
   *
   * Naming it is sufficient on its own: a Discretization defaults its vector and matrix from the
   * mesh (see LAVectorFor in common/linear_algebra.hh), so this pins the linear algebra serial
   * too rather than leaving it on the build-configuration default.
   */
  template <uint dim> using RectangularMeshSerial = RectangularMesh<dim, dealii::Triangulation<dim>>;

#ifdef DEAL_II_WITH_MPI
  /**
   * @brief A rectangular mesh pinned to a partitioned triangulation.
   *
   * Only cell ownership is distributed; the mesh itself stays replicated on every rank.
   */
  template <uint dim>
  using RectangularMeshParallel = RectangularMesh<dim, dealii::parallel::shared::Triangulation<dim>>;
#endif
} // namespace DiFfRG
