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

  /// Options of RectangularMesh.
  struct RectangularMeshOptions {
    /**
     * @brief Shift the grid so that, after all refinements, the first cell along every axis is centered on 0.
     *
     * The domain then starts at \f$-h/2\f$ along every axis, \f$h\f$ the width of the finest first cell; all axes
     * must start at 0 in the configuration. Useful when field-space quantities are evaluated at cell centers (FV) and
     * the origin must be one of them.
     */
    bool origin_cell_centered = false;
  };

  namespace internal
  {
    /// Holds the triangulation, constructed as its type requires.
    template <typename Tria> struct TriangulationHolder {
      Tria triangulation;
    };

#ifdef DEAL_II_WITH_MPI
    template <int dim> struct TriangulationHolder<dealii::parallel::shared::Triangulation<dim>> {
      /// Without artificial cells: every rank has every cell, with the same active_cell_index().
      dealii::parallel::shared::Triangulation<dim> triangulation{MPI_COMM_WORLD, dealii::Triangulation<dim>::none,
                                                                 /*allow_artificial_cells=*/false};
    };
#endif
  } // namespace internal

  /**
   * @brief A rectangular field-space mesh of rectangular cells, built from a Config::ConfigurationMesh.
   *
   * The coarse grid has the cell widths of the ConfigurationMesh along each axis (cells of different width per
   * subrange), and is then refined globally `/discretization/grid/refine` times.
   *
   * ```cpp
   * RectangularMesh<1> mesh(Config::ConfigurationMesh<1>(config));
   * ```
   *
   * @tparam dim_ Field-space dimension, 1 to 3.
   * @tparam TriangulationType_ The deal.II triangulation. Default: `parallel::shared::Triangulation` in an MPI+PETSc
   * build (every rank holds the whole mesh, cell ownership is split between ranks), `dealii::Triangulation` otherwise.
   * RectangularMeshSerial and RectangularMeshParallel fix one choice.
   */
  template <uint dim_, typename TriangulationType_ = DefaultTriangulation<dim_>>
  class RectangularMesh : protected internal::TriangulationHolder<TriangulationType_>
  {
  public:
    /// Field-space dimension.
    static constexpr uint dim = dim_;
    /// The deal.II triangulation type.
    using TriangulationType = TriangulationType_;
    /// Marks the mesh as rectangular.
    static constexpr bool is_rectangular = true;

    /// Whether cells carry subdomain ids, i.e. whether `cell->is_locally_owned()` is selective.
    static constexpr bool is_parallel = !std::is_same_v<TriangulationType, dealii::Triangulation<dim_>>;

    /// Build from the `/discretization/grid` keys (see Config::ConfigurationMesh).
    [[deprecated("Please use RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config) instead")]]
    RectangularMesh(const ConfigTree &config);

    /// Build from the `/discretization/grid` keys with @p options.
    [[deprecated("Please use RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config, "
                 "RectangularMeshOptions options) instead")]]
    RectangularMesh(const ConfigTree &config, RectangularMeshOptions options);

    /// Build the mesh described by @p mesh_config.
    RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config);

    /// Build the mesh described by @p mesh_config, with @p options.
    RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config, RectangularMeshOptions options);

    /// The triangulation.
    TriangulationType &get_triangulation() { return this->triangulation; }
    /// The triangulation.
    const TriangulationType &get_triangulation() const { return this->triangulation; }

  protected:
    /// Create the coarse grid and refine it.
    virtual void make_grid();

    const Config::ConfigurationMesh<dim> mesh_config;
    const RectangularMeshOptions options;
  };

  /**
   * @brief A rectangular mesh on a serial triangulation, also in an MPI build. Required for LDG.
   *
   * A Discretization on this mesh also uses serial vectors and matrices.
   */
  template <uint dim> using RectangularMeshSerial = RectangularMesh<dim, dealii::Triangulation<dim>>;

#ifdef DEAL_II_WITH_MPI
  /**
   * @brief A rectangular mesh on a parallel::shared::Triangulation: every rank holds the whole mesh, cell ownership
   * is split between ranks.
   */
  template <uint dim>
  using RectangularMeshParallel = RectangularMesh<dim, dealii::parallel::shared::Triangulation<dim>>;
#endif
} // namespace DiFfRG
