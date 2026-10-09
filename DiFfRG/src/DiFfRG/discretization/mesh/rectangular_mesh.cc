// standard library
#include "DiFfRG/discretization/mesh/configuration_mesh.hh"

// external libraries
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

// DiFfRG
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

namespace DiFfRG
{
  using namespace dealii;

  template <uint dim, typename Tria>
  RectangularMesh<dim, Tria>::RectangularMesh(const ConfigTree &config)
      : RectangularMesh<dim, Tria>(Config::ConfigurationMesh<dim>(config), RectangularMeshOptions{})
  {
  }

  template <uint dim, typename Tria>
  RectangularMesh<dim, Tria>::RectangularMesh(const ConfigTree &config, const RectangularMeshOptions options)
      : RectangularMesh<dim, Tria>(Config::ConfigurationMesh<dim>(config), options)
  {
  }

  template <uint dim, typename Tria>
  RectangularMesh<dim, Tria>::RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config)
      : RectangularMesh<dim, Tria>(mesh_config, RectangularMeshOptions{})
  {
  }

  template <uint dim, typename Tria>
  RectangularMesh<dim, Tria>::RectangularMesh(const Config::ConfigurationMesh<dim> &mesh_config,
                                              const RectangularMeshOptions options)
      : mesh_config(mesh_config), options(options)
  {
    static_assert(dim > 0 && dim < 4);
    make_grid();
  }

  template <uint dim, typename Tria> void RectangularMesh<dim, Tria>::make_grid()
  {
    const auto triangulation_data = mesh_config.get_triangulation_data(options.origin_cell_centered);
    dealii::GridGenerator::subdivided_hyper_rectangle(this->triangulation, triangulation_data.step_sizes,
                                                      triangulation_data.lower_left, triangulation_data.upper_right);

    this->triangulation.refine_global(mesh_config.refine);
  }

  // Spelled with the triangulation explicit rather than through the default argument. Since the
  // default follows the build configuration, `RectangularMesh<1>` names the *parallel* type in an
  // MPI build, which would both collide with the block below and leave the serial variant -- whose
  // constructors and make_grid() are out of line -- with no instantiation at all.
  template class RectangularMesh<1, dealii::Triangulation<1>>;
  template class RectangularMesh<2, dealii::Triangulation<2>>;
  template class RectangularMesh<3, dealii::Triangulation<3>>;

#ifdef DEAL_II_WITH_MPI
  // The partitioned variants. GridGenerator::subdivided_hyper_rectangle and refine_global both
  // work verbatim on a parallel::shared::Triangulation -- it is filled identically on every rank
  // and partitioned afterwards -- so make_grid() needs no specialization.
  template class RectangularMesh<1, dealii::parallel::shared::Triangulation<1>>;
  template class RectangularMesh<2, dealii::parallel::shared::Triangulation<2>>;
  template class RectangularMesh<3, dealii::parallel::shared::Triangulation<3>>;
#endif
} // namespace DiFfRG
