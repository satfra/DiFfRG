#pragma once

#include <DiFfRG/discretization/FV/mesh.hh>
#include <DiFfRG/discretization/common/cell_geometry.hh>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_values.h>

#include <array>
#include <cstddef>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    /**
     * @brief The deal.II backend of the finite-volume assembler.
     *
     * References the existing discretization, which must outlive the backend. Imports cell geometry into the
     * assembler's complete FV mesh, populates cell bindings, and maps local deal.II dofs to model components.
     */
    template <typename Discretization> class DealIIBackend
    {
      using Components = typename Discretization::Components;
      static constexpr std::size_t n_components = Components::count_fe_functions(0);

    public:
      explicit DealIIBackend(Discretization &discretization) : discretization_(discretization)
      {
        const auto &fe = discretization_.get_fe();
        AssertThrow(fe.dofs_per_cell == n_components, dealii::ExcMessage("FV backend expects one dof per component."));
        for (unsigned int i = 0; i < n_components; ++i)
          local_component_of_dof_[i] = fe.system_to_component_index(i).first;
      }

      const auto &get_triangulation() const { return discretization_.get_triangulation(); }

      /// Rebuild the caller-owned geometry and reference connectivity, including physical boundary ghosts.
      template <typename CellState, typename FaceState>
      void populate(Mesh<Discretization::dim, CellState, FaceState> &topology,
                    const dealii::Quadrature<Discretization::dim> &quadrature) const
      {
        topology.faces_.clear();
        topology.cells_.clear();
        topology.ghost_cells_.clear();
        const auto n_real_cells = discretization_.get_triangulation().n_active_cells();
        unsigned int n_boundary_faces = 0;
        for (const auto &cell : discretization_.get_dof_handler().active_cell_iterators())
          for (const auto f : cell->face_indices())
            if (cell->at_boundary(f)) ++n_boundary_faces;
        // Allocate every cell before constructing faces that reference them.
        topology.cells_.resize(n_real_cells);
        topology.ghost_cells_.resize(n_boundary_faces);
        std::vector<std::vector<unsigned int>> cell_faces(topology.cells_.size());
        std::vector<unsigned int> ghost_faces(n_boundary_faces);
        for (const auto &cell : discretization_.get_dof_handler().active_cell_iterators()) {
          auto &geometry = topology.cells_[cell->active_cell_index()];
          geometry.index = cell->active_cell_index();
          geometry.x = cell->center();
          cell_faces[cell->active_cell_index()].resize(2 * Discretization::dim);
        }

        unsigned int ghost_index = 0;
        topology.faces_.reserve(n_real_cells * Discretization::dim + n_boundary_faces);
        for (const auto &cell : discretization_.get_dof_handler().active_cell_iterators()) {
          const auto minus_index = cell->active_cell_index();
          auto &minus = topology.cells_[minus_index];
          for (const auto f : cell->face_indices()) {
            unsigned int plus_index;
            unsigned int plus_face_index;
            const auto mesh_face = cell->face(f);
            const auto center = mesh_face->center();
            const bool boundary = cell->at_boundary(f);
            if (boundary) {
              plus_index = ghost_index++;
              plus_face_index = 0;
              auto &ghost = topology.ghost_cells_[plus_index];
              ghost.index = n_real_cells + plus_index;
              ghost.is_boundary_ghost = true;
              ghost.x = minus.x + 2. * (center - minus.x);
            } else {
              const auto neighbor = DiFfRG::internal::active_face_neighbor(cell, f);
              plus_index = neighbor->active_cell_index();
              if (plus_index < minus_index) continue;
              plus_face_index = find_neighbor_face(cell, neighbor);
            }

            const auto offset = center - minus.x;
            const auto norm = offset.norm();
            AssertThrow(norm > 0., dealii::ExcMessage("Degenerate FV face normal."));
            double jxw = 1.;
            if constexpr (Discretization::dim > 1) jxw = mesh_face->measure();
            const auto face_index = topology.faces_.size();
            auto &plus = boundary ? topology.ghost_cells_[plus_index] : topology.cells_[plus_index];
            topology.faces_.emplace_back(minus, plus, f, plus_face_index, center, offset / norm, jxw,
                                         boundary ? mesh_face->boundary_id() : dealii::numbers::invalid_boundary_id);
            cell_faces[minus_index][f] = face_index;
            if (boundary)
              ghost_faces[plus_index] = face_index;
            else
              cell_faces[plus_index][plus_face_index] = face_index;
          }
        }

        // Only now bind cell references to faces: no face storage may move after this point.
        for (unsigned int i = 0; i < topology.cells_.size(); ++i) {
          auto &cell = topology.cells_[i];
          cell.faces.reserve(cell_faces[i].size());
          cell.neighbors.reserve(cell_faces[i].size());
          for (const auto face_index : cell_faces[i]) {
            auto &face = topology.faces_[face_index];
            cell.faces.emplace_back(face);
            cell.neighbors.emplace_back(&cell == &face.minus ? face.plus : face.minus);
          }
        }
        for (unsigned int i = 0; i < topology.ghost_cells_.size(); ++i) {
          auto &ghost = topology.ghost_cells_[i];
          auto &face = topology.faces_[ghost_faces[i]];
          ghost.faces.emplace_back(face);
          ghost.neighbors.emplace_back(face.minus);
        }

        dealii::FEValues<Discretization::dim> fe_values(discretization_.get_mapping(), discretization_.get_fe(),
                                                        quadrature,
                                                        dealii::update_quadrature_points | dealii::update_JxW_values);
        std::vector<dealii::types::global_dof_index> dof_indices(n_components);
        for (const auto &cell : discretization_.get_dof_handler().active_cell_iterators()) {
          auto &data = topology.cells_[cell->active_cell_index()];
          cell->get_dof_indices(dof_indices);
          for (unsigned int i = 0; i < n_components; ++i)
            data.dof_indices[i] = dof_indices[i];
          fe_values.reinit(cell);
          data.quadrature_points = fe_values.get_quadrature_points();
          data.jxw.assign(fe_values.get_JxW_values().begin(), fe_values.get_JxW_values().end());
        }
      }

      unsigned int component_of_local_dof(const unsigned int i) const { return local_component_of_dof_[i]; }

    private:
      template <typename Iterator>
      static unsigned int find_neighbor_face(const Iterator &cell, const Iterator &neighbor)
      {
        for (const auto f : neighbor->face_indices()) {
          if (neighbor->at_boundary(f)) continue;
          if (DiFfRG::internal::active_face_neighbor(neighbor, f) == cell) return f;
        }
        AssertThrow(false, dealii::ExcMessage("Could not find reciprocal FV neighbor face."));
        return 0;
      }

      Discretization &discretization_;
      std::array<unsigned int, n_components> local_component_of_dof_{};
    };
  } // namespace FV
} // namespace DiFfRG
