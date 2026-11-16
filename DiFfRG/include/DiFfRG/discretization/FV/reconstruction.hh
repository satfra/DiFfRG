#pragma once

#include <DiFfRG/discretization/FV/assembler/reconstruction_tools.hh>
#include <DiFfRG/discretization/FV/assembler/state.hh>

#include <algorithm>
#include <array>
#include <autodiff/forward/real/real.hpp>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/numbers.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <utility>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    /// Reconstruction algorithms borrowing a complete mesh. All persistent data belongs to its cells and faces.
    template <int dim, typename NumberType, std::size_t n_components> class MeshReconstruction
    {
      static constexpr unsigned int n_faces = 2 * dim;
      using Stencil = CellStencilData<dim, NumberType, n_components>;
      using Mesh = KurganovTadmor::Mesh<dim, NumberType, n_components>;
      using Cell = typename Mesh::Cell;

    public:
      using JacobianDependencies = KurganovTadmor::JacobianDependencies;
      explicit MeshReconstruction(Mesh &mesh) : mesh_(mesh) {}

      template <def::HasReconstructor JacobianReconstructor> void initialize_dependencies()
      {
        for (auto &face : mesh_.faces()) {
          if (!face.at_boundary()) continue;
          const auto &cell = face.minus;
          const auto face_index = face.minus_face_index;
          face.state.boundary = std::make_unique<
              KurganovTadmor::internal::BoundaryReconstructionStencilTopologyData<dim, n_components>>();
          auto &boundary_reconstruction_topology = *face.state.boundary;
          boundary_reconstruction_topology.tangential_ghost_neighbor_valid.fill(false);
          boundary_reconstruction_topology.corner_tangential_stencil_valid.fill(false);
          for (auto &dofs : boundary_reconstruction_topology.primary.dof_indices)
            dofs.fill(dealii::numbers::invalid_dof_index);
          for (auto &topology : boundary_reconstruction_topology.tangential_ghost_neighbors)
            for (auto &dofs : topology.dof_indices)
              dofs.fill(dealii::numbers::invalid_dof_index);
          for (auto &corner_stencils : boundary_reconstruction_topology.corner_tangential_stencils)
            for (auto &topology : corner_stencils)
              for (auto &dofs : topology.dof_indices)
                dofs.fill(dealii::numbers::invalid_dof_index);
          fill_boundary_topology(boundary_reconstruction_topology.primary, cell, face_index);

          if constexpr (dim == 2) {
            const unsigned int normal_axis = face_index / 2;
            const unsigned int tangential_axis = 1U - normal_axis;
            const unsigned int tangential_minus = 2 * tangential_axis;
            const unsigned int tangential_plus = tangential_minus + 1;
            const auto interior_face = dealii::GeometryInfo<dim>::opposite_face[face_index];
            const auto &normal_neighbor = cell.neighbors[interior_face].get();
            const auto &second_normal_neighbor = normal_neighbor.neighbors[interior_face].get();

            for (const auto tangential_face : {tangential_minus, tangential_plus}) {
              const auto &tangential_neighbor = cell.neighbors[tangential_face].get();
              if (!tangential_neighbor.is_boundary_ghost) {
                if (tangential_neighbor.neighbors[face_index].get().is_boundary_ghost) {
                  fill_boundary_topology(boundary_reconstruction_topology.tangential_ghost_neighbors[tangential_face],
                                         tangential_neighbor, face_index);
                  boundary_reconstruction_topology.tangential_ghost_neighbor_valid[tangential_face] = true;
                }
                continue;
              }

              fill_boundary_topology(boundary_reconstruction_topology.corner_tangential_stencils[tangential_face][0],
                                     cell, tangential_face);
              fill_boundary_topology(boundary_reconstruction_topology.corner_tangential_stencils[tangential_face][1],
                                     normal_neighbor, tangential_face);
              fill_boundary_topology(boundary_reconstruction_topology.corner_tangential_stencils[tangential_face][2],
                                     second_normal_neighbor, tangential_face);
              boundary_reconstruction_topology.corner_tangential_stencil_valid[tangential_face] = true;
            }
          }
        }
        for (auto &face : mesh_.faces())
          build_face_dependencies<JacobianReconstructor>(face.minus, face.minus_face_index, face.state.dependencies);
        for (auto &cell : mesh_.cells())
          build_source_dependencies(cell, cell.state.source_dependencies);
      }

      /// Copy geometry and bindings into the numerical stencil expected by existing reconstructors.
      /// No second cell/neighbor topology is stored.
      void initialize_cell_stencil(const unsigned int i, Stencil &stencil) const
      {
        const auto &geometry = mesh_.cells()[i];
        stencil.cell.x = geometry.x;
        stencil.cell.dof_indices = mesh_.cells()[i].dof_indices;
        for (unsigned int f = 0; f < geometry.faces.size(); ++f) {
          stencil.boundary_ids[f] = geometry.boundary_id(f);
          stencil.face_centers[f] = geometry.faces[f].get().center;
          const auto &neighbor = geometry.neighbors[f].get();
          if (neighbor.is_boundary_ghost) {
            stencil.neighbor_x[f] = stencil.face_centers[f];
            stencil.neighbor_dof_indices[f].fill(dealii::numbers::invalid_dof_index);
          } else {
            stencil.neighbor_x[f] = neighbor.x;
            stencil.neighbor_dof_indices[f] = neighbor.dof_indices;
          }
        }
      }

      template <typename VectorType, typename Model>
      void fill_cell_stencil(const unsigned int i, const VectorType &solution, Model &model, Stencil &stencil) const
      {
        initialize_cell_stencil(i, stencil);
        refresh_cell_stencil(i, solution, model, stencil);
      }

      template <typename VectorType>
      void fill_cell_values(const unsigned int i, const VectorType &solution,
                            CellData<dim, NumberType, n_components> &data) const
      {
        data.x = mesh_.cells()[i].x;
        data.dof_indices = mesh_.cells()[i].dof_indices;
        for (unsigned int c = 0; c < n_components; ++c)
          data.u[c] = solution(data.dof_indices[c]);
      }

      template <typename BoundaryNumberType, typename VectorType>
      auto boundary_stencil(const unsigned int i, const unsigned int f, const VectorType &solution) const
      {
        return KurganovTadmor::internal::fill_boundary_stencil_from_topology<BoundaryNumberType, dim, n_components>(
            boundary_stencil_topology(i, f).primary, solution);
      }

      template <typename BoundaryNumberType, typename VectorType>
      auto boundary_reconstruction_stencil(const unsigned int i, const unsigned int f, const VectorType &solution) const
      {
        return KurganovTadmor::internal::fill_boundary_reconstruction_stencil_from_topology<BoundaryNumberType, dim,
                                                                                            n_components>(
            boundary_stencil_topology(i, f), solution);
      }

      template <typename VectorType, typename Model, typename InputStencil>
      auto tag_cell_stencil(const unsigned int i, const InputStencil &stencil, const VectorType &solution, Model &model,
                            const dealii::types::global_dof_index dof) const
      {
        auto tagged = KurganovTadmor::internal::tag_cell_stencil_dofs<dim, NumberType, n_components>(
            materialize_stencil(stencil), dof);
        const auto &geometry = mesh_.cells()[i];
        for (unsigned int f = 0; f < geometry.faces.size(); ++f) {
          if (!geometry.neighbors[f].get().is_boundary_ghost) continue;
          const auto boundary = boundary_stencil<NumberType>(i, f, solution);
          auto boundary_ad =
              KurganovTadmor::internal::tag_boundary_stencil_dofs<dim, NumberType, n_components>(boundary, dof);
          KurganovTadmor::internal::populate_boundary_neighbor_from_model_stencil(boundary_ad, tagged, f,
                                                                                  stencil.face_center(f), model);
        }
        return tagged;
      }

      /// Refresh real values, then physical ghosts, then each shared face. Each parallel phase completes first.
      template <def::HasReconstructor ActiveReconstructor, typename VectorType, typename Model>
      void reconstruct(const VectorType &solution_global, Model &model) const
      {
        for (auto &face : mesh_.faces())
          face.state.reconstruction_valid = false;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mesh_.cells().size()), [&](const auto &r) {
          for (size_t i = r.begin(); i != r.end(); ++i) {
            auto &cell = mesh_.cells()[i];
            for (unsigned int c = 0; c < n_components; ++c)
              cell.state.u[c] = solution_global(cell.dof_indices[c]);
          }
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mesh_.ghost_cells().size()), [&](const auto &r) {
          for (size_t g = r.begin(); g != r.end(); ++g) {
            auto &ghost = mesh_.ghost_cells()[g];
            const auto &face = ghost.faces[0].get();
            const auto i = face.minus.index;
            const auto f = face.minus_face_index;
            Stencil temporary{};
            initialize_cell_stencil(i, temporary);
            temporary.cell.u = face.minus.state.u;
            auto boundary = boundary_stencil<NumberType>(i, f, solution_global);
            KurganovTadmor::internal::populate_boundary_neighbor_from_model_stencil(boundary, temporary, f, face.center,
                                                                                    model);
            ghost.x = temporary.neighbor_x[f];
            ghost.state.u = temporary.neighbor_u[f];
            ghost.dof_indices = temporary.neighbor_dof_indices[f];
          }
        });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, mesh_.faces().size()), [&](const auto &r) {
          for (size_t k = r.begin(); k != r.end(); ++k) {
            auto &face = mesh_.faces()[k];
            if (face.at_boundary()) {
              const auto boundary =
                  boundary_reconstruction_stencil<NumberType>(face.minus.index, face.minus_face_index, solution_global);
              const auto sides = KurganovTadmor::internal::make_model_boundary_reconstruction_side_stencils(
                  boundary, materialize_stencil(face.minus), face.center, model);
              face.state.reconstruction = KurganovTadmor::internal::compute_interior_face_reconstruction_state<
                  ActiveReconstructor, Model::use_diffusion_flux>(sides.first, sides.second, face.center);
            } else {
              face.state.reconstruction = KurganovTadmor::internal::compute_interior_face_reconstruction_state<
                  ActiveReconstructor, Model::use_diffusion_flux>(face.minus, face.plus, face.center);
            }
            face.state.reconstruction_valid = true;
          }
        });
      }

    private:
      template <typename VectorType, typename Model>
      void refresh_cell_stencil(const unsigned int i, const VectorType &solution, Model &model, Stencil &stencil) const
      {
        for (unsigned int c = 0; c < n_components; ++c)
          stencil.cell.u[c] = solution(stencil.cell.dof_indices[c]);
        for (unsigned int f = 0; f < n_faces; ++f) {
          if (!mesh_.cells()[i].neighbors[f].get().is_boundary_ghost) {
            for (unsigned int c = 0; c < n_components; ++c)
              stencil.neighbor_u[f][c] = solution(stencil.neighbor_dof_indices[f][c]);
          } else {
            auto boundary = boundary_stencil<NumberType>(i, f, solution);
            KurganovTadmor::internal::populate_boundary_neighbor_from_model_stencil(boundary, stencil, f,
                                                                                    stencil.face_centers[f], model);
          }
        }
      }

      const auto &boundary_stencil_topology(const unsigned int i, const unsigned int f) const
      {
        const auto &ghost = mesh_.cells()[i].neighbors[f].get();
        Assert(ghost.is_boundary_ghost, dealii::ExcMessage("Not a physical boundary face."));
        return *mesh_.cells()[i].faces[f].get().state.boundary;
      }

      void fill_boundary_topology(
          KurganovTadmor::internal::BoundaryStencilTopologyData<dim, n_components> &boundary_topology, const Cell &cell,
          const unsigned int boundary_face_no) const
      {
        using namespace def::BoundaryStencilIndex;
        boundary_topology.lower_boundary = boundary_face_no % 2 == 0;
        boundary_topology.cell_face = boundary_face_no;
        boundary_topology.face_center = cell.faces[boundary_face_no].get().center;
        boundary_topology.ghost_center = boundary_topology.lower_boundary ? lower_inner : upper_inner;
        boundary_topology.ghost_left = boundary_topology.lower_boundary ? lower_outer : physical_cell;
        boundary_topology.ghost_right = boundary_topology.lower_boundary ? physical_cell : upper_outer;
        for (auto &dofs : boundary_topology.dof_indices)
          dofs.fill(dealii::numbers::invalid_dof_index);

        boundary_topology.x[physical_cell] = cell.x;
        boundary_topology.dof_indices[physical_cell] = cell.dof_indices;

        const auto interior_face = dealii::GeometryInfo<dim>::opposite_face[boundary_face_no];
        const auto &neighbor = cell.neighbors[interior_face].get();
        AssertThrow(
            !neighbor.is_boundary_ghost,
            dealii::ExcMessage("KT boundary stencil requires at least two interior cells behind the boundary face."));
        const auto &next_neighbor = neighbor.neighbors[interior_face].get();
        AssertThrow(
            !next_neighbor.is_boundary_ghost,
            dealii::ExcMessage("KT boundary stencil requires a second interior cell behind the boundary face."));
        const auto &first_interior_dofs = neighbor.dof_indices;
        const auto &second_interior_dofs = next_neighbor.dof_indices;

        if (boundary_topology.lower_boundary) {
          boundary_topology.x[upper_inner] = neighbor.x;
          boundary_topology.x[upper_outer] = next_neighbor.x;
          boundary_topology.dof_indices[upper_inner] = first_interior_dofs;
          boundary_topology.dof_indices[upper_outer] = second_interior_dofs;
        } else {
          boundary_topology.x[lower_inner] = neighbor.x;
          boundary_topology.x[lower_outer] = next_neighbor.x;
          boundary_topology.dof_indices[lower_inner] = first_interior_dofs;
          boundary_topology.dof_indices[lower_outer] = second_interior_dofs;
        }
      }

      template <typename DoFContainer>
      static void append_dofs(std::vector<dealii::types::global_dof_index> &target, const DoFContainer &source)
      {
        target.insert(target.end(), source.begin(), source.end());
      }

      template <typename DoFContainer>
      static void append_valid_dofs(std::vector<dealii::types::global_dof_index> &target, const DoFContainer &source)
      {
        for (const auto dof : source)
          if (dof != dealii::numbers::invalid_dof_index) target.push_back(dof);
      }

      static void sort_unique_dofs(std::vector<dealii::types::global_dof_index> &dofs)
      {
        std::sort(dofs.begin(), dofs.end());
        dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
      }

      /**
       * @brief The dofs @p cell's reconstruction stencil reads: the cell and its 2*dim face neighbours, or, across a
       * physical boundary face, the dofs the model's ghost value there is built from.
       */
      void append_cell_stencil_dofs(std::vector<dealii::types::global_dof_index> &from_dofs, const Cell &cell) const
      {
        const auto &cell_data = cell;
        append_dofs(from_dofs, cell_data.dof_indices);
        for (unsigned int face_index = 0; face_index < cell.faces.size(); ++face_index) {
          if (cell.neighbors[face_index].get().is_boundary_ghost)
            append_boundary_reconstruction_dofs(from_dofs, cell.index, face_index);
          else
            append_valid_dofs(from_dofs, cell.neighbors[face_index].get().dof_indices);
        }
      }

      void append_boundary_reconstruction_dofs(std::vector<dealii::types::global_dof_index> &from_dofs,
                                               const unsigned int cell_index, const unsigned int face_index) const
      {
        const auto append_boundary_stencil_dofs =
            [&](const KurganovTadmor::internal::BoundaryStencilTopologyData<dim, n_components> &topology) {
              for (const auto &dofs : topology.dof_indices)
                append_valid_dofs(from_dofs, dofs);
            };

        const auto &topology = boundary_stencil_topology(cell_index, face_index);
        append_boundary_stencil_dofs(topology.primary);
        for (size_t face = 0; face < topology.tangential_ghost_neighbor_valid.size(); ++face) {
          if (topology.tangential_ghost_neighbor_valid[face])
            append_boundary_stencil_dofs(topology.tangential_ghost_neighbors[face]);
          if (topology.corner_tangential_stencil_valid[face]) {
            for (const auto &corner_stencil : topology.corner_tangential_stencils[face])
              append_boundary_stencil_dofs(corner_stencil);
          }
        }
      }

      template <def::HasReconstructor JacobianReconstructor>
      void build_face_dependencies(const Cell &cell, const unsigned int face_index,
                                   JacobianDependencies &dependencies) const
      {
        const auto &cell_data = cell;

        dependencies.to_dofs.clear();
        append_dofs(dependencies.to_dofs, cell_data.dof_indices);
        if (!cell.neighbors[face_index].get().is_boundary_ghost)
          append_dofs(dependencies.to_dofs, cell.neighbors[face_index].get().dof_indices);

        // The two traces of a face are reconstructed from the stencils of the two cells (a boundary face's ghost
        // side from the boundary stencil), so a row reaches jacobian_stencil_radius = 2 cells. A first-order
        // reconstruction reads the two cells only.
        dependencies.from_dofs = dependencies.to_dofs;
        if (cell.neighbors[face_index].get().is_boundary_ghost)
          append_boundary_reconstruction_dofs(dependencies.from_dofs, cell.index, face_index);
        if constexpr (JacobianReconstructor::jacobian_stencil_radius > 1) {
          append_cell_stencil_dofs(dependencies.from_dofs, cell);
          if (!cell.neighbors[face_index].get().is_boundary_ghost)
            append_cell_stencil_dofs(dependencies.from_dofs, cell.neighbors[face_index].get());
        }
        sort_unique_dofs(dependencies.from_dofs);
      }

      /**
       * @brief The dofs the nonlocal part of the source jacobian writes to and reads from.
       *
       * Radius 1, tighter than the face dependencies: the gradient handed to model.source() is
       * reconstructed from the cell and its 2*dim face neighbours only. Across a physical boundary face
       * that neighbour is a model ghost, so the boundary stencil's dofs take its place.
       */
      void build_source_dependencies(const Cell &cell, JacobianDependencies &dependencies) const
      {
        const auto &cell_data = cell;

        dependencies.to_dofs.clear();
        append_dofs(dependencies.to_dofs, cell_data.dof_indices);

        dependencies.from_dofs.clear();
        append_cell_stencil_dofs(dependencies.from_dofs, cell);
        sort_unique_dofs(dependencies.from_dofs);
      }

      Mesh &mesh_;
    };
  } // namespace FV
} // namespace DiFfRG
