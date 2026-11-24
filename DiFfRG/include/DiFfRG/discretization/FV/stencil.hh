#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>

#include <array>
#include <cstddef>
#include <deal.II/base/numbers.h>

namespace DiFfRG
{
  namespace FV
  {
    template <int dim, typename NumberType, size_t n_components> struct CellData {
      dealii::Point<dim> x;
      std::array<NumberType, n_components> u;
      std::array<dealii::types::global_dof_index, n_components> dof_indices;
    };

    template <int dim, typename NumberType, size_t n_components> struct CellStencilData {
      static constexpr size_t n_faces = 2 * dim;
      static constexpr int dimension = dim;
      static constexpr size_t components = n_components;
      using scalar_type = NumberType;
      CellData<dim, NumberType, n_components> cell;
      std::array<dealii::Point<dim>, n_faces> neighbor_x{};
      std::array<std::array<NumberType, n_components>, n_faces> neighbor_u{};
      std::array<std::array<dealii::types::global_dof_index, n_components>, n_faces> neighbor_dof_indices{};
      std::array<dealii::types::boundary_id, n_faces> boundary_ids;
      std::array<dealii::Point<dim>, n_faces> face_centers;
      const auto &center() const { return cell.x; }
      const auto &values() const { return cell.u; }
      const auto &dofs() const { return cell.dof_indices; }
      const auto &neighbor_center(unsigned int f) const { return neighbor_x[f]; }
      const auto &neighbor_values(unsigned int f) const { return neighbor_u[f]; }
      const auto &neighbor_dofs(unsigned int f) const { return neighbor_dof_indices[f]; }
      const auto &face_center(unsigned int f) const { return face_centers[f]; }
      auto boundary_id(unsigned int f) const { return boundary_ids[f]; }
    };

    /// An owned copy only where a boundary callback or AD seeding needs mutable inputs.
    template <typename Stencil> auto materialize_stencil(const Stencil &view)
    {
      CellStencilData<Stencil::dimension, typename Stencil::scalar_type, Stencil::components> result{};
      result.cell = {view.center(), view.values(), view.dofs()};
      for (unsigned int f = 0; f < Stencil::n_faces; ++f) {
        result.neighbor_x[f] = view.neighbor_center(f);
        result.neighbor_u[f] = view.neighbor_values(f);
        result.neighbor_dof_indices[f] = view.neighbor_dofs(f);
        result.face_centers[f] = view.face_center(f);
        result.boundary_ids[f] = view.boundary_id(f);
      }
      return result;
    }
  } // namespace FV
} // namespace DiFfRG
