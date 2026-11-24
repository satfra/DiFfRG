#pragma once

#include "DiFfRG/common/math.hh"
#include <DiFfRG/discretization/FV/face_state.hh>
#include <DiFfRG/discretization/FV/stencil.hh>

#include <DiFfRG/discretization/FV/reconstructor/advection/abstract_reconstructor.hh>
#include <DiFfRG/discretization/FV/reconstructor/diffusion/corrected_weighted_least_squares_reconstructor.hh>
#include <DiFfRG/discretization/common/types.hh>
#include <DiFfRG/model/fv_boundaries.hh>

#include <algorithm>
#include <array>
#include <autodiff/forward/real/real.hpp>
#include <cmath>
#include <cstddef>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <utility>
#include <vector>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      namespace internal
      {
        template <int dim, typename NumberType, size_t n_components>
        using GradientType = def::GradientType<dim, NumberType, n_components>;

        template <int dim, typename NumberType, size_t n_components>
        using ThirdDerivativeType = def::ThirdDerivativeType<dim, NumberType, n_components>;

        template <int dim, typename NumberType, size_t n_components> struct BoundaryStencilData;

        using def::ReconstructionDerivativeData;
        using FV::CellData;
        using FV::CellStencilData;
        using FV::FaceReconstructionState;

        template <int dim, typename NumberType, size_t n_components>
        using DiffusionFaceState = def::DiffusionFaceState<dim, NumberType, n_components>;

        template <int dim> struct BoundaryStencilIndex {
          static_assert(dim == 1, "Boundary stencil indices currently support only dim=1.");
        };

        template <> struct BoundaryStencilIndex<1> {
          static constexpr size_t lower_outer = 0;
          static constexpr size_t lower_inner = 1;
          static constexpr size_t physical_cell = 2;
          static constexpr size_t upper_inner = 3;
          static constexpr size_t upper_outer = 4;
        };

        template <int dim, typename NumberType, size_t n_components>
        FaceReconstructionState<dim, NumberType, n_components>
        reverse_face_reconstruction(const FaceReconstructionState<dim, NumberType, n_components> &state)
        {
          FaceReconstructionState<dim, NumberType, n_components> reversed{};
          reversed.u_minus = state.u_plus;
          reversed.u_plus = state.u_minus;
          reversed.center_grad_minus = state.center_grad_plus;
          reversed.center_grad_plus = state.center_grad_minus;
          reversed.face_grad_minus = state.face_grad_plus;
          reversed.face_grad_plus = state.face_grad_minus;
          reversed.diffusion_u_minus = state.diffusion_u_plus;
          reversed.diffusion_u_plus = state.diffusion_u_minus;
          reversed.diffusion_grad_minus = state.diffusion_grad_plus;
          reversed.diffusion_grad_plus = state.diffusion_grad_minus;
          reversed.diffusion_u = state.diffusion_u;
          reversed.diffusion_grad = state.diffusion_grad;
          reversed.third_derivatives_minus = state.third_derivatives_plus;
          reversed.third_derivatives_plus = state.third_derivatives_minus;
          return reversed;
        }

        template <int dim, typename NumberType, size_t n_components>
        std::array<NumberType, n_components>
        reconstruct_u(const std::array<NumberType, n_components> &u_center, const dealii::Point<dim> &center,
                      const dealii::Point<dim> &x, const GradientType<dim, NumberType, n_components> &u_grad)
        {
          AssertDimension(u_center.size(), n_components);
          std::array<NumberType, n_components> result;
          for (size_t c = 0; c < n_components; ++c) {
            result[c] = u_center[c];
            result[c] += dealii::scalar_product(u_grad[c], x - center);
          }
          return result;
        }

        template <int dim, typename NumberType, size_t n_components> struct FourPointStencil {
          std::array<dealii::Point<dim>, 4> x{};
          std::array<std::array<NumberType, n_components>, 4> u{};
        };

        template <int dim, typename NumberType, size_t n_components>
        void sort_four_point_stencil(FourPointStencil<dim, NumberType, n_components> &stencil)
        {
          static_assert(dim == 1, "Third-derivative FV stencils currently support only dim=1.");

          std::array<size_t, 4> order = {0, 1, 2, 3};
          std::sort(order.begin(), order.end(),
                    [&](const size_t left, const size_t right) { return stencil.x[left][0] < stencil.x[right][0]; });

          auto x_sorted = stencil.x;
          auto u_sorted = stencil.u;
          for (size_t i = 0; i < order.size(); ++i) {
            x_sorted[i] = stencil.x[order[i]];
            u_sorted[i] = stencil.u[order[i]];
          }
          stencil.x = x_sorted;
          stencil.u = u_sorted;
        }

        template <typename Stencil>
        auto make_interior_third_derivative_stencil(const Stencil &minus_stencil, const Stencil &plus_stencil,
                                                    const dealii::Point<Stencil::dimension> &x_q)
        {
          constexpr int dim = Stencil::dimension;
          constexpr size_t n_components = Stencil::components;
          using NumberType = typename Stencil::scalar_type;
          static_assert(dim == 1, "Third-derivative FV stencils currently support only dim=1.");

          const auto outer_face = [&](const Stencil &stencil) { return stencil.center()[0] < x_q[0] ? 0U : 1U; };

          FourPointStencil<dim, NumberType, n_components> result{};
          const auto minus_outer = outer_face(minus_stencil);
          const auto plus_outer = outer_face(plus_stencil);
          result.x = {minus_stencil.neighbor_center(minus_outer), minus_stencil.center(), plus_stencil.center(),
                      plus_stencil.neighbor_center(plus_outer)};
          result.u = {minus_stencil.neighbor_values(minus_outer), minus_stencil.values(), plus_stencil.values(),
                      plus_stencil.neighbor_values(plus_outer)};
          sort_four_point_stencil(result);
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        FourPointStencil<dim, NumberType, n_components> make_boundary_third_derivative_stencil(
            const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil)
        {
          static_assert(dim == 1, "Third-derivative boundary stencils currently support only dim=1.");

          using BoundaryIndex = BoundaryStencilIndex<dim>;
          FourPointStencil<dim, NumberType, n_components> result{};
          if (boundary_stencil.lower_boundary) {
            result.x = {boundary_stencil.x[BoundaryIndex::lower_outer], boundary_stencil.x[BoundaryIndex::lower_inner],
                        boundary_stencil.x[BoundaryIndex::physical_cell],
                        boundary_stencil.x[BoundaryIndex::upper_inner]};
            result.u = {boundary_stencil.u[BoundaryIndex::lower_outer], boundary_stencil.u[BoundaryIndex::lower_inner],
                        boundary_stencil.u[BoundaryIndex::physical_cell],
                        boundary_stencil.u[BoundaryIndex::upper_inner]};
          } else {
            result.x = {boundary_stencil.x[BoundaryIndex::lower_inner],
                        boundary_stencil.x[BoundaryIndex::physical_cell],
                        boundary_stencil.x[BoundaryIndex::upper_inner], boundary_stencil.x[BoundaryIndex::upper_outer]};
            result.u = {boundary_stencil.u[BoundaryIndex::lower_inner],
                        boundary_stencil.u[BoundaryIndex::physical_cell],
                        boundary_stencil.u[BoundaryIndex::upper_inner], boundary_stencil.u[BoundaryIndex::upper_outer]};
          }
          sort_four_point_stencil(result);
          return result;
        }

        template <typename Stencil>
        auto compute_diffusion_face_state(const Stencil &minus_stencil, const Stencil &plus_stencil)
        {
          using DiffusionReconstructor =
              def::CorrectedWeightedLeastSquaresDiffusionReconstructor<Stencil::dimension,
                                                                       typename Stencil::scalar_type>;
          return DiffusionReconstructor::template compute_face_state<Stencil::components>(minus_stencil, plus_stencil);
        }

        template <int dim, typename NumberType, size_t n_components>
        std::array<ReconstructionDerivativeData<dim, NumberType, n_components>, 2> extract_diffusion_face_derivatives(
            const DiffusionFaceState<dim, autodiff::Real<1, NumberType>, n_components> &state)
        {
          using DiffusionReconstructor = def::CorrectedWeightedLeastSquaresDiffusionReconstructor<dim, NumberType>;
          return DiffusionReconstructor::template extract_derivatives<n_components>(state);
        }

        /// The face state from the two side stencils. Without @p with_diffusion, the diffusion entries (and the 1D
        /// third derivatives, which only the diffusion flux reads) stay zero.
        template <def::HasReconstructor Reconstructor, bool with_diffusion = true, typename Stencil,
                  typename DiffusionReconstructor = def::CorrectedWeightedLeastSquaresDiffusionReconstructor<
                      Stencil::dimension, typename Stencil::scalar_type>>
        auto compute_interior_face_reconstruction_state(const Stencil &minus_stencil, const Stencil &plus_stencil,
                                                        const dealii::Point<Stencil::dimension> &x_q)
        {
          constexpr int dim = Stencil::dimension;
          constexpr size_t n_components = Stencil::components;
          using NumberType = typename Stencil::scalar_type;
          FaceReconstructionState<dim, NumberType, n_components> state{};
          state.center_grad_minus = Reconstructor::template compute_gradient<n_components>(minus_stencil);
          state.center_grad_plus = Reconstructor::template compute_gradient<n_components>(plus_stencil);
          state.face_grad_minus = Reconstructor::template compute_gradient_at_point<n_components>(minus_stencil, x_q);
          state.face_grad_plus = Reconstructor::template compute_gradient_at_point<n_components>(plus_stencil, x_q);
          state.u_minus = reconstruct_u(minus_stencil.values(), minus_stencil.center(), x_q, state.center_grad_minus);
          state.u_plus = reconstruct_u(plus_stencil.values(), plus_stencil.center(), x_q, state.center_grad_plus);
          if constexpr (!with_diffusion) return state;
          const auto diffusion_state =
              DiffusionReconstructor::template compute_face_state<n_components>(minus_stencil, plus_stencil);
          state.diffusion_u_minus = diffusion_state.u_minus;
          state.diffusion_u_plus = diffusion_state.u_plus;
          state.diffusion_grad_minus = diffusion_state.grad_minus;
          state.diffusion_grad_plus = diffusion_state.grad_plus;
          state.diffusion_u = diffusion_state.u;
          state.diffusion_grad = diffusion_state.grad;
          if constexpr (dim == 1) {
            const auto third_derivative_stencil =
                make_interior_third_derivative_stencil(minus_stencil, plus_stencil, x_q);
            const auto third_derivatives = Reconstructor::template compute_third_derivatives_at_face<n_components>(
                third_derivative_stencil.x, third_derivative_stencil.u);
            state.third_derivatives_minus = third_derivatives;
            state.third_derivatives_plus = third_derivatives;
          }
          return state;
        }

        template <def::HasReconstructor Reconstructor, typename Stencil>
        auto reconstruct_u_derivative(const Stencil &stencil, const dealii::Point<Stencil::dimension> &x)
        {
          using Scalar = std::remove_cvref_t<decltype(autodiff::derivative(stencil.values()[0]))>;
          constexpr size_t n_components = Stencil::components;
          const auto gradient = Reconstructor::template compute_gradient_derivative<n_components>(stencil);
          std::array<Scalar, n_components> result{};
          for (size_t c = 0; c < n_components; ++c)
            result[c] =
                autodiff::derivative(stencil.values()[c]) + dealii::scalar_product(gradient[c], x - stencil.center());
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        bool is_lower_boundary_stencil(const std::array<dealii::Point<dim>, 2 * dim + 3> &x_stencil,
                                       const dealii::Point<dim> &x_face)
        {
          static_assert(dim == 1, "Paper-style boundary stencil orientation currently supports only dim=1.");

          return x_face[0] <= x_stencil[BoundaryStencilIndex<dim>::physical_cell][0];
        }

        template <int dim, typename NumberType, size_t n_components> struct BoundaryStencilData {
          static constexpr size_t stencil_size = 2 * dim + 3;
          std::array<dealii::Point<dim>, stencil_size> x{};
          std::array<std::array<NumberType, n_components>, stencil_size> u{};
          std::array<std::array<dealii::types::global_dof_index, n_components>, stencil_size> dof_indices{};
          bool lower_boundary = true;
          unsigned int cell_face = 0;
          dealii::Point<dim> face_center{};
          size_t ghost_center = def::BoundaryStencilIndex::lower_inner;
          size_t ghost_left = def::BoundaryStencilIndex::lower_outer;
          size_t ghost_right = def::BoundaryStencilIndex::physical_cell;
        };

        template <int dim, size_t n_components> struct BoundaryStencilTopologyData {
          static constexpr size_t stencil_size = 2 * dim + 3;
          std::array<dealii::Point<dim>, stencil_size> x{};
          std::array<std::array<dealii::types::global_dof_index, n_components>, stencil_size> dof_indices{};
          bool lower_boundary = true;
          unsigned int cell_face = 0;
          dealii::Point<dim> face_center{};
          size_t ghost_center = def::BoundaryStencilIndex::lower_inner;
          size_t ghost_left = def::BoundaryStencilIndex::lower_outer;
          size_t ghost_right = def::BoundaryStencilIndex::physical_cell;
        };

        template <int dim, typename NumberType, size_t n_components> struct BoundaryReconstructionStencilData {
          static constexpr size_t n_faces = 2 * dim;
          BoundaryStencilData<dim, NumberType, n_components> primary{};
          std::array<BoundaryStencilData<dim, NumberType, n_components>, n_faces> tangential_ghost_neighbors{};
          std::array<bool, n_faces> tangential_ghost_neighbor_valid{};
          std::array<std::array<BoundaryStencilData<dim, NumberType, n_components>, 3>, n_faces>
              corner_tangential_stencils{};
          std::array<bool, n_faces> corner_tangential_stencil_valid{};
        };

        template <int dim, size_t n_components> struct BoundaryReconstructionStencilTopologyData {
          static constexpr size_t n_faces = 2 * dim;
          BoundaryStencilTopologyData<dim, n_components> primary{};
          std::array<BoundaryStencilTopologyData<dim, n_components>, n_faces> tangential_ghost_neighbors{};
          std::array<bool, n_faces> tangential_ghost_neighbor_valid{};
          std::array<std::array<BoundaryStencilTopologyData<dim, n_components>, 3>, n_faces>
              corner_tangential_stencils{};
          std::array<bool, n_faces> corner_tangential_stencil_valid{};
        };

        template <typename BoundaryNumberType, int dim, size_t n_components, typename VectorType>
        BoundaryStencilData<dim, BoundaryNumberType, n_components>
        fill_boundary_stencil_from_topology(const BoundaryStencilTopologyData<dim, n_components> &topology,
                                            const VectorType &solution_global)
        {
          BoundaryStencilData<dim, BoundaryNumberType, n_components> boundary_stencil{};
          boundary_stencil.x = topology.x;
          boundary_stencil.dof_indices = topology.dof_indices;
          boundary_stencil.lower_boundary = topology.lower_boundary;
          boundary_stencil.cell_face = topology.cell_face;
          boundary_stencil.face_center = topology.face_center;
          boundary_stencil.ghost_center = topology.ghost_center;
          boundary_stencil.ghost_left = topology.ghost_left;
          boundary_stencil.ghost_right = topology.ghost_right;
          for (size_t stencil_index = 0; stencil_index < boundary_stencil.u.size(); ++stencil_index) {
            for (size_t c = 0; c < n_components; ++c) {
              const auto dof = topology.dof_indices[stencil_index][c];
              boundary_stencil.u[stencil_index][c] = dof == dealii::numbers::invalid_dof_index
                                                         ? BoundaryNumberType{}
                                                         : BoundaryNumberType(solution_global(dof));
            }
          }
          return boundary_stencil;
        }

        template <typename BoundaryNumberType, int dim, size_t n_components, typename VectorType>
        BoundaryReconstructionStencilData<dim, BoundaryNumberType, n_components>
        fill_boundary_reconstruction_stencil_from_topology(
            const BoundaryReconstructionStencilTopologyData<dim, n_components> &topology,
            const VectorType &solution_global)
        {
          BoundaryReconstructionStencilData<dim, BoundaryNumberType, n_components> result{};
          result.primary = fill_boundary_stencil_from_topology<BoundaryNumberType, dim, n_components>(topology.primary,
                                                                                                      solution_global);
          result.tangential_ghost_neighbor_valid = topology.tangential_ghost_neighbor_valid;
          result.corner_tangential_stencil_valid = topology.corner_tangential_stencil_valid;
          for (size_t face = 0; face < result.n_faces; ++face) {
            if (result.tangential_ghost_neighbor_valid[face])
              result.tangential_ghost_neighbors[face] =
                  fill_boundary_stencil_from_topology<BoundaryNumberType, dim, n_components>(
                      topology.tangential_ghost_neighbors[face], solution_global);
            if (result.corner_tangential_stencil_valid[face]) {
              for (size_t stencil_index = 0; stencil_index < result.corner_tangential_stencils[face].size();
                   ++stencil_index) {
                result.corner_tangential_stencils[face][stencil_index] =
                    fill_boundary_stencil_from_topology<BoundaryNumberType, dim, n_components>(
                        topology.corner_tangential_stencils[face][stencil_index], solution_global);
              }
            }
          }
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_boundary_side_stencil(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil,
                                   const size_t center_index, const size_t left_index, const size_t right_index)
        {
          static_assert(dim == 1, "Paper-style boundary side stencils currently support only dim=1.");

          CellStencilData<dim, NumberType, n_components> result{};
          result.boundary_ids.fill(dealii::numbers::invalid_boundary_id);
          result.face_centers = {};

          result.cell.x = boundary_stencil.x[center_index];
          result.cell.u = boundary_stencil.u[center_index];
          result.cell.dof_indices = boundary_stencil.dof_indices[center_index];
          result.neighbor_x[0] = boundary_stencil.x[left_index];
          result.neighbor_x[1] = boundary_stencil.x[right_index];
          result.neighbor_u[0] = boundary_stencil.u[left_index];
          result.neighbor_u[1] = boundary_stencil.u[right_index];
          result.neighbor_dof_indices[0] = boundary_stencil.dof_indices[left_index];
          result.neighbor_dof_indices[1] = boundary_stencil.dof_indices[right_index];
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_physical_boundary_side_stencil(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil)
        {
          static_assert(dim == 1, "Paper-style physical boundary side stencils currently support only dim=1.");

          return make_boundary_side_stencil<dim>(boundary_stencil, BoundaryStencilIndex<dim>::physical_cell,
                                                 BoundaryStencilIndex<dim>::lower_inner,
                                                 BoundaryStencilIndex<dim>::upper_inner);
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_ghost_boundary_side_stencil(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil)
        {
          static_assert(dim == 1, "Paper-style ghost boundary side stencils currently support only dim=1.");

          return make_boundary_side_stencil<dim>(boundary_stencil, boundary_stencil.ghost_center,
                                                 boundary_stencil.ghost_left, boundary_stencil.ghost_right);
        }

        template <int dim, typename NumberType, size_t n_components>
        void prepare_boundary_reconstruction_stencil(
            BoundaryStencilData<dim, NumberType, n_components> & /*boundary_stencil*/,
            const CellStencilData<dim, NumberType, n_components> & /*physical_stencil*/,
            const dealii::Point<dim> & /*x_q*/)
          requires(dim == 1)
        {
        }

        template <int dim, typename NumberType, size_t n_components>
        void
        prepare_boundary_reconstruction_stencil(BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil,
                                                const CellStencilData<dim, NumberType, n_components> &physical_stencil,
                                                const dealii::Point<dim> &x_q)
          requires(dim == 2)
        {
          const unsigned int axis = boundary_stencil.cell_face / 2;
          const size_t ghost_center = boundary_stencil.lower_boundary ? def::BoundaryStencilIndex::lower_inner
                                                                      : def::BoundaryStencilIndex::upper_inner;
          const size_t ghost_outer = boundary_stencil.lower_boundary ? def::BoundaryStencilIndex::lower_outer
                                                                     : def::BoundaryStencilIndex::upper_outer;
          boundary_stencil.ghost_center = ghost_center;
          boundary_stencil.ghost_left =
              boundary_stencil.lower_boundary ? ghost_outer : def::BoundaryStencilIndex::physical_cell;
          boundary_stencil.ghost_right =
              boundary_stencil.lower_boundary ? def::BoundaryStencilIndex::physical_cell : ghost_outer;

          const auto &physical_x = boundary_stencil.x[def::BoundaryStencilIndex::physical_cell];
          const double delta = std::abs(physical_x[axis] - x_q[axis]);
          const double direction = boundary_stencil.lower_boundary ? -1.0 : 1.0;
          boundary_stencil.x[ghost_center] = physical_x;
          boundary_stencil.x[ghost_center][axis] = physical_x[axis] + direction * 2.0 * delta;
          boundary_stencil.x[ghost_outer] = physical_x;
          boundary_stencil.x[ghost_outer][axis] = physical_x[axis] + direction * 4.0 * delta;

          (void)physical_stencil;
        }

        template <typename Model, int dim, typename NumberType, size_t n_components>
        void apply_boundary_reconstruction_stencil(BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil,
                                                   const CellStencilData<dim, NumberType, n_components> &cell_stencil,
                                                   const Model &model)
        {
          prepare_boundary_reconstruction_stencil(boundary_stencil, cell_stencil, boundary_stencil.face_center);
          const bool boundary_supported =
              model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x, boundary_stencil.face_center);
          AssertThrow(boundary_supported,
                      dealii::ExcMessage("KT boundary stencil was rejected by the model boundary policy."));
        }

        template <typename Model, typename NumberType, size_t n_components>
        BoundaryStencilData<2, NumberType, n_components> make_composed_corner_boundary_stencil(
            const BoundaryReconstructionStencilData<2, NumberType, n_components> &boundary_reconstruction_stencil,
            const unsigned int result_face, const CellStencilData<2, NumberType, n_components> &physical_stencil,
            const Model &model)
        {
          using namespace def::BoundaryStencilIndex;
          std::array<BoundaryStencilData<2, NumberType, n_components>, 3> tangential_stencils =
              boundary_reconstruction_stencil.corner_tangential_stencils[result_face];
          for (auto &tangential_stencil : tangential_stencils)
            apply_boundary_reconstruction_stencil(tangential_stencil, physical_stencil, model);

          BoundaryStencilData<2, NumberType, n_components> normal_stencil{};
          const auto &primary = boundary_reconstruction_stencil.primary;
          normal_stencil.lower_boundary = primary.lower_boundary;
          normal_stencil.cell_face = primary.cell_face;
          normal_stencil.face_center = primary.face_center;
          normal_stencil.ghost_center = primary.ghost_center;
          normal_stencil.ghost_left = primary.ghost_left;
          normal_stencil.ghost_right = primary.ghost_right;
          for (auto &dofs : normal_stencil.dof_indices)
            dofs.fill(dealii::numbers::invalid_dof_index);

          const unsigned int normal_axis = primary.cell_face / 2;
          const unsigned int tangential_axis = 1U - normal_axis;
          normal_stencil.x[physical_cell] = tangential_stencils[0].x[tangential_stencils[0].ghost_center];
          normal_stencil.u[physical_cell] = tangential_stencils[0].u[tangential_stencils[0].ghost_center];
          normal_stencil.dof_indices[physical_cell] =
              tangential_stencils[0].dof_indices[tangential_stencils[0].ghost_center];
          normal_stencil.face_center[tangential_axis] = normal_stencil.x[physical_cell][tangential_axis];

          const size_t inner_index = primary.lower_boundary ? upper_inner : lower_inner;
          const size_t outer_index = primary.lower_boundary ? upper_outer : lower_outer;
          normal_stencil.x[inner_index] = tangential_stencils[1].x[tangential_stencils[1].ghost_center];
          normal_stencil.u[inner_index] = tangential_stencils[1].u[tangential_stencils[1].ghost_center];
          normal_stencil.dof_indices[inner_index] =
              tangential_stencils[1].dof_indices[tangential_stencils[1].ghost_center];
          normal_stencil.x[outer_index] = tangential_stencils[2].x[tangential_stencils[2].ghost_center];
          normal_stencil.u[outer_index] = tangential_stencils[2].u[tangential_stencils[2].ghost_center];
          normal_stencil.dof_indices[outer_index] =
              tangential_stencils[2].dof_indices[tangential_stencils[2].ghost_center];

          apply_boundary_reconstruction_stencil(normal_stencil, physical_stencil, model);
          return normal_stencil;
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_physical_boundary_side_stencil(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil,
                                            const CellStencilData<dim, NumberType, n_components> & /*physical_stencil*/)
          requires(dim == 1)
        {
          return make_physical_boundary_side_stencil<dim, NumberType, n_components>(boundary_stencil);
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_physical_boundary_side_stencil(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil,
                                            const CellStencilData<dim, NumberType, n_components> &physical_stencil)
          requires(dim == 2)
        {
          CellStencilData<dim, NumberType, n_components> result = physical_stencil;
          const unsigned int normal_face = boundary_stencil.cell_face;
          result.neighbor_x[normal_face] = boundary_stencil.x[boundary_stencil.ghost_center];
          result.neighbor_u[normal_face] = boundary_stencil.u[boundary_stencil.ghost_center];
          result.neighbor_dof_indices[normal_face] = boundary_stencil.dof_indices[boundary_stencil.ghost_center];
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_ghost_boundary_side_stencil(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil,
                                         const CellStencilData<dim, NumberType, n_components> & /*physical_stencil*/)
          requires(dim == 1)
        {
          return make_ghost_boundary_side_stencil<dim, NumberType, n_components>(boundary_stencil);
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, NumberType, n_components>
        make_ghost_boundary_side_stencil(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil,
                                         const CellStencilData<dim, NumberType, n_components> &physical_stencil)
          requires(dim == 2)
        {
          (void)boundary_stencil;
          (void)physical_stencil;
          AssertThrow(false, dealii::ExcMessage(
                                 "KT 2D ghost boundary side stencils require model-owned tangential ghost support."));
          return {};
        }

        template <typename Model, typename NumberType, size_t n_components>
        CellStencilData<2, NumberType, n_components> make_ghost_boundary_side_stencil(
            const BoundaryReconstructionStencilData<2, NumberType, n_components> &boundary_reconstruction_stencil,
            const CellStencilData<2, NumberType, n_components> &physical_stencil, const Model &model)
        {
          using namespace def::BoundaryStencilIndex;
          const auto &boundary_stencil = boundary_reconstruction_stencil.primary;
          CellStencilData<2, NumberType, n_components> result{};
          result.boundary_ids.fill(dealii::numbers::invalid_boundary_id);
          result.face_centers = {};

          result.cell.x = boundary_stencil.x[boundary_stencil.ghost_center];
          result.cell.u = boundary_stencil.u[boundary_stencil.ghost_center];
          result.cell.dof_indices = boundary_stencil.dof_indices[boundary_stencil.ghost_center];

          const unsigned int axis = boundary_stencil.cell_face / 2;
          const unsigned int tangential_axis = 1U - axis;
          const unsigned int normal_minus = 2 * axis;
          const unsigned int normal_plus = normal_minus + 1;
          const unsigned int tangential_minus = 2 * tangential_axis;
          const unsigned int tangential_plus = tangential_minus + 1;

          const size_t normal_outer =
              boundary_stencil.lower_boundary ? boundary_stencil.ghost_left : boundary_stencil.ghost_right;
          if (boundary_stencil.lower_boundary) {
            result.neighbor_x[normal_minus] = boundary_stencil.x[normal_outer];
            result.neighbor_u[normal_minus] = boundary_stencil.u[normal_outer];
            result.neighbor_dof_indices[normal_minus] = boundary_stencil.dof_indices[normal_outer];
            result.neighbor_x[normal_plus] = boundary_stencil.x[physical_cell];
            result.neighbor_u[normal_plus] = boundary_stencil.u[physical_cell];
            result.neighbor_dof_indices[normal_plus] = boundary_stencil.dof_indices[physical_cell];
          } else {
            result.neighbor_x[normal_minus] = boundary_stencil.x[physical_cell];
            result.neighbor_u[normal_minus] = boundary_stencil.u[physical_cell];
            result.neighbor_dof_indices[normal_minus] = boundary_stencil.dof_indices[physical_cell];
            result.neighbor_x[normal_plus] = boundary_stencil.x[normal_outer];
            result.neighbor_u[normal_plus] = boundary_stencil.u[normal_outer];
            result.neighbor_dof_indices[normal_plus] = boundary_stencil.dof_indices[normal_outer];
          }

          const auto set_model_tangential_neighbor = [&](const unsigned int result_face) {
            BoundaryStencilData<2, NumberType, n_components> neighbor_stencil{};
            if (boundary_reconstruction_stencil.tangential_ghost_neighbor_valid[result_face]) {
              neighbor_stencil = boundary_reconstruction_stencil.tangential_ghost_neighbors[result_face];
              apply_boundary_reconstruction_stencil(neighbor_stencil, physical_stencil, model);
            } else {
              AssertThrow(boundary_reconstruction_stencil.corner_tangential_stencil_valid[result_face],
                          dealii::ExcMessage("KT 2D boundary reconstruction is missing tangential ghost support."));
              neighbor_stencil = make_composed_corner_boundary_stencil(boundary_reconstruction_stencil, result_face,
                                                                       physical_stencil, model);
            }
            result.neighbor_x[result_face] = neighbor_stencil.x[neighbor_stencil.ghost_center];
            result.neighbor_u[result_face] = neighbor_stencil.u[neighbor_stencil.ghost_center];
            result.neighbor_dof_indices[result_face] = neighbor_stencil.dof_indices[neighbor_stencil.ghost_center];
          };

          set_model_tangential_neighbor(tangential_minus);
          set_model_tangential_neighbor(tangential_plus);

          return result;
        }

        template <typename Model, typename NumberType, size_t n_components>
        void populate_boundary_neighbor_from_model_stencil(
            BoundaryStencilData<1, NumberType, n_components> &boundary_stencil,
            CellStencilData<1, NumberType, n_components> &cell_stencil, const unsigned int face_index,
            const dealii::Point<1> &x_q, const Model &model)
        {
          const bool boundary_supported = model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x, x_q);
          AssertThrow(boundary_supported,
                      dealii::ExcMessage(
                          "KT boundary stencil was rejected while populating a boundary-adjacent cell stencil."));

          cell_stencil.neighbor_x[face_index] = boundary_stencil.x[boundary_stencil.ghost_center];
          cell_stencil.neighbor_u[face_index] = boundary_stencil.u[boundary_stencil.ghost_center];
          cell_stencil.neighbor_dof_indices[face_index] = boundary_stencil.dof_indices[boundary_stencil.ghost_center];
        }

        template <typename Model, typename NumberType, size_t n_components>
        void populate_boundary_neighbor_from_model_stencil(
            BoundaryStencilData<2, NumberType, n_components> &boundary_stencil,
            CellStencilData<2, NumberType, n_components> &cell_stencil, const unsigned int face_index,
            const dealii::Point<2> &x_q, const Model &model)
        {
          boundary_stencil.face_center = x_q;
          prepare_boundary_reconstruction_stencil(boundary_stencil, cell_stencil, x_q);

          const bool boundary_supported = model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x, x_q);
          AssertThrow(boundary_supported,
                      dealii::ExcMessage(
                          "KT boundary stencil was rejected while populating a boundary-adjacent cell stencil."));

          cell_stencil.neighbor_x[face_index] = boundary_stencil.x[boundary_stencil.ghost_center];
          cell_stencil.neighbor_u[face_index] = boundary_stencil.u[boundary_stencil.ghost_center];
          cell_stencil.neighbor_dof_indices[face_index] = boundary_stencil.dof_indices[boundary_stencil.ghost_center];
        }

        template <typename Model, typename NumberType, size_t n_components>
        std::pair<CellStencilData<1, NumberType, n_components>, CellStencilData<1, NumberType, n_components>>
        make_model_boundary_reconstruction_side_stencils(
            BoundaryStencilData<1, NumberType, n_components> boundary_stencil,
            const CellStencilData<1, NumberType, n_components> &physical_cell_stencil, const dealii::Point<1> &x_q,
            const Model &model)
        {
          prepare_boundary_reconstruction_stencil(boundary_stencil, physical_cell_stencil, x_q);

          const bool boundary_supported = model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x, x_q);
          AssertThrow(boundary_supported,
                      dealii::ExcMessage("KT boundary stencil was rejected by the model boundary policy."));

          return {make_physical_boundary_side_stencil(boundary_stencil, physical_cell_stencil),
                  make_ghost_boundary_side_stencil(boundary_stencil, physical_cell_stencil)};
        }

        template <typename Model, typename NumberType, size_t n_components>
        std::pair<CellStencilData<1, NumberType, n_components>, CellStencilData<1, NumberType, n_components>>
        make_model_boundary_reconstruction_side_stencils(
            BoundaryReconstructionStencilData<1, NumberType, n_components> boundary_reconstruction_stencil,
            const CellStencilData<1, NumberType, n_components> &physical_cell_stencil, const dealii::Point<1> &x_q,
            const Model &model)
        {
          return make_model_boundary_reconstruction_side_stencils(boundary_reconstruction_stencil.primary,
                                                                  physical_cell_stencil, x_q, model);
        }

        template <typename Model, typename NumberType, size_t n_components>
        std::pair<CellStencilData<2, NumberType, n_components>, CellStencilData<2, NumberType, n_components>>
        make_model_boundary_reconstruction_side_stencils(
            BoundaryStencilData<2, NumberType, n_components> boundary_stencil,
            const CellStencilData<2, NumberType, n_components> &physical_cell_stencil, const dealii::Point<2> &x_q,
            const Model &model)
        {
          prepare_boundary_reconstruction_stencil(boundary_stencil, physical_cell_stencil, x_q);

          const bool boundary_supported = model.apply_boundary_stencil(boundary_stencil.u, boundary_stencil.x, x_q);
          AssertThrow(boundary_supported,
                      dealii::ExcMessage("KT boundary stencil was rejected by the model boundary policy."));

          return {make_physical_boundary_side_stencil(boundary_stencil, physical_cell_stencil),
                  make_ghost_boundary_side_stencil(boundary_stencil, physical_cell_stencil)};
        }

        template <typename Model, typename NumberType, size_t n_components>
        std::pair<CellStencilData<2, NumberType, n_components>, CellStencilData<2, NumberType, n_components>>
        make_model_boundary_reconstruction_side_stencils(
            BoundaryReconstructionStencilData<2, NumberType, n_components> boundary_reconstruction_stencil,
            const CellStencilData<2, NumberType, n_components> &physical_cell_stencil, const dealii::Point<2> &x_q,
            const Model &model)
        {
          boundary_reconstruction_stencil.primary.face_center = x_q;
          apply_boundary_reconstruction_stencil(boundary_reconstruction_stencil.primary, physical_cell_stencil, model);

          return {make_physical_boundary_side_stencil(boundary_reconstruction_stencil.primary, physical_cell_stencil),
                  make_ghost_boundary_side_stencil(boundary_reconstruction_stencil, physical_cell_stencil, model)};
        }

        template <def::HasReconstructor Reconstructor, bool with_diffusion = true, typename Model, typename NumberType,
                  size_t n_components>
        FaceReconstructionState<1, NumberType, n_components> compute_boundary_face_reconstruction_state(
            BoundaryStencilData<1, NumberType, n_components> boundary_stencil,
            const CellStencilData<1, NumberType, n_components> &physical_cell_stencil, const dealii::Point<1> &x_q,
            const Model &model)
        {
          const auto [physical_stencil, ghost_stencil] =
              make_model_boundary_reconstruction_side_stencils(boundary_stencil, physical_cell_stencil, x_q, model);
          return compute_interior_face_reconstruction_state<Reconstructor, with_diffusion>(physical_stencil,
                                                                                           ghost_stencil, x_q);
        }

        template <def::HasReconstructor Reconstructor, bool with_diffusion = true, typename Model, typename NumberType,
                  size_t n_components>
        FaceReconstructionState<1, NumberType, n_components> compute_boundary_face_reconstruction_state(
            BoundaryReconstructionStencilData<1, NumberType, n_components> boundary_reconstruction_stencil,
            const CellStencilData<1, NumberType, n_components> &physical_cell_stencil, const dealii::Point<1> &x_q,
            const Model &model)
        {
          return compute_boundary_face_reconstruction_state<Reconstructor, with_diffusion>(
              boundary_reconstruction_stencil.primary, physical_cell_stencil, x_q, model);
        }

        template <def::HasReconstructor Reconstructor, bool with_diffusion = true, typename Model, typename NumberType,
                  size_t n_components>
        FaceReconstructionState<2, NumberType, n_components> compute_boundary_face_reconstruction_state(
            BoundaryStencilData<2, NumberType, n_components> boundary_stencil,
            const CellStencilData<2, NumberType, n_components> &physical_cell_stencil, const dealii::Point<2> &x_q,
            const Model &model)
        {
          const auto [physical_stencil, ghost_stencil] =
              make_model_boundary_reconstruction_side_stencils(boundary_stencil, physical_cell_stencil, x_q, model);
          return compute_interior_face_reconstruction_state<Reconstructor, with_diffusion>(physical_stencil,
                                                                                           ghost_stencil, x_q);
        }

        template <def::HasReconstructor Reconstructor, bool with_diffusion = true, typename Model, typename NumberType,
                  size_t n_components>
        FaceReconstructionState<2, NumberType, n_components> compute_boundary_face_reconstruction_state(
            BoundaryReconstructionStencilData<2, NumberType, n_components> boundary_reconstruction_stencil,
            const CellStencilData<2, NumberType, n_components> &physical_cell_stencil, const dealii::Point<2> &x_q,
            const Model &model)
        {
          const auto [physical_stencil, ghost_stencil] = make_model_boundary_reconstruction_side_stencils(
              boundary_reconstruction_stencil, physical_cell_stencil, x_q, model);
          return compute_interior_face_reconstruction_state<Reconstructor, with_diffusion>(physical_stencil,
                                                                                           ghost_stencil, x_q);
        }

        template <int dim, typename NumberType, size_t n_components>
        CellData<dim, autodiff::Real<1, NumberType>, n_components>
        tag_cell_dofs(const CellData<dim, NumberType, n_components> &cell_data, dealii::types::global_dof_index dof_j)
        {
          using AD = autodiff::Real<1, NumberType>;
          CellData<dim, AD, n_components> result;
          result.x = cell_data.x;
          result.dof_indices = cell_data.dof_indices;
          for (size_t c = 0; c < n_components; ++c) {
            result.u[c] = AD(cell_data.u[c]);
            if (cell_data.dof_indices[c] == dof_j) seed(result.u[c]);
          }
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        CellStencilData<dim, autodiff::Real<1, NumberType>, n_components>
        tag_cell_stencil_dofs(const CellStencilData<dim, NumberType, n_components> &cell_stencil,
                              dealii::types::global_dof_index dof_j)
        {
          CellStencilData<dim, autodiff::Real<1, NumberType>, n_components> result;
          result.cell = tag_cell_dofs(cell_stencil.cell, dof_j);
          result.neighbor_x = cell_stencil.neighbor_x;
          result.neighbor_dof_indices = cell_stencil.neighbor_dof_indices;
          for (size_t f = 0; f < 2 * dim; ++f)
            for (size_t c = 0; c < n_components; ++c) {
              result.neighbor_u[f][c] = cell_stencil.neighbor_u[f][c];
              if (cell_stencil.neighbor_dof_indices[f][c] == dof_j) seed(result.neighbor_u[f][c]);
            }
          result.boundary_ids = cell_stencil.boundary_ids;
          result.face_centers = cell_stencil.face_centers;
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        BoundaryStencilData<dim, autodiff::Real<1, NumberType>, n_components>
        tag_boundary_stencil_dofs(const BoundaryStencilData<dim, NumberType, n_components> &boundary_stencil,
                                  dealii::types::global_dof_index dof_j)
        {
          using AD = autodiff::Real<1, NumberType>;
          BoundaryStencilData<dim, AD, n_components> result;
          result.x = boundary_stencil.x;
          result.dof_indices = boundary_stencil.dof_indices;
          result.lower_boundary = boundary_stencil.lower_boundary;
          result.cell_face = boundary_stencil.cell_face;
          result.ghost_center = boundary_stencil.ghost_center;
          result.ghost_left = boundary_stencil.ghost_left;
          result.ghost_right = boundary_stencil.ghost_right;
          result.face_center = boundary_stencil.face_center;
          for (size_t stencil_index = 0; stencil_index < boundary_stencil.u.size(); ++stencil_index) {
            for (size_t c = 0; c < n_components; ++c) {
              result.u[stencil_index][c] = AD(boundary_stencil.u[stencil_index][c]);
              if (boundary_stencil.dof_indices[stencil_index][c] == dof_j) seed(result.u[stencil_index][c]);
            }
          }
          return result;
        }

        template <int dim, typename NumberType, size_t n_components>
        BoundaryReconstructionStencilData<dim, autodiff::Real<1, NumberType>, n_components>
        tag_boundary_reconstruction_stencil_dofs(
            const BoundaryReconstructionStencilData<dim, NumberType, n_components> &boundary_reconstruction_stencil,
            dealii::types::global_dof_index dof_j)
        {
          BoundaryReconstructionStencilData<dim, autodiff::Real<1, NumberType>, n_components> result{};
          result.primary =
              tag_boundary_stencil_dofs<dim, NumberType, n_components>(boundary_reconstruction_stencil.primary, dof_j);
          result.tangential_ghost_neighbor_valid = boundary_reconstruction_stencil.tangential_ghost_neighbor_valid;
          result.corner_tangential_stencil_valid = boundary_reconstruction_stencil.corner_tangential_stencil_valid;
          for (size_t face = 0; face < result.n_faces; ++face) {
            if (result.tangential_ghost_neighbor_valid[face])
              result.tangential_ghost_neighbors[face] = tag_boundary_stencil_dofs<dim, NumberType, n_components>(
                  boundary_reconstruction_stencil.tangential_ghost_neighbors[face], dof_j);
            if (result.corner_tangential_stencil_valid[face]) {
              for (size_t stencil_index = 0; stencil_index < result.corner_tangential_stencils[face].size();
                   ++stencil_index) {
                result.corner_tangential_stencils[face][stencil_index] =
                    tag_boundary_stencil_dofs<dim, NumberType, n_components>(
                        boundary_reconstruction_stencil.corner_tangential_stencils[face][stencil_index], dof_j);
              }
            }
          }
          return result;
        }

      } // namespace internal
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
