#pragma once

#include <DiFfRG/discretization/FV/reconstructor/types.hh>

#include <array>
#include <autodiff/forward/real.hpp>
#include <cstddef>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

namespace DiFfRG
{
  namespace def
  {
    template <int dim, typename NumberType, size_t n_components> struct DiffusionFaceState {
      std::array<NumberType, n_components> u_minus{};
      std::array<NumberType, n_components> u_plus{};
      GradientType<dim, NumberType, n_components> grad_minus{};
      GradientType<dim, NumberType, n_components> grad_plus{};
      std::array<NumberType, n_components> u{};
      GradientType<dim, NumberType, n_components> grad{};
    };

    template <int dim_, typename NumberType> class CorrectedWeightedLeastSquaresDiffusionReconstructor
    {
    public:
      static constexpr int dim = dim_;
      static constexpr int n_faces = 2 * dim;
      static constexpr int jacobian_stencil_radius = 2;

      template <size_t n_components, typename CellStencil>
      static GradientType<dim, NumberType, n_components> compute_cell_gradient(const CellStencil &stencil)
      {
        dealii::Tensor<2, dim> normal_matrix{};
        std::array<dealii::Tensor<1, dim, NumberType>, n_components> rhs{};

        for (size_t face = 0; face < stencil.neighbors.x.size(); ++face) {
          const auto dx = stencil.neighbors.x[face] - stencil.cell.x;
          double distance2 = 0.0;
          for (size_t d = 0; d < dim; ++d)
            distance2 += dx[d] * dx[d];
          if (distance2 <= 1.0e-30) continue;

          const double weight = 1.0 / distance2;
          for (size_t d = 0; d < dim; ++d)
            for (size_t e = 0; e < dim; ++e)
              normal_matrix[d][e] += weight * dx[d] * dx[e];

          for (size_t c = 0; c < n_components; ++c) {
            const NumberType du = stencil.neighbors.u[face][c] - stencil.cell.u[c];
            for (size_t d = 0; d < dim; ++d)
              rhs[c][d] += NumberType(weight * dx[d]) * du;
          }
        }

        const auto inverse = dealii::invert(normal_matrix);
        GradientType<dim, NumberType, n_components> gradients{};

        for (size_t c = 0; c < n_components; ++c)
          for (size_t d = 0; d < dim; ++d)
            for (size_t e = 0; e < dim; ++e)
              gradients[c][d] += NumberType(inverse[d][e]) * rhs[c][e];
        return gradients;
      }

      template <size_t n_components>
      static GradientType<dim, NumberType, n_components> correct_gradient(
          const GradientType<dim, NumberType, n_components> &base_gradient,
          const std::array<NumberType, n_components> &u_minus, const std::array<NumberType, n_components> &u_plus,
          const dealii::Tensor<1, dim> &d)
      {
        GradientType<dim, NumberType, n_components> corrected = base_gradient;
        const double distance2 = dealii::scalar_product(d, d);
        if (distance2 <= 1.0e-30) return corrected;

        for (size_t c = 0; c < n_components; ++c) {
          const NumberType directional_residual =
              u_plus[c] - u_minus[c] - dealii::scalar_product(base_gradient[c], d);
          for (size_t axis = 0; axis < dim; ++axis)
            corrected[c][axis] += directional_residual * NumberType(d[axis] / distance2);
        }
        return corrected;
      }

      template <size_t n_components, typename CellStencil>
      static DiffusionFaceState<dim, NumberType, n_components> compute_face_state(
          const CellStencil &minus_stencil, const CellStencil &plus_stencil)
      {
        DiffusionFaceState<dim, NumberType, n_components> state{};
        const auto grad_minus = compute_cell_gradient<n_components>(minus_stencil);
        const auto grad_plus = compute_cell_gradient<n_components>(plus_stencil);
        const auto d = plus_stencil.cell.x - minus_stencil.cell.x;

        for (size_t c = 0; c < n_components; ++c) {
          state.u_minus[c] = minus_stencil.cell.u[c];
          state.u_plus[c] = plus_stencil.cell.u[c];
        }
        state.grad_minus = correct_gradient<n_components>(grad_minus, state.u_minus, state.u_plus, d);
        state.grad_plus = correct_gradient<n_components>(grad_plus, state.u_minus, state.u_plus, d);

        for (size_t c = 0; c < n_components; ++c) {
          state.u[c] = NumberType(0.5) * (state.u_minus[c] + state.u_plus[c]);
          for (size_t axis = 0; axis < dim; ++axis)
            state.grad[c][axis] = NumberType(0.5) * (state.grad_minus[c][axis] + state.grad_plus[c][axis]);
        }
        return state;
      }

      template <size_t n_components>
      static std::array<ReconstructionDerivativeData<dim, NumberType, n_components>, 2> extract_derivatives(
          const DiffusionFaceState<dim, autodiff::Real<1, NumberType>, n_components> &state)
      {
        std::array<ReconstructionDerivativeData<dim, NumberType, n_components>, 2> result{};
        for (size_t c = 0; c < n_components; ++c) {
          result[0].u[c] = autodiff::derivative(state.u_minus[c]);
          result[1].u[c] = autodiff::derivative(state.u_plus[c]);
          for (size_t d = 0; d < dim; ++d) {
            result[0].grad[c][d] = autodiff::derivative(state.grad_minus[c][d]);
            result[1].grad[c][d] = autodiff::derivative(state.grad_plus[c][d]);
          }
        }
        return result;
      }
    };

  } // namespace def
} // namespace DiFfRG
