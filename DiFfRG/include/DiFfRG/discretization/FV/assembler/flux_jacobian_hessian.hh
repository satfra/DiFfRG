#pragma once

// DiFfRG
#include <DiFfRG/common/tuples.hh>

// external libraries
#include <deal.II/base/tensor.h>

// standard library
#include <array>
#include <cstddef>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      namespace internal
      {
        template <typename NumberType, size_t n_components>
        using JacobianMatrix = std::array<std::array<NumberType, n_components>, n_components>;

        template <typename NumberType, int dim, size_t n_components>
        using HessianTensor =
            std::array<std::array<std::array<std::array<NumberType, n_components>, n_components>, n_components>, dim>;

        /**
         * @brief Derivative of every flux component/direction with respect to every component/direction of grad(u).
         *
         * Indexed as [flux_component][gradient_component][flux_direction][gradient_direction].
         */
        template <typename NumberType, int dim, size_t n_components>
        using FluxGradientJacobian =
            std::array<std::array<dealii::Tensor<2, dim, NumberType>, n_components>, n_components>;

        template <typename NumberType, int dim, size_t n_components>
        using MixedHessianTensor = std::array<HessianTensor<NumberType, dim, n_components>, dim>;

        /// The flux at a trace and its derivatives with respect to the trace's values and gradients, as the numerical
        /// flux jacobian needs them (see flux_derivatives).
        template <typename NumberType, int dim, size_t n_components> struct FluxDerivativeData {
          std::array<dealii::Tensor<1, dim, NumberType>, n_components> F{};
          std::array<JacobianMatrix<NumberType, n_components>, dim> J{};
          HessianTensor<NumberType, dim, n_components> H{};
          FluxGradientJacobian<NumberType, dim, n_components> grad_J{};
          MixedHessianTensor<NumberType, dim, n_components> mixed_H{};
        };

        /// Half the derivative of the diffusion flux on one trace with respect to its inputs: the face's diffusion
        /// flux is the average of the two traces' fluxes.
        template <int dim, typename NumberType, size_t n_components> struct DiffusionSideJacobian {
          SimpleMatrix<dealii::Tensor<1, dim, NumberType>, n_components> u{};
          SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<1, dim, NumberType>>, n_components> grad{};
          SimpleMatrix<dealii::Tensor<1, dim, dealii::Tensor<3, dim, NumberType>>, n_components> third_derivatives{};
        };

      } // namespace internal
    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
