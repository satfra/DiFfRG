#pragma once

#include <array>
#include <cstddef>
#include <deal.II/base/tensor.h>

namespace DiFfRG
{
  namespace def
  {
    /// Faces of a rectangular cell in dim dimensions.
    template <int dim> inline constexpr std::size_t n_faces = 2 * dim;

    /// One gradient per component.
    template <int dim, typename NumberType, size_t n_components>
    using GradientType = std::array<dealii::Tensor<1, dim, NumberType>, n_components>;

    /// One tensor of third derivatives per component.
    template <int dim, typename NumberType, size_t n_components>
    using ThirdDerivativeType = std::array<dealii::Tensor<3, dim, NumberType>, n_components>;

    /// The derivatives of a reconstructed value, gradient and third derivatives by one stencil value.
    template <int dim, typename NumberType, size_t n_components> struct ReconstructionDerivativeData {
      std::array<NumberType, n_components> u{};
      GradientType<dim, NumberType, n_components> grad{};
      ThirdDerivativeType<dim, NumberType, n_components> third_derivatives{};
    };

  } // namespace def
} // namespace DiFfRG
