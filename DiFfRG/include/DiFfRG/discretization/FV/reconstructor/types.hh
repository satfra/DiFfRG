#pragma once

#include <array>
#include <cstddef>
#include <deal.II/base/tensor.h>

namespace DiFfRG
{
  namespace def
  {
    template <int dim> inline constexpr std::size_t n_faces = 2 * dim;

    /**
     * @brief Per-component gradient type: one Tensor<1,dim> per solution component.
     */
    template <int dim, typename NumberType, size_t n_components>
    using GradientType = std::array<dealii::Tensor<1, dim, NumberType>, n_components>;

    /**
     * @brief Per-component third spatial derivative type.
     */
    template <int dim, typename NumberType, size_t n_components>
    using ThirdDerivativeType = std::array<dealii::Tensor<3, dim, NumberType>, n_components>;

    template <int dim, typename NumberType, size_t n_components> struct ReconstructionDerivativeData {
      std::array<NumberType, n_components> u{};
      GradientType<dim, NumberType, n_components> grad{};
      ThirdDerivativeType<dim, NumberType, n_components> third_derivatives{};
    };

  } // namespace def
} // namespace DiFfRG
