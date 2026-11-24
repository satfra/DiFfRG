#pragma once
#include <DiFfRG/discretization/FV/reconstructor/types.hh>
#include <array>
#include <type_traits>
namespace DiFfRG::FV
{
  template <int dim, typename NumberType, size_t n_components> struct FaceReconstructionState {
    std::array<NumberType, n_components> u_minus{};
    std::array<NumberType, n_components> u_plus{};
    def::GradientType<dim, NumberType, n_components> center_grad_minus{};
    def::GradientType<dim, NumberType, n_components> center_grad_plus{};
    def::GradientType<dim, NumberType, n_components> face_grad_minus{};
    def::GradientType<dim, NumberType, n_components> face_grad_plus{};
    std::array<NumberType, n_components> diffusion_u_minus{};
    std::array<NumberType, n_components> diffusion_u_plus{};
    def::GradientType<dim, NumberType, n_components> diffusion_grad_minus{};
    def::GradientType<dim, NumberType, n_components> diffusion_grad_plus{};
    std::array<NumberType, n_components> diffusion_u{};
    def::GradientType<dim, NumberType, n_components> diffusion_grad{};
    /// Only 1D reconstructs third derivatives; in more dimensions they would be zeros (half the state in 2D),
    /// so they are not stored.
    struct NoThirdDerivatives {};
    using ThirdDerivatives =
        std::conditional_t<dim == 1, def::ThirdDerivativeType<dim, NumberType, n_components>, NoThirdDerivatives>;
    [[no_unique_address]] ThirdDerivatives third_derivatives_minus{};
    [[no_unique_address]] ThirdDerivatives third_derivatives_plus{};
  };

} // namespace DiFfRG::FV
