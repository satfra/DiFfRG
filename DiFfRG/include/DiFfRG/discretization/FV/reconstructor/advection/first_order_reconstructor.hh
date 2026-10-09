#pragma once

// DiFfRG
#include <DiFfRG/discretization/FV/reconstructor/advection/abstract_reconstructor.hh>

namespace DiFfRG
{
  namespace def
  {
    template <int dim_, typename NumberType> class FirstOrderReconstructor
    {
      using ADNumberType = autodiff::Real<1, NumberType>;

    public:
      static constexpr int dim = dim_;
      static constexpr int n_faces = 2 * dim;
      static constexpr int jacobian_stencil_radius = 1;

      template <int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient([[maybe_unused]] const dealii::Point<dim> &center_pos,
                       [[maybe_unused]] const std::array<NumberType, n_components> &u_center,
                       [[maybe_unused]] const std::array<dealii::Point<dim>, n_faces> &x_n,
                       [[maybe_unused]] const std::array<std::array<NumberType, n_components>, n_faces> &u_n)
      {
        return {};
      }

      template <int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient_at_point([[maybe_unused]] const dealii::Point<dim> &center_pos,
                                [[maybe_unused]] const dealii::Point<dim> &x,
                                [[maybe_unused]] const std::array<NumberType, n_components> &u_center,
                                [[maybe_unused]] const std::array<dealii::Point<dim>, n_faces> &x_n,
                                [[maybe_unused]] const std::array<std::array<NumberType, n_components>, n_faces> &u_n)
      {
        return {};
      }

      template <int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient_derivative([[maybe_unused]] const dealii::Point<dim> &center_pos,
                                  [[maybe_unused]] const std::array<ADNumberType, n_components> &u_center,
                                  [[maybe_unused]] const std::array<dealii::Point<dim>, n_faces> &x_n,
                                  [[maybe_unused]] const std::array<std::array<ADNumberType, n_components>, n_faces> &u_n)
      {
        return {};
      }

      template <int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient_at_point_derivative(
          [[maybe_unused]] const dealii::Point<dim> &center_pos, [[maybe_unused]] const dealii::Point<dim> &x,
          [[maybe_unused]] const std::array<ADNumberType, n_components> &u_center,
          [[maybe_unused]] const std::array<dealii::Point<dim>, n_faces> &x_n,
          [[maybe_unused]] const std::array<std::array<ADNumberType, n_components>, n_faces> &u_n)
      {
        return {};
      }

      template <int n_components>
      static ThirdDerivativeType<dim, NumberType, n_components> compute_third_derivatives_at_face(
          [[maybe_unused]] const std::array<dealii::Point<dim>, 4> &x_stencil,
          [[maybe_unused]] const std::array<std::array<NumberType, n_components>, 4> &u_stencil)
      {
        return {};
      }

      template <int n_components>
      static ThirdDerivativeType<dim, NumberType, n_components> compute_third_derivatives_at_face_derivative(
          [[maybe_unused]] const std::array<dealii::Point<dim>, 4> &x_stencil,
          [[maybe_unused]] const std::array<std::array<ADNumberType, n_components>, 4> &u_stencil)
      {
        return {};
      }
    };

    static_assert(HasReconstructor<FirstOrderReconstructor<1, double>>);
    static_assert(HasReconstructor<FirstOrderReconstructor<2, double>>);

  } // namespace def
} // namespace DiFfRG
