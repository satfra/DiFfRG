#pragma once

// standard library
#include <algorithm>
#include <cmath>

// DiFfRG
#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>
#include <DiFfRG/discretization/FV/limiter/minmod_limiter.hh>
#include <DiFfRG/discretization/FV/reconstructor/advection/abstract_reconstructor.hh>

// deal.II
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

// autodiff
#include <autodiff/forward/real/real.hpp>

// standard library
#include <array>
#include <cstddef>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Slope-limited linear reconstruction on a rectangular mesh.
     *
     * Per direction \f$d\f$, the backward and forward slopes of cell \f$K\f$,
     * \f$a = (\bar u_K - \bar u_{K^-})/(x_K - x_{K^-})_d\f$ and \f$b = (\bar u_{K^+} - \bar u_K)/(x_{K^+} - x_K)_d\f$
     * with \f$K^-, K^+\f$ the lower and upper neighbours, give the cell slope
     * \f$(g_K)_d\f$ = Limiter::slope_limit(a, b).
     * The reconstruction is total variation diminishing (TVD) for a TVD limiter (all but CentralLimiter).
     *
     * The gradient handed to the model at a point \f$x\f$ (compute_gradient_at_point) is, per direction, \f$a\f$ if
     * \f$x\f$ lies below the centre, \f$b\f$ if above, and the limited slope if level with it (up to rounding). At a
     * face this is the unlimited one-sided slope across the face in the normal direction, and the limited slope in
     * the others.
     *
     * Third derivatives (1D) are the divided difference of four cell averages.
     *
     * @tparam dim_ The spatial dimension.
     * @tparam Limiter The slope limiter (def::HasSlopeLimiter), e.g. MinModLimiter.
     * @tparam NumberType The scalar type of the cell averages.
     *
     * Usage with the KT assembler:
     * @code
     * using Assembler = FV::KurganovTadmor::Assembler<
     *     Discretization, Model, def::TVDReconstructor<Discretization::dim, def::MinModLimiter, double>>;
     * @endcode
     */
    template <int dim_, HasSlopeLimiter Limiter, typename NumberType> class TVDReconstructor
    {
      using ADNumberType = autodiff::Real<1, NumberType>;

    public:
      static constexpr int dim = dim_;
      static constexpr int n_faces = 2 * dim;
      /// A face depends on its two cells and their face neighbours.
      static constexpr int jacobian_stencil_radius = 2;

    private:
      template <int n_components>
      static GradientType<dim, NumberType, n_components>
      derivative_of(const GradientType<dim, ADNumberType, n_components> &gradient)
      {
        GradientType<dim, NumberType, n_components> result{};
        for (size_t c = 0; c < n_components; ++c)
          for (int d = 0; d < dim; ++d)
            result[c][d] = autodiff::derivative(gradient[c][d]);
        return result;
      }

    public:
      /// The slope limiter.
      using LimiterType = Limiter;
      /**
       * @brief The limited cell slope \f$g_K\f$, per component and direction.
       *
       * @param center_pos The cell centre.
       * @param u_center The cell averages.
       * @param x_n The centres of the \f$2d\f$ face neighbours, ordered (lower, upper) for direction 0, then 1.
       * @param u_n Their cell averages (model ghosts across the boundary).
       */
      template <int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient(const dealii::Point<dim> &center_pos, const std::array<NumberType, n_components> &u_center,
                       const std::array<dealii::Point<dim>, n_faces> &x_n,
                       const std::array<std::array<NumberType, n_components>, n_faces> &u_n)
      {
        GradientType<dim, NumberType, n_components> u_grad{};
        for (size_t c = 0; c < static_cast<size_t>(n_components); c++) {
          const NumberType &u_val = u_center[c];
          for (int d = 0, i_n_1 = 0, i_n_2 = 1; d < dim; d++, i_n_1 += 2, i_n_2 += 2) {

            const auto &u_n_1 = u_n[i_n_1][c];
            const auto &u_n_2 = u_n[i_n_2][c];

            const auto dx_1 = x_n[i_n_1] - center_pos;
            const NumberType du_1 = (u_n_1 - u_val) / dx_1[d];
            const auto dx_2 = x_n[i_n_2] - center_pos;
            const NumberType du_2 = (u_n_2 - u_val) / dx_2[d];

            u_grad[c][d] = Limiter::slope_limit(du_1, du_2);
          }
        }

        return u_grad;
      }

      /// The gradient handed to the model at @p x: per direction the one-sided slope towards @p x, or the limited
      /// slope if @p x is level with the centre.
      template <int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient_at_point(const dealii::Point<dim> &center_pos, const dealii::Point<dim> &x,
                                const std::array<NumberType, n_components> &u_center,
                                const std::array<dealii::Point<dim>, n_faces> &x_n,
                                const std::array<std::array<NumberType, n_components>, n_faces> &u_n)
      {
        GradientType<dim, NumberType, n_components> u_grad{};

        for (size_t c = 0; c < static_cast<size_t>(n_components); ++c) {
          const NumberType &u_val = u_center[c];
          for (int d = 0, i_n_1 = 0, i_n_2 = 1; d < dim; ++d, i_n_1 += 2, i_n_2 += 2) {
            const auto dx_1 = x_n[i_n_1] - center_pos;
            const NumberType du_1 = (u_n[i_n_1][c] - u_val) / dx_1[d];
            const auto dx_2 = x_n[i_n_2] - center_pos;
            const NumberType du_2 = (u_n[i_n_2][c] - u_val) / dx_2[d];

            // x at the centre along d, up to rounding: the source points come from FEValues, the centre from
            // the cell, and the two need not agree to the last bit.
            const double tolerance = 1e-10 * std::min(std::abs(dx_1[d]), std::abs(dx_2[d]));
            if (x[d] < center_pos[d] - tolerance)
              u_grad[c][d] = du_1;
            else if (x[d] > center_pos[d] + tolerance)
              u_grad[c][d] = du_2;
            else
              u_grad[c][d] = Limiter::slope_limit(du_1, du_2);
          }
        }

        return u_grad;
      }

      /**
       * @brief The derivative of compute_gradient by the one stencil value whose autodiff input is seeded.
       *
       * @param center_pos The cell centre.
       * @param u_center The cell averages, as autodiff numbers.
       * @param x_n The centres of the \f$2d\f$ face neighbours.
       * @param u_n Their cell averages, as autodiff numbers.
       */
      template <int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient_derivative(const dealii::Point<dim> &center_pos,
                                  const std::array<ADNumberType, n_components> &u_center,
                                  const std::array<dealii::Point<dim>, n_faces> &x_n,
                                  const std::array<std::array<ADNumberType, n_components>, n_faces> &u_n)
      {
        // Forward AD through the limiter, not a finite difference: a difference step straddles the limiter's
        // kink whenever the two one-sided slopes differ by less than the step, and then returns the mean of
        // both branches instead of the derivative of the branch the residual takes.
        return derivative_of<n_components>(TVDReconstructor<dim, Limiter, ADNumberType>::template compute_gradient<n_components>(
            center_pos, u_center, x_n, u_n));
      }

      /// The derivative of compute_gradient_at_point by the one seeded stencil value.
      template <int n_components>
      static GradientType<dim, NumberType, n_components>
      compute_gradient_at_point_derivative(const dealii::Point<dim> &center_pos, const dealii::Point<dim> &x,
                                           const std::array<ADNumberType, n_components> &u_center,
                                           const std::array<dealii::Point<dim>, n_faces> &x_n,
                                           const std::array<std::array<ADNumberType, n_components>, n_faces> &u_n)
      {
        return derivative_of<n_components>(
            TVDReconstructor<dim, Limiter, ADNumberType>::template compute_gradient_at_point<n_components>(
                center_pos, x, u_center, x_n, u_n));
      }

      /// In 1D, \f$u'''\f$ from the divided difference of the four cell averages around a face; zero otherwise.
      template <int n_components>
      static ThirdDerivativeType<dim, NumberType, n_components> compute_third_derivatives_at_face(
          const std::array<dealii::Point<dim>, 4> &x_stencil,
          const std::array<std::array<NumberType, n_components>, 4> &u_stencil)
      {
        ThirdDerivativeType<dim, NumberType, n_components> result{};
        if constexpr (dim == 1) {
          const auto x0 = x_stencil[0][0];
          const auto x1 = x_stencil[1][0];
          const auto x2 = x_stencil[2][0];
          const auto x3 = x_stencil[3][0];
          for (size_t c = 0; c < n_components; ++c) {
            const auto f01 = (u_stencil[1][c] - u_stencil[0][c]) / (x1 - x0);
            const auto f12 = (u_stencil[2][c] - u_stencil[1][c]) / (x2 - x1);
            const auto f23 = (u_stencil[3][c] - u_stencil[2][c]) / (x3 - x2);
            const auto f012 = (f12 - f01) / (x2 - x0);
            const auto f123 = (f23 - f12) / (x3 - x1);
            result[c][0][0][0] = NumberType(6.) * (f123 - f012) / (x3 - x0);
          }
        }
        return result;
      }

      /// The derivative of compute_third_derivatives_at_face by the one seeded stencil value.
      template <int n_components>
      static ThirdDerivativeType<dim, NumberType, n_components> compute_third_derivatives_at_face_derivative(
          const std::array<dealii::Point<dim>, 4> &x_stencil,
          const std::array<std::array<ADNumberType, n_components>, 4> &u_stencil)
      {
        const auto third_derivatives_AD =
            TVDReconstructor<dim, Limiter, ADNumberType>::template compute_third_derivatives_at_face<n_components>(
                x_stencil, u_stencil);

        ThirdDerivativeType<dim, NumberType, n_components> result{};
        for (size_t c = 0; c < n_components; ++c)
          for (int d0 = 0; d0 < dim; ++d0)
            for (int d1 = 0; d1 < dim; ++d1)
              for (int d2 = 0; d2 < dim; ++d2)
                result[c][d0][d1][d2] = autodiff::derivative(third_derivatives_AD[c][d0][d1][d2]);
        return result;
      }
    };

    // Verify the default instantiation satisfies the concept.
    static_assert(HasReconstructor<TVDReconstructor<1, MinModLimiter, double>>);
    static_assert(HasReconstructor<TVDReconstructor<2, MinModLimiter, double>>);

  } // namespace def
} // namespace DiFfRG
