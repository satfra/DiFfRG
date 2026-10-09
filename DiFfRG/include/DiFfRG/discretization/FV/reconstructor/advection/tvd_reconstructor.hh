#pragma once

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
     * @brief TVD gradient reconstructor parameterised by a slope limiter.
     *
     * Computes piecewise-linear cell gradients on a rectangular mesh using
     * two one-sided finite-difference slopes per dimension, each pair
     * limited by the injected @c Limiter.
     *
     * When the limiter satisfies the TVD property (e.g. @c MinModLimiter),
     * the resulting reconstruction is Total Variation Diminishing.
     *
     * @tparam dim          The spatial dimension of the reconstruction.
     * @tparam Limiter      A type satisfying the @c HasSlopeLimiter concept
     *                      (e.g. @c MinModLimiter).
     * @tparam NumberType   The numeric type used for computations
     *                      (e.g. @c double, @c float).
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

      // Expose the Limiter type for use by other templates
      using LimiterType = Limiter;
      /**
       * @brief Compute the gradient of u using the injected slope limiter.
       *
       * For every component and every spatial dimension the method computes
       * two one-sided slopes from the cell centre to its axis-aligned
       * neighbours and returns the limiter-processed gradient.
       *
       * @tparam n_components the number of components in u
       *
       * @param center_pos  position of the cell centre
       * @param u_center    solution values at the cell centre
       * @param x_n         positions of the def::n_faces<dim> neighbouring cell centres,
       *                    ordered as pairs (left, right) for dim 0, then dim 1, …
       * @param u_n         solution values at those neighbours
       *
       * @return per-component gradient vector
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

            if (x[d] < center_pos[d])
              u_grad[c][d] = du_1;
            else if (x[d] > center_pos[d])
              u_grad[c][d] = du_2;
            else
              u_grad[c][d] = Limiter::slope_limit(du_1, du_2);
          }
        }

        return u_grad;
      }

      /**
       * @brief Compute the derivative of the limited gradient w.r.t. a single stencil DOF.
       *
       * The inputs are already seeded @c autodiff::Real<1,NumberType> values;
       * exactly one entry across @p u_center and @p u_n should have a non-zero
       * derivative part (set via @c seed()).
       *
       * Unlike @c compute_gradient, this method does not evaluate the
       * reconstruction at a face point — it returns the per-component
       * derivative of the gradient itself.
       *
       * @tparam n_components the number of solution components
       *
       * @param center_pos  position of the cell centre
       * @param u_center    cell-centre values as seeded AD types
       * @param x_n         positions of the def::n_faces<dim> neighbouring cell centres
       * @param u_n         neighbour values as seeded AD types
       *
       * @return per-component gradient-derivative tensor  d(grad u_c) / d(u_target)
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
