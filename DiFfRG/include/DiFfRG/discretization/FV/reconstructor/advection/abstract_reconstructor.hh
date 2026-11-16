#pragma once

#include <DiFfRG/discretization/FV/reconstructor/types.hh>
#include <DiFfRG/discretization/FV/stencil.hh>

// deal.II
#include <deal.II/base/point.h>

// autodiff
#include <autodiff/forward/real/real.hpp>

#include <array>
#include <concepts>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Concept of a slope reconstruction, the Reconstructor_ / JacobianReconstructor_ of
     * FV::KurganovTadmor::Assembler. Implementations: TVDReconstructor, FirstOrderReconstructor.
     *
     * All functions are static. A stencil provides center(), values(), neighbor_center(f) and neighbor_values(f),
     * with neighbors ordered lower/upper per direction. It can borrow FV storage or own mutable AD/boundary data.
     * - `compute_gradient<n>(stencil)`: the slope \f$g_K\f$ of the reconstruction
     *   \f$u_K(x) = \bar u_K + g_K \cdot (x - x_K)\f$, which gives the face values;
     * - `compute_gradient_at_point<n>(stencil, x)`: the gradient the model sees at a point \f$x\f$ of the
     *   cell (face centre or quadrature point);
     * - `compute_gradient_derivative`, `compute_gradient_at_point_derivative`: their derivative by the one stencil
     *   value seeded in the autodiff inputs;
     * - `compute_third_derivatives_at_face<n>(x, u)` and its `_derivative`: third derivatives at a face from four
     *   cell averages (used in 1D only).
     *
     * Constants: `dim`, and `jacobian_stencil_radius`, how far a face's reconstruction reaches: 1 = the two cells of
     * the face, 2 = also their face neighbours.
     */
    template <typename T, size_t n>
    concept ReconstructorForComponents =
        requires(const FV::CellStencilData<T::dim, double, n> &stencil,
                 const FV::CellStencilData<T::dim, autodiff::Real<1, double>, n> &seeded,
                 const dealii::Point<T::dim> &x, const std::array<dealii::Point<T::dim>, 4> &x_four,
                 const std::array<std::array<double, n>, 4> &u_four,
                 const std::array<std::array<autodiff::Real<1, double>, n>, 4> &seeded_four) {
          { T::template compute_gradient<n>(stencil) } -> std::same_as<GradientType<T::dim, double, n>>;
          { T::template compute_gradient_at_point<n>(stencil, x) } -> std::same_as<GradientType<T::dim, double, n>>;
          { T::template compute_gradient_derivative<n>(seeded) } -> std::same_as<GradientType<T::dim, double, n>>;
          {
            T::template compute_gradient_at_point_derivative<n>(seeded, x)
          } -> std::same_as<GradientType<T::dim, double, n>>;
          {
            T::template compute_third_derivatives_at_face<n>(x_four, u_four)
          } -> std::same_as<ThirdDerivativeType<T::dim, double, n>>;
          {
            T::template compute_third_derivatives_at_face_derivative<n>(x_four, seeded_four)
          } -> std::same_as<ThirdDerivativeType<T::dim, double, n>>;
        };

    template <typename T>
    concept HasReconstructor = requires {
      { T::dim } -> std::convertible_to<int>;
      { T::jacobian_stencil_radius } -> std::convertible_to<int>;
    } && ReconstructorForComponents<T, 1> && ReconstructorForComponents<T, 2>;

  } // namespace def
} // namespace DiFfRG
