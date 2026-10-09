#pragma once

#include <DiFfRG/discretization/FV/reconstructor/types.hh>

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
     * @brief Concept that any gradient-reconstruction strategy must satisfy.
     */
	    template <typename T>
	    concept HasReconstructor =
	        requires {
	          {
	            T::dim
	          } -> std::convertible_to<int>;
	          {
	            T::jacobian_stencil_radius
	          } -> std::convertible_to<int>;
	        } &&
        // ---- compute_gradient (n_components=1) ----
        requires(const dealii::Point<T::dim> &center, const std::array<double, 1> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<double, 1>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient<1>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 1>>;
        } &&
        // ---- compute_gradient (n_components=2) ----
        requires(const dealii::Point<T::dim> &center, const std::array<double, 2> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<double, 2>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient<2>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 2>>;
        } &&
        // ---- compute_gradient_at_point (n_components=1) ----
        requires(const dealii::Point<T::dim> &center, const dealii::Point<T::dim> &x,
                 const std::array<double, 1> &u_center, const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<double, 1>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_at_point<1>(center, x, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 1>>;
        } &&
        // ---- compute_gradient_at_point (n_components=2) ----
        requires(const dealii::Point<T::dim> &center, const dealii::Point<T::dim> &x,
                 const std::array<double, 2> &u_center, const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<double, 2>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_at_point<2>(center, x, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 2>>;
        } &&
        // ---- compute_gradient_derivative (n_components=1) ----
        requires(const dealii::Point<T::dim> &center, const std::array<autodiff::Real<1, double>, 1> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<autodiff::Real<1, double>, 1>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_derivative<1>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 1>>;
        } &&
        // ---- compute_gradient_derivative (n_components=2) ----
        requires(const dealii::Point<T::dim> &center, const std::array<autodiff::Real<1, double>, 2> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<autodiff::Real<1, double>, 2>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_derivative<2>(center, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 2>>;
        } &&
        // ---- compute_gradient_at_point_derivative (n_components=1) ----
        requires(const dealii::Point<T::dim> &center, const dealii::Point<T::dim> &x,
                 const std::array<autodiff::Real<1, double>, 1> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<autodiff::Real<1, double>, 1>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_at_point_derivative<1>(center, x, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 1>>;
        } &&
        // ---- compute_gradient_at_point_derivative (n_components=2) ----
        requires(const dealii::Point<T::dim> &center, const dealii::Point<T::dim> &x,
                 const std::array<autodiff::Real<1, double>, 2> &u_center,
                 const std::array<dealii::Point<T::dim>, n_faces<T::dim>> &x_n,
                 const std::array<std::array<autodiff::Real<1, double>, 2>, n_faces<T::dim>> &u_n) {
          {
            T::template compute_gradient_at_point_derivative<2>(center, x, u_center, x_n, u_n)
          } -> std::same_as<GradientType<T::dim, double, 2>>;
        } &&
        // ---- compute_third_derivatives_at_face (n_components=1) ----
        requires(const std::array<dealii::Point<T::dim>, 4> &x_stencil,
                 const std::array<std::array<double, 1>, 4> &u_stencil) {
          {
            T::template compute_third_derivatives_at_face<1>(x_stencil, u_stencil)
          } -> std::same_as<ThirdDerivativeType<T::dim, double, 1>>;
        } &&
        // ---- compute_third_derivatives_at_face (n_components=2) ----
        requires(const std::array<dealii::Point<T::dim>, 4> &x_stencil,
                 const std::array<std::array<double, 2>, 4> &u_stencil) {
          {
            T::template compute_third_derivatives_at_face<2>(x_stencil, u_stencil)
          } -> std::same_as<ThirdDerivativeType<T::dim, double, 2>>;
        } &&
        // ---- compute_third_derivatives_at_face_derivative (n_components=1) ----
        requires(const std::array<dealii::Point<T::dim>, 4> &x_stencil,
                 const std::array<std::array<autodiff::Real<1, double>, 1>, 4> &u_stencil) {
          {
            T::template compute_third_derivatives_at_face_derivative<1>(x_stencil, u_stencil)
          } -> std::same_as<ThirdDerivativeType<T::dim, double, 1>>;
        } &&
        // ---- compute_third_derivatives_at_face_derivative (n_components=2) ----
        requires(const std::array<dealii::Point<T::dim>, 4> &x_stencil,
                 const std::array<std::array<autodiff::Real<1, double>, 2>, 4> &u_stencil) {
          {
            T::template compute_third_derivatives_at_face_derivative<2>(x_stencil, u_stencil)
          } -> std::same_as<ThirdDerivativeType<T::dim, double, 2>>;
        };

  } // namespace def
} // namespace DiFfRG
