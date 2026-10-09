#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <utility>

#include <DiFfRG/discretization/FV/assembler/flux_jacobian_hessian.hh>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Concept that any wave-speed strategy must satisfy.
     *
     * A wave-speed class must provide:
     * - `compute_speeds`:  given plus/minus Jacobians, return per-dimension wave speeds.
     * - `compute_selected_speed_derivatives`: return the differentiated speed for the
     *   side selected by the maximum spectral radius.
     *
     * New wave-speed strategies only need to satisfy this concept:
     * @code
     * struct MyWaveSpeed
     * {
     *   template <typename NumberType, int dim, size_t n_components>
     *   static std::array<NumberType, dim>
     *   compute_speeds(
     *       const std::array<FV::KurganovTadmor::internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
     *       const std::array<FV::KurganovTadmor::internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus);
     *
     *   template <typename NumberType, int dim, size_t n_components>
     *   static std::pair<std::array<std::array<NumberType, n_components>, dim>,
     *                    std::array<std::array<NumberType, n_components>, dim>>
     *   compute_selected_speed_derivatives(
     *       const std::array<FV::KurganovTadmor::internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
     *       const std::array<FV::KurganovTadmor::internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus,
     *       const FV::KurganovTadmor::internal::HessianTensor<NumberType, dim, n_components> &H_plus,
     *       const FV::KurganovTadmor::internal::HessianTensor<NumberType, dim, n_components> &H_minus);
     * };
     * static_assert(HasWaveSpeed<MyWaveSpeed>);
     * @endcode
     */
    template <typename T>
    concept HasWaveSpeed =
        // ---- compute_speeds (dim=1, n_components=1) ----
        requires(const std::array<FV::KurganovTadmor::internal::JacobianMatrix<double, 1>, 1> &J1_plus,
                 const std::array<FV::KurganovTadmor::internal::JacobianMatrix<double, 1>, 1> &J1_minus) {
          {
            T::template compute_speeds<double, 1, 1>(J1_plus, J1_minus)
          } -> std::same_as<std::array<double, 1>>;
        } &&
        // ---- compute_speeds (dim=2, n_components=2) ----
        requires(const std::array<FV::KurganovTadmor::internal::JacobianMatrix<double, 2>, 2> &J2_plus,
                 const std::array<FV::KurganovTadmor::internal::JacobianMatrix<double, 2>, 2> &J2_minus) {
          {
            T::template compute_speeds<double, 2, 2>(J2_plus, J2_minus)
          } -> std::same_as<std::array<double, 2>>;
        } &&
        // ---- compute_selected_speed_derivatives (dim=1, n_components=1) ----
        requires(const std::array<FV::KurganovTadmor::internal::JacobianMatrix<double, 1>, 1> &J1_plus,
                 const std::array<FV::KurganovTadmor::internal::JacobianMatrix<double, 1>, 1> &J1_minus,
                 const FV::KurganovTadmor::internal::HessianTensor<double, 1, 1> &H1_plus,
                 const FV::KurganovTadmor::internal::HessianTensor<double, 1, 1> &H1_minus) {
          {
            T::template compute_selected_speed_derivatives<double, 1, 1>(J1_plus, J1_minus, H1_plus, H1_minus)
          } -> std::same_as<std::pair<std::array<std::array<double, 1>, 1>, std::array<std::array<double, 1>, 1>>>;
        } &&
        // ---- compute_selected_speed_derivatives (dim=2, n_components=2) ----
        requires(const std::array<FV::KurganovTadmor::internal::JacobianMatrix<double, 2>, 2> &J2_plus,
                 const std::array<FV::KurganovTadmor::internal::JacobianMatrix<double, 2>, 2> &J2_minus,
                 const FV::KurganovTadmor::internal::HessianTensor<double, 2, 2> &H2_plus,
                 const FV::KurganovTadmor::internal::HessianTensor<double, 2, 2> &H2_minus) {
          {
            T::template compute_selected_speed_derivatives<double, 2, 2>(J2_plus, J2_minus, H2_plus, H2_minus)
          } -> std::same_as<std::pair<std::array<std::array<double, 2>, 2>, std::array<std::array<double, 2>, 2>>>;
        };

  } // namespace def
} // namespace DiFfRG
