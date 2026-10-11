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
     * @brief Concept of a wave-speed strategy, the WaveSpeedStrategy_ of FV::KurganovTadmor::Assembler.
     *
     * The local wave speed \f$a_d\f$ of a face sets the dissipation \f$\tfrac12 a_d n_d (u^+ - u^-)\f$ of the KT flux.
     * A strategy provides, from the flux jacobians \f$J^\pm_d = \partial F_d/\partial u\f$ on both traces (restricted
     * to one block of Model::wave_speed_blocks):
     * - `compute_speeds`: \f$a_d\f$ per direction \f$d\f$;
     * - `compute_selected_speed_derivatives`: \f$\partial a_d/\partial u^+\f$ and \f$\partial a_d/\partial u^-\f$,
     *   from the flux hessians \f$H^\pm\f$, for the jacobian. They must be the derivatives of what compute_speeds
     *   returns, else the Newton iteration is inexact.
     *
     * Implementation: MaxEigenvalueWaveSpeed. A new strategy only has to satisfy this concept:
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
