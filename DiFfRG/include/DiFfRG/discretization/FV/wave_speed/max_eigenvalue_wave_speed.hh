#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <utility>

#include <DiFfRG/discretization/FV/assembler/flux_jacobian_hessian.hh>

namespace DiFfRG
{
  namespace FV
  {
    namespace KurganovTadmor
    {
      enum class WaveSpeedBranch { plus, minus, average };

      /**
       * @brief Default wave-speed strategy.
       *
       * compute_speeds: a[d] = max(spectral_radius(J_plus[d]), spectral_radius(J_minus[d]))
       * compute_speed_derivatives: analytical derivative via eigenvector perturbation theory
       * compute_selected_speed_derivatives: derivative for the branch selected by compute_speeds
       */
      struct MaxEigenvalueWaveSpeed {
        template <typename NumberType, int dim, size_t n_components>
        static std::array<NumberType, dim>
        compute_speeds(const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
                       const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus)
        {
          const auto [spectral_radius_plus, spectral_radius_minus] =
              compute_spectral_radii<NumberType, dim, n_components>(J_plus, J_minus);

          std::array<NumberType, dim> a{};
          for (size_t d = 0; d < dim; ++d)
            a[d] = std::max(spectral_radius_plus[d], spectral_radius_minus[d]);
          return a;
        }

        template <typename NumberType, int dim, size_t n_components>
        static std::array<WaveSpeedBranch, dim>
        select_speed_branches(const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
                              const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus)
        {
          const auto [spectral_radius_plus, spectral_radius_minus] =
              compute_spectral_radii<NumberType, dim, n_components>(J_plus, J_minus);

          std::array<WaveSpeedBranch, dim> branches{};
          for (size_t d = 0; d < dim; ++d) {
            if (spectral_radius_plus[d] > spectral_radius_minus[d])
              branches[d] = WaveSpeedBranch::plus;
            else if (spectral_radius_minus[d] > spectral_radius_plus[d])
              branches[d] = WaveSpeedBranch::minus;
            else
              branches[d] = WaveSpeedBranch::average;
          }
          return branches;
        }

      private:
        template <typename NumberType, int dim, size_t n_components>
        static std::pair<std::array<NumberType, dim>, std::array<NumberType, dim>>
        compute_spectral_radii(const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
                               const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus)
        {
          std::array<NumberType, dim> spectral_radius_plus{}, spectral_radius_minus{};
          for (size_t d = 0; d < dim; ++d) {
            NumberType max_eig_plus = 0.0;
            NumberType max_eig_minus = 0.0;

            if constexpr (n_components == 1) {
              max_eig_plus = std::abs(J_plus[d][0][0]);
              max_eig_minus = std::abs(J_minus[d][0][0]);
            } else {
              Eigen::Matrix<NumberType, n_components, n_components> J_plus_eigen, J_minus_eigen;
              for (size_t i = 0; i < n_components; ++i)
                for (size_t j = 0; j < n_components; ++j) {
                  J_plus_eigen(i, j) = J_plus[d][i][j];
                  J_minus_eigen(i, j) = J_minus[d][i][j];
                }
              Eigen::EigenSolver<Eigen::Matrix<NumberType, n_components, n_components>> es_plus(J_plus_eigen);
              Eigen::EigenSolver<Eigen::Matrix<NumberType, n_components, n_components>> es_minus(J_minus_eigen);
              max_eig_plus = es_plus.eigenvalues().cwiseAbs().maxCoeff();
              max_eig_minus = es_minus.eigenvalues().cwiseAbs().maxCoeff();
            }

            spectral_radius_plus[d] = max_eig_plus;
            spectral_radius_minus[d] = max_eig_minus;
          }
          return {spectral_radius_plus, spectral_radius_minus};
        }

      public:
        /**
         * @brief Compute da[d]/du_c analytically from J and H:
         *   a[d] = spectral_radius(J[d])
         *   da[d]/du_c = d(spectral_radius)/dJ[d] : H[d][:][:][ c]
         *
         * For n_components == 1:
         *   a[d] = |J[d][0][0]|,  da/du_c = sign(J[d][0][0]) * H[d][0][0][c]
         *
         * For n_components > 1:
         *   Find the dominant eigenvalue lambda* and its right/left eigenvectors v, w
         *   (Eigen EigenSolver).  Then by first-order eigenvalue perturbation theory:
         *     da[d]/du_c = sign(Re(lambda*)) * sum_{i,j} Re(w[i]) * Re(v[j]) * H[d][i][j][c]
         *                  / Re(w . v)
         *
         * @return {da_plus, da_minus}
         */
        template <typename NumberType, int dim, size_t n_components>
        static std::pair<std::array<std::array<NumberType, n_components>, dim>,
                         std::array<std::array<NumberType, n_components>, dim>>
        compute_speed_derivatives(const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
                                  const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus,
                                  const internal::HessianTensor<NumberType, dim, n_components> &H_plus,
                                  const internal::HessianTensor<NumberType, dim, n_components> &H_minus)
        {
          std::array<std::array<NumberType, n_components>, dim> da_plus{}, da_minus{};
          for (size_t d = 0; d < dim; ++d) {
            if constexpr (n_components == 1) {
              const NumberType sign_J_plus = (J_plus[d][0][0] >= NumberType(0)) ? NumberType(1) : NumberType(-1);
              const NumberType sign_J_minus = (J_minus[d][0][0] >= NumberType(0)) ? NumberType(1) : NumberType(-1);
              for (size_t c = 0; c < n_components; ++c) {
                da_plus[d][c] = sign_J_plus * H_plus[d][0][0][c];
                da_minus[d][c] = sign_J_minus * H_minus[d][0][0][c];
              }
            } else {
              Eigen::Matrix<NumberType, n_components, n_components> Jp, Jm;
              for (size_t i = 0; i < n_components; ++i)
                for (size_t j = 0; j < n_components; ++j) {
                  Jp(i, j) = J_plus[d][i][j];
                  Jm(i, j) = J_minus[d][i][j];
                }

              // Helper lambda: fill da[d] from Jacobian eigen-decomposition and Hessian
              auto fill_da = [&](const Eigen::Matrix<NumberType, n_components, n_components> &J, const auto &H_side,
                                 std::array<std::array<NumberType, n_components>, dim> &da) {
                using CplxVec = Eigen::Matrix<std::complex<NumberType>, n_components, 1>;
                using CplxMat = Eigen::Matrix<std::complex<NumberType>, n_components, n_components>;

                Eigen::EigenSolver<Eigen::Matrix<NumberType, n_components, n_components>> es(J);
                // Find index of eigenvalue with largest absolute value
                Eigen::Index idx;
                es.eigenvalues().cwiseAbs().maxCoeff(&idx);

                const NumberType sign_lam =
                    (es.eigenvalues()(idx).real() >= NumberType(0)) ? NumberType(1) : NumberType(-1);

                // Materialize both eigenvectors to avoid dangling lazy-expression issues.
                const CplxVec right_evec = es.eigenvectors().col(idx);
                const CplxMat V_inv = es.eigenvectors().inverse();
                const CplxVec left_evec = V_inv.row(idx).transpose();

                // Bilinear form w^T * v (no conjugation) — this equals 1 by construction of V^{-1}.
                const NumberType wdotv = (left_evec.transpose() * right_evec)(0, 0).real();

                for (size_t c = 0; c < n_components; ++c) {
                  NumberType contraction = NumberType(0);
                  for (size_t i = 0; i < n_components; ++i)
                    for (size_t j = 0; j < n_components; ++j)
                      contraction += left_evec(i).real() * right_evec(j).real() * H_side[d][i][j][c];
                  da[d][c] = sign_lam * contraction / wdotv;
                }
              };

              fill_da(Jp, H_plus, da_plus);
              fill_da(Jm, H_minus, da_minus);
            }
          }
          return {da_plus, da_minus};
        }

        template <typename NumberType, int dim, size_t n_components>
        static std::pair<std::array<std::array<NumberType, n_components>, dim>,
                         std::array<std::array<NumberType, n_components>, dim>>
        compute_selected_speed_derivatives(
            const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_plus,
            const std::array<internal::JacobianMatrix<NumberType, n_components>, dim> &J_minus,
            const internal::HessianTensor<NumberType, dim, n_components> &H_plus,
            const internal::HessianTensor<NumberType, dim, n_components> &H_minus)
        {
          const auto [raw_da_plus, raw_da_minus] =
              compute_speed_derivatives<NumberType, dim, n_components>(J_plus, J_minus, H_plus, H_minus);
          const auto branches = select_speed_branches<NumberType, dim, n_components>(J_plus, J_minus);

          std::array<std::array<NumberType, n_components>, dim> da_plus{}, da_minus{};
          for (size_t d = 0; d < dim; ++d) {
            if (branches[d] == WaveSpeedBranch::plus)
              da_plus[d] = raw_da_plus[d];
            else if (branches[d] == WaveSpeedBranch::minus)
              da_minus[d] = raw_da_minus[d];
            else {
              for (size_t c = 0; c < n_components; ++c) {
                da_plus[d][c] = NumberType(0.5) * raw_da_plus[d][c];
                da_minus[d][c] = NumberType(0.5) * raw_da_minus[d][c];
              }
            }
          }

          return {da_plus, da_minus};
        }
      };

    } // namespace KurganovTadmor
  } // namespace FV
} // namespace DiFfRG
