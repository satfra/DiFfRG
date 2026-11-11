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
      /// Which trace sets the wave speed: the one with the larger spectral radius, or both (a tie).
      enum class WaveSpeedBranch { plus, minus, average };

      /**
       * @brief The default wave speed of FV::KurganovTadmor::Assembler,
       * \f$a_d = \max\big(\rho(J^+_d), \rho(J^-_d)\big)\f$, with \f$\rho\f$ the spectral radius (largest eigenvalue modulus) of the flux jacobian
       * \f$J^\pm_d = \partial F_d/\partial u\f$ on either trace.
       *
       * For one component \f$\rho = |J|\f$, for two in closed form; for more, the eigenvalues are computed numerically
       * (Eigen). The derivative for the jacobian is that of the selected trace (half of each at a tie), by first-order
       * eigenvalue perturbation theory. For a dominant complex eigenvalue the derivative uses only the real parts of the
       * eigenvectors, i.e. it is approximate.
       */
      struct MaxEigenvalueWaveSpeed {
        /// \f$a_d = \max(\rho(J^+_d), \rho(J^-_d))\f$ per direction.
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

        /// Per direction, which trace has the larger spectral radius.
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
        /// For J = [[a, b], [c, d]]: the eigenvalues are tr/2 +- sqrt(q) with q = ((a - d)/2)^2 + bc.
        template <typename Matrix> static auto half_trace_and_discriminant(const Matrix &J)
        {
          const auto half_trace = 0.5 * (J[0][0] + J[1][1]);
          const auto half_difference = 0.5 * (J[0][0] - J[1][1]);
          return std::pair{half_trace, half_difference * half_difference + J[0][1] * J[1][0]};
        }

        /// Spectral radius of a 2x2 matrix: |tr|/2 + sqrt(q) for real eigenvalues, sqrt(det) for a complex pair.
        template <typename Matrix> static auto spectral_radius_2x2(const Matrix &J)
        {
          using std::abs, std::sqrt;
          const auto [half_trace, q] = half_trace_and_discriminant(J);
          if (q >= 0) return abs(half_trace) + sqrt(q);
          return sqrt(J[0][0] * J[1][1] - J[0][1] * J[1][0]);
        }

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
            } else if constexpr (n_components == 2) {
              max_eig_plus = spectral_radius_2x2(J_plus[d]);
              max_eig_minus = spectral_radius_2x2(J_minus[d]);
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
         * @brief \f$\partial \rho(J^\pm_d)/\partial u_c\f$ on each trace, from the flux hessian
         * \f$H_{d,ijc} = \partial^2 F_{d,i}/\partial u_j \partial u_c\f$.
         *
         * One component: \f$\mathrm{sgn}(J_d)\, H_{d,00c}\f$. Several: with \f$\lambda\f$ the eigenvalue of largest
         * modulus and \f$v, w\f$ its right and left eigenvectors, \f$\mathrm{sgn}(\mathrm{Re}\,\lambda)
         * \sum_{ij} \mathrm{Re}\,w_i\, \mathrm{Re}\,v_j\, H_{d,ijc} / \mathrm{Re}(w \cdot v)\f$ (exact for a real
         * eigenvalue).
         *
         * @return {derivatives on the + trace, derivatives on the - trace}
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

              // Two components with distinct real eigenvalues: the dominant eigenvalue lambda and its eigenvectors in
              // closed form. With J = [[a, b], [c, d]], (J - lambda) v = 0 is solved by v = (b, lambda - a) and by
              // (lambda - d, c), and w^T (J - lambda) = 0 by w = (c, lambda - a) and by (lambda - d, b); of each pair
              // the longer one is used. The result does not depend on their normalisation. Otherwise (a complex pair
              // or a repeated eigenvalue) the Eigen path below.
              auto fill_da_2x2 = [&](const auto &J, const auto &H_side,
                                     std::array<std::array<NumberType, n_components>, dim> &da) {
                if constexpr (n_components == 2) {
                  const auto [half_trace, q] = half_trace_and_discriminant(J);
                  if (!(q > NumberType(0))) return false;
                  const NumberType root = std::sqrt(q);
                  const NumberType lambda = half_trace >= NumberType(0) ? half_trace + root : half_trace - root;
                  const auto longer = [](const std::array<NumberType, 2> &x, const std::array<NumberType, 2> &y) {
                    return x[0] * x[0] + x[1] * x[1] >= y[0] * y[0] + y[1] * y[1] ? x : y;
                  };
                  const auto a = J[0][0], b = J[0][1], c = J[1][0], dd = J[1][1];
                  const auto v = longer({b, lambda - a}, {lambda - dd, c});
                  const auto w = longer({c, lambda - a}, {lambda - dd, b});
                  const NumberType sign_lam = lambda >= NumberType(0) ? NumberType(1) : NumberType(-1);
                  const NumberType wdotv = w[0] * v[0] + w[1] * v[1];
                  for (size_t comp = 0; comp < n_components; ++comp) {
                    NumberType contraction = NumberType(0);
                    for (size_t i = 0; i < 2; ++i)
                      for (size_t j = 0; j < 2; ++j)
                        contraction += w[i] * v[j] * H_side[d][i][j][comp];
                    da[d][comp] = sign_lam * contraction / wdotv;
                  }
                  return true;
                } else
                  return false;
              };

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

              if (!fill_da_2x2(J_plus[d], H_plus, da_plus)) fill_da(Jp, H_plus, da_plus);
              if (!fill_da_2x2(J_minus[d], H_minus, da_minus)) fill_da(Jm, H_minus, da_minus);
            }
          }
          return {da_plus, da_minus};
        }

        /// \f$\partial a_d/\partial u^\pm\f$: compute_speed_derivatives of the trace that sets \f$a_d\f$ (half each
        /// at a tie), zero for the other.
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
