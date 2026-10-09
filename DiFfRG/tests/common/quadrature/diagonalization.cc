#include "catch2/catch_test_macros.hpp"
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>

#include <DiFfRG/common/quadrature/diagonalization.hh>

using namespace DiFfRG;

TEST_CASE("Test diagonalization of tridiagonal symmetric matrices", "[common]")
{
  SECTION("Test diagonalization of 2x2 matrix")
  {
    const double a = GENERATE(take(5, random(1.0, 10.0)));
    const double b = GENERATE(take(5, random(1.0, 10.0)));
    const double c = GENERATE(take(5, random(1.0, 10.0)));

    std::vector<double> diag = {a, b};
    std::vector<double> off_diag = {c, 0.};
    std::vector<double> eigvecs = {1.0, 0.0};

    const double eig_1 = (a + b - std::sqrt(std::pow(a, 2) - 2 * a * b + std::pow(b, 2) + 4 * std::pow(c, 2))) / 2.;
    const double eig_2 = (a + b + std::sqrt(std::pow(a, 2) - 2 * a * b + std::pow(b, 2) + 4 * std::pow(c, 2))) / 2.;

    const double eigvec_1 =
        -0.5 * (-a + b + std::sqrt(std::pow(a, 2) - 2 * a * b + std::pow(b, 2) + 4 * std::pow(c, 2))) /
        (c * std::sqrt(
                 1 + std::pow(-a + b + std::sqrt(std::pow(a, 2) - 2 * a * b + std::pow(b, 2) + 4 * std::pow(c, 2)), 2) /
                         (4. * std::pow(c, 2))));
    const double eigvec_2 =
        -0.5 * (-a + b - std::sqrt(std::pow(a, 2) - 2 * a * b + std::pow(b, 2) + 4 * std::pow(c, 2))) /
        (c * std::sqrt(
                 1 + std::pow(-a + b - std::sqrt(std::pow(a, 2) - 2 * a * b + std::pow(b, 2) + 4 * std::pow(c, 2)), 2) /
                         (4. * std::pow(c, 2))));

    diagonalize_tridiagonal_symmetric_matrix(diag, off_diag, eigvecs);

    std::vector<double> sorted_eig = {eig_1, eig_2};
    std::sort(sorted_eig.begin(), sorted_eig.end());

    REQUIRE(diag[0] == Catch::Approx(sorted_eig[0]));
    REQUIRE(diag[1] == Catch::Approx(sorted_eig[1]));

    std::vector<double> sorted_abs_eigvecs = {std::abs(eigvec_1), std::abs(eigvec_2)};
    std::sort(sorted_abs_eigvecs.begin(), sorted_abs_eigvecs.end());

    std::vector<double> eigvecs_abs = {std::abs(eigvecs[0]), std::abs(eigvecs[1])};
    std::sort(eigvecs_abs.begin(), eigvecs_abs.end());

    REQUIRE(eigvecs_abs[0] == Catch::Approx(sorted_abs_eigvecs[0]));
    REQUIRE(eigvecs_abs[1] == Catch::Approx(sorted_abs_eigvecs[1]));
  }
  SECTION("Test diagonalization of 3x3 matrix", "[common]")
  {
    const double a0 = GENERATE(take(5, random(1.0, 10.0)));
    const double a1 = GENERATE(take(5, random(1.0, 10.0)));
    const double a2 = GENERATE(take(5, random(1.0, 10.0)));

    const double b0 = GENERATE(take(5, random(1.0, 10.0)));
    const double b1 = GENERATE(take(5, random(1.0, 10.0)));

    std::vector<double> diag = {a0, a1, a2};
    std::vector<double> off_diag = {std::sqrt(b0), std::sqrt(b1), 0.};
    std::vector<double> eigvecs = {1.0, 0.0, 0.0};

    const double eig_1 = std::real(
        0.3333333333333333 * (a0 + a1 + a2) -
        (0.41997368329829105 * (-1. * std::pow(a0, 2) + a0 * a1 - 1. * std::pow(a1, 2) + a0 * a2 + a1 * a2 -
                                1. * std::pow(a2, 2) - 3. * b0 - 3. * b1)) /
            std::pow(
                2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) + 2. * std::pow(a1, 3) -
                    3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 - 3. * std::pow(a1, 2) * a2 -
                    3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                    9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                    std::sqrt(std::complex<double>(1, 0) * 4. *
                                  std::pow(-1. * std::pow(a0, 2) + a0 * a1 - 1. * std::pow(a1, 2) + a0 * a2 + a1 * a2 -
                                               1. * std::pow(a2, 2) - 3. * b0 - 3. * b1,
                                           3) +
                              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                           3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                                           9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1,
                                       2)),
                0.3333333333333333) +
        0.26456684199469993 *
            std::pow(
                2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) + 2. * std::pow(a1, 3) -
                    3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 - 3. * std::pow(a1, 2) * a2 -
                    3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                    9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                    std::sqrt(std::complex<double>(1, 0) * 4. *
                                  std::pow(-1. * std::pow(a0, 2) + a0 * a1 - 1. * std::pow(a1, 2) + a0 * a2 + a1 * a2 -
                                               1. * std::pow(a2, 2) - 3. * b0 - 3. * b1,
                                           3) +
                              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                           3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                                           9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1,
                                       2)),
                0.3333333333333333));
    const double eig_2 = std::real(
        0.3333333333333333 * (a0 + a1 + a2) +
        (std::complex<double>(0.20998684164914552, 0.3637078786572404) *
         (-1. * std::pow(a0, 2) + a0 * a1 - 1. * std::pow(a1, 2) + a0 * a2 + a1 * a2 - 1. * std::pow(a2, 2) - 3. * b0 -
          3. * b1)) /
            std::pow(
                2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) + 2. * std::pow(a1, 3) -
                    3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 - 3. * std::pow(a1, 2) * a2 -
                    3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                    9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                    std::sqrt(std::complex<double>(1, 0) * 4. *
                                  std::pow(-1. * std::pow(a0, 2) + a0 * a1 - 1. * std::pow(a1, 2) + a0 * a2 + a1 * a2 -
                                               1. * std::pow(a2, 2) - 3. * b0 - 3. * b1,
                                           3) +
                              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                           3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                                           9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1,
                                       2)),
                0.3333333333333333) -
        std::complex<double>(0.13228342099734997, -0.22912160616643376) *
            std::pow(
                2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) + 2. * std::pow(a1, 3) -
                    3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 - 3. * std::pow(a1, 2) * a2 -
                    3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                    9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                    std::sqrt(std::complex<double>(1, 0) * 4. *
                                  std::pow(-1. * std::pow(a0, 2) + a0 * a1 - 1. * std::pow(a1, 2) + a0 * a2 + a1 * a2 -
                                               1. * std::pow(a2, 2) - 3. * b0 - 3. * b1,
                                           3) +
                              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                           3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                                           9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1,
                                       2)),
                0.3333333333333333));
    const double eig_3 = std::real(
        0.3333333333333333 * (a0 + a1 + a2) +
        (std::complex<double>(0.20998684164914552, -0.3637078786572404) *
         (-1. * std::pow(a0, 2) + a0 * a1 - 1. * std::pow(a1, 2) + a0 * a2 + a1 * a2 - 1. * std::pow(a2, 2) - 3. * b0 -
          3. * b1)) /
            std::pow(
                2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) + 2. * std::pow(a1, 3) -
                    3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 - 3. * std::pow(a1, 2) * a2 -
                    3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                    9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                    std::sqrt(std::complex<double>(1, 0) * 4. *
                                  std::pow(-1. * std::pow(a0, 2) + a0 * a1 - 1. * std::pow(a1, 2) + a0 * a2 + a1 * a2 -
                                               1. * std::pow(a2, 2) - 3. * b0 - 3. * b1,
                                           3) +
                              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                           3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                                           9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1,
                                       2)),
                0.3333333333333333) -
        std::complex<double>(0.13228342099734997, 0.22912160616643376) *
            std::pow(
                2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) + 2. * std::pow(a1, 3) -
                    3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 - 3. * std::pow(a1, 2) * a2 -
                    3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                    9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                    std::sqrt(std::complex<double>(1, 0) * 4. *
                                  std::pow(-1. * std::pow(a0, 2) + a0 * a1 - 1. * std::pow(a1, 2) + a0 * a2 + a1 * a2 -
                                               1. * std::pow(a2, 2) - 3. * b0 - 3. * b1,
                                           3) +
                              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                           3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 +
                                           9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1,
                                       2)),
                0.3333333333333333));

    const double eigvec_1 = std::real(
        (-1. * b1 +
         (-0.3333333333333333 * a0 + 0.6666666666666667 * a1 - 0.3333333333333333 * a2 +
          (0.41997368329829105 * (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) +
                                  a0 * (a1 + a2) - 3. * b0 - 3. * b1)) /
              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                           2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                           9. * a1 * b1 + 9. * a2 * b1 +
                           std::sqrt(std::complex<double>(1, 0) * 4. *
                                         std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                      1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                  3) +
                                     std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                  std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                  2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                  a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                        9. * b0 - 18. * b1) +
                                                  9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                              2)),
                       0.3333333333333333) -
          0.26456684199469993 *
              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                           2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                           9. * a1 * b1 + 9. * a2 * b1 +
                           std::sqrt(std::complex<double>(1, 0) * 4. *
                                         std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                      1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                  3) +
                                     std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                  std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                  2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                  a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                        9. * b0 - 18. * b1) +
                                                  9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                              2)),
                       0.3333333333333333)) *
             (-0.3333333333333333 * a0 - 0.3333333333333333 * a1 + 0.6666666666666667 * a2 +
              (0.41997368329829105 * (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) +
                                      a0 * (a1 + a2) - 3. * b0 - 3. * b1)) /
                  std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                               2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                               3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                               2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                               9. * a1 * b1 + 9. * a2 * b1 +
                               std::sqrt(std::complex<double>(1, 0) * 4. *
                                             std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                          1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                      3) +
                                         std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                      std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                      3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                      a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                            3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                      9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                  2)),
                           0.3333333333333333) -
              0.26456684199469993 *
                  std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                               2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                               3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                               2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                               9. * a1 * b1 + 9. * a2 * b1 +
                               std::sqrt(std::complex<double>(1, 0) * 4. *
                                             std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                          1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                      3) +
                                         std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                      std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                      3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                      a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                            3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                      9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                  2)),
                           0.3333333333333333))) /
        (std::sqrt(std::complex<double>(1, 0) * b0) * std::sqrt(std::complex<double>(1, 0) * b1) *
         std::sqrt(
             std::complex<double>(1, 0) * 1. +
             std::pow(
                 0.3333333333333333 * a0 + 0.3333333333333333 * a1 - 0.6666666666666667 * a2 -
                     (0.41997368329829105 * (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                             1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1)) /
                         std::pow(
                             2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                 2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                 3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                 9. * a1 * b1 + 9. * a2 * b1 +
                                 std::sqrt(std::complex<double>(1, 0) * 4. *
                                               std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                            1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                        3) +
                                           std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                        std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                        3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                        18. * a2 * b0 +
                                                        a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                              3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                        9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                    2)),
                             0.3333333333333333) +
                     0.26456684199469993 *
                         std::pow(
                             2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 -
                                 3. * a0 * std::pow(a1, 2) + 2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 +
                                 12. * a0 * a1 * a2 - 3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                 3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 -
                                 18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                                 std::sqrt(std::complex<double>(1, 0) * 4. *
                                               std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                            1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                        3) +
                                           std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                        std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                        3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                        18. * a2 * b0 +
                                                        a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                              3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                        9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                    2)),
                             0.3333333333333333),
                 2) /
                 b1 +
             std::pow(
                 -1. * b1 +
                     (-0.3333333333333333 * a0 + 0.6666666666666667 * a1 - 0.3333333333333333 * a2 +
                      (0.41997368329829105 * (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) +
                                              a1 * a2 - 1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1)) /
                          std::pow(
                              2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                  2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                  3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                  2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                  9. * a1 * b1 + 9. * a2 * b1 +
                                  std::sqrt(
                                      std::complex<double>(1, 0) * 4. *
                                          std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                       1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                   3) +
                                      std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                   std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                   2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                   a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                         9. * b0 - 18. * b1) +
                                                   9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                               2)),
                              0.3333333333333333) -
                      0.26456684199469993 *
                          std::pow(
                              2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 -
                                  3. * a0 * std::pow(a1, 2) + 2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 +
                                  12. * a0 * a1 * a2 - 3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                  3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 -
                                  18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                                  std::sqrt(
                                      std::complex<double>(1, 0) * 4. *
                                          std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                       1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                   3) +
                                      std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                   std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                   2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                   a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                         9. * b0 - 18. * b1) +
                                                   9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                               2)),
                              0.3333333333333333)) *
                         (-0.3333333333333333 * a0 - 0.3333333333333333 * a1 + 0.6666666666666667 * a2 +
                          (0.41997368329829105 * (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                  1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1)) /
                              std::pow(
                                  2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                      2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                      3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                      3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 -
                                      18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                                      std::sqrt(
                                          std::complex<double>(1, 0) * 4. *
                                              std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                           1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                       3) +
                                          std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                       std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                       3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                       18. * a2 * b0 +
                                                       a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                             3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                       9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                   2)),
                                  0.3333333333333333) -
                          0.26456684199469993 *
                              std::pow(
                                  2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 -
                                      3. * a0 * std::pow(a1, 2) + 2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 +
                                      12. * a0 * a1 * a2 - 3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                      3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 -
                                      18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                                      std::sqrt(
                                          std::complex<double>(1, 0) * 4. *
                                              std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                           1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                       3) +
                                          std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                       std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                       3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                       18. * a2 * b0 +
                                                       a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                             3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                       9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                   2)),
                                  0.3333333333333333)),
                 2) /
                 (b0 * b1))));
    const double eigvec_2 = std::real(
        (-1. * b1 +
         (-0.3333333333333333 * a0 + 0.6666666666666667 * a1 - 0.3333333333333333 * a2 -
          (std::complex<double>(0.20998684164914552, 0.3637078786572404) *
           (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 -
            3. * b1)) /
              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                           2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                           9. * a1 * b1 + 9. * a2 * b1 +
                           std::sqrt(std::complex<double>(1, 0) * 4. *
                                         std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                      1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                  3) +
                                     std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                  std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                  2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                  a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                        9. * b0 - 18. * b1) +
                                                  9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                              2)),
                       0.3333333333333333) +
          std::complex<double>(0.13228342099734997, -0.22912160616643376) *
              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                           2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                           9. * a1 * b1 + 9. * a2 * b1 +
                           std::sqrt(std::complex<double>(1, 0) * 4. *
                                         std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                      1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                  3) +
                                     std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                  std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                  2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                  a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                        9. * b0 - 18. * b1) +
                                                  9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                              2)),
                       0.3333333333333333)) *
             (-0.3333333333333333 * a0 - 0.3333333333333333 * a1 + 0.6666666666666667 * a2 -
              (std::complex<double>(0.20998684164914552, 0.3637078786572404) *
               (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) + a0 * (a1 + a2) -
                3. * b0 - 3. * b1)) /
                  std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                               2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                               3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                               2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                               9. * a1 * b1 + 9. * a2 * b1 +
                               std::sqrt(std::complex<double>(1, 0) * 4. *
                                             std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                          1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                      3) +
                                         std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                      std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                      3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                      a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                            3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                      9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                  2)),
                           0.3333333333333333) +
              std::complex<double>(0.13228342099734997, -0.22912160616643376) *
                  std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                               2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                               3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                               2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                               9. * a1 * b1 + 9. * a2 * b1 +
                               std::sqrt(std::complex<double>(1, 0) * 4. *
                                             std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                          1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                      3) +
                                         std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                      std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                      3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                      a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                            3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                      9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                  2)),
                           0.3333333333333333))) /
        (std::sqrt(std::complex<double>(1, 0) * b0) * std::sqrt(std::complex<double>(1, 0) * b1) *
         std::sqrt(
             std::complex<double>(1, 0) * 1. +
             std::pow(
                 0.3333333333333333 * a0 + 0.3333333333333333 * a1 - 0.6666666666666667 * a2 +
                     (std::complex<double>(0.20998684164914552, 0.3637078786572404) *
                      (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) + a0 * (a1 + a2) -
                       3. * b0 - 3. * b1)) /
                         std::pow(
                             2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                 2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                 3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                 9. * a1 * b1 + 9. * a2 * b1 +
                                 std::sqrt(std::complex<double>(1, 0) * 4. *
                                               std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                            1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                        3) +
                                           std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                        std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                        3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                        18. * a2 * b0 +
                                                        a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                              3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                        9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                    2)),
                             0.3333333333333333) -
                     std::complex<double>(0.13228342099734997, -0.22912160616643376) *
                         std::pow(
                             2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                 2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                 3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                 9. * a1 * b1 + 9. * a2 * b1 +
                                 std::sqrt(std::complex<double>(1, 0) * 4. *
                                               std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                            1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                        3) +
                                           std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                        std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                        3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                        18. * a2 * b0 +
                                                        a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                              3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                        9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                    2)),
                             0.3333333333333333),
                 2) /
                 b1 +
             std::pow(
                 -1. * b1 +
                     (-0.3333333333333333 * a0 + 0.6666666666666667 * a1 - 0.3333333333333333 * a2 -
                      (std::complex<double>(0.20998684164914552, 0.3637078786572404) *
                       (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) + a0 * (a1 + a2) -
                        3. * b0 - 3. * b1)) /
                          std::pow(
                              2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                  2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                  3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                  2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                  9. * a1 * b1 + 9. * a2 * b1 +
                                  std::sqrt(
                                      std::complex<double>(1, 0) * 4. *
                                          std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                       1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                   3) +
                                      std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                   std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                   2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                   a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                         9. * b0 - 18. * b1) +
                                                   9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                               2)),
                              0.3333333333333333) +
                      std::complex<double>(0.13228342099734997, -0.22912160616643376) *
                          std::pow(
                              2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                  2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                  3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                  2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                  9. * a1 * b1 + 9. * a2 * b1 +
                                  std::sqrt(
                                      std::complex<double>(1, 0) * 4. *
                                          std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                       1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                   3) +
                                      std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                   std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                   2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                   a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                         9. * b0 - 18. * b1) +
                                                   9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                               2)),
                              0.3333333333333333)) *
                         (-0.3333333333333333 * a0 - 0.3333333333333333 * a1 + 0.6666666666666667 * a2 -
                          (std::complex<double>(0.20998684164914552, 0.3637078786572404) *
                           (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) +
                            a0 * (a1 + a2) - 3. * b0 - 3. * b1)) /
                              std::pow(
                                  2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                      2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                      3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                      3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 -
                                      18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                                      std::sqrt(
                                          std::complex<double>(1, 0) * 4. *
                                              std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                           1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                       3) +
                                          std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                       std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                       3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                       18. * a2 * b0 +
                                                       a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                             3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                       9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                   2)),
                                  0.3333333333333333) +
                          std::complex<double>(0.13228342099734997, -0.22912160616643376) *
                              std::pow(
                                  2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                      2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                      3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                      3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 -
                                      18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                                      std::sqrt(
                                          std::complex<double>(1, 0) * 4. *
                                              std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                           1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                       3) +
                                          std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                       std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                       3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                       18. * a2 * b0 +
                                                       a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                             3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                       9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                   2)),
                                  0.3333333333333333)),
                 2) /
                 (b0 * b1))));
    const double eigvec_3 = std::real(
        (-1. * b1 +
         (-0.3333333333333333 * a0 + 0.6666666666666667 * a1 - 0.3333333333333333 * a2 -
          (std::complex<double>(0.20998684164914552, -0.3637078786572404) *
           (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 -
            3. * b1)) /
              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                           2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                           9. * a1 * b1 + 9. * a2 * b1 +
                           std::sqrt(std::complex<double>(1, 0) * 4. *
                                         std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                      1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                  3) +
                                     std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                  std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                  2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                  a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                        9. * b0 - 18. * b1) +
                                                  9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                              2)),
                       0.3333333333333333) +
          std::complex<double>(0.13228342099734997, 0.22912160616643376) *
              std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                           2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                           3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                           2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                           9. * a1 * b1 + 9. * a2 * b1 +
                           std::sqrt(std::complex<double>(1, 0) * 4. *
                                         std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                      1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                  3) +
                                     std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                  std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                  2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                  a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                        9. * b0 - 18. * b1) +
                                                  9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                              2)),
                       0.3333333333333333)) *
             (-0.3333333333333333 * a0 - 0.3333333333333333 * a1 + 0.6666666666666667 * a2 -
              (std::complex<double>(0.20998684164914552, -0.3637078786572404) *
               (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) + a0 * (a1 + a2) -
                3. * b0 - 3. * b1)) /
                  std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                               2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                               3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                               2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                               9. * a1 * b1 + 9. * a2 * b1 +
                               std::sqrt(std::complex<double>(1, 0) * 4. *
                                             std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                          1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                      3) +
                                         std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                      std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                      3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                      a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                            3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                      9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                  2)),
                           0.3333333333333333) +
              std::complex<double>(0.13228342099734997, 0.22912160616643376) *
                  std::pow(2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                               2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                               3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                               2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                               9. * a1 * b1 + 9. * a2 * b1 +
                               std::sqrt(std::complex<double>(1, 0) * 4. *
                                             std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                          1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                      3) +
                                         std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                      std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                      3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                      a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                            3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                      9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                  2)),
                           0.3333333333333333))) /
        (std::sqrt(std::complex<double>(1, 0) * b0) * std::sqrt(std::complex<double>(1, 0) * b1) *
         std::sqrt(
             std::complex<double>(1, 0) * 1. +
             std::pow(
                 0.3333333333333333 * a0 + 0.3333333333333333 * a1 - 0.6666666666666667 * a2 +
                     (std::complex<double>(0.20998684164914552, -0.3637078786572404) *
                      (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) + a0 * (a1 + a2) -
                       3. * b0 - 3. * b1)) /
                         std::pow(
                             2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                 2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                 3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                 9. * a1 * b1 + 9. * a2 * b1 +
                                 std::sqrt(std::complex<double>(1, 0) * 4. *
                                               std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                            1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                        3) +
                                           std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                        std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                        3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                        18. * a2 * b0 +
                                                        a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                              3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                        9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                    2)),
                             0.3333333333333333) -
                     std::complex<double>(0.13228342099734997, 0.22912160616643376) *
                         std::pow(
                             2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                 2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                 3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                 9. * a1 * b1 + 9. * a2 * b1 +
                                 std::sqrt(std::complex<double>(1, 0) * 4. *
                                               std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                            1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                        3) +
                                           std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                        std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                        3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                        18. * a2 * b0 +
                                                        a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                              3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                        9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                    2)),
                             0.3333333333333333),
                 2) /
                 b1 +
             std::pow(
                 -1. * b1 +
                     (-0.3333333333333333 * a0 + 0.6666666666666667 * a1 - 0.3333333333333333 * a2 -
                      (std::complex<double>(0.20998684164914552, -0.3637078786572404) *
                       (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) + a0 * (a1 + a2) -
                        3. * b0 - 3. * b1)) /
                          std::pow(
                              2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                  2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                  3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                  2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                  9. * a1 * b1 + 9. * a2 * b1 +
                                  std::sqrt(
                                      std::complex<double>(1, 0) * 4. *
                                          std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                       1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                   3) +
                                      std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                   std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                   2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                   a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                         9. * b0 - 18. * b1) +
                                                   9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                               2)),
                              0.3333333333333333) +
                      std::complex<double>(0.13228342099734997, 0.22912160616643376) *
                          std::pow(
                              2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                  2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                  3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) - 3. * a1 * std::pow(a2, 2) +
                                  2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 - 18. * a2 * b0 - 18. * a0 * b1 +
                                  9. * a1 * b1 + 9. * a2 * b1 +
                                  std::sqrt(
                                      std::complex<double>(1, 0) * 4. *
                                          std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                       1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                   3) +
                                      std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                   std::pow(a0, 2) * (-3. * a1 - 3. * a2) - 3. * std::pow(a1, 2) * a2 +
                                                   2. * std::pow(a2, 3) - 18. * a2 * b0 +
                                                   a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 - 3. * std::pow(a2, 2) +
                                                         9. * b0 - 18. * b1) +
                                                   9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                               2)),
                              0.3333333333333333)) *
                         (-0.3333333333333333 * a0 - 0.3333333333333333 * a1 + 0.6666666666666667 * a2 -
                          (std::complex<double>(0.20998684164914552, -0.3637078786572404) *
                           (-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 - 1. * std::pow(a2, 2) +
                            a0 * (a1 + a2) - 3. * b0 - 3. * b1)) /
                              std::pow(
                                  2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                      2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                      3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                      3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 -
                                      18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                                      std::sqrt(
                                          std::complex<double>(1, 0) * 4. *
                                              std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                           1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                       3) +
                                          std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                       std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                       3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                       18. * a2 * b0 +
                                                       a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                             3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                       9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                   2)),
                                  0.3333333333333333) +
                          std::complex<double>(0.13228342099734997, 0.22912160616643376) *
                              std::pow(
                                  2. * std::pow(a0, 3) - 3. * std::pow(a0, 2) * a1 - 3. * a0 * std::pow(a1, 2) +
                                      2. * std::pow(a1, 3) - 3. * std::pow(a0, 2) * a2 + 12. * a0 * a1 * a2 -
                                      3. * std::pow(a1, 2) * a2 - 3. * a0 * std::pow(a2, 2) -
                                      3. * a1 * std::pow(a2, 2) + 2. * std::pow(a2, 3) + 9. * a0 * b0 + 9. * a1 * b0 -
                                      18. * a2 * b0 - 18. * a0 * b1 + 9. * a1 * b1 + 9. * a2 * b1 +
                                      std::sqrt(
                                          std::complex<double>(1, 0) * 4. *
                                              std::pow(-1. * std::pow(a0, 2) - 1. * std::pow(a1, 2) + a1 * a2 -
                                                           1. * std::pow(a2, 2) + a0 * (a1 + a2) - 3. * b0 - 3. * b1,
                                                       3) +
                                          std::pow(2. * std::pow(a0, 3) + 2. * std::pow(a1, 3) +
                                                       std::pow(a0, 2) * (-3. * a1 - 3. * a2) -
                                                       3. * std::pow(a1, 2) * a2 + 2. * std::pow(a2, 3) -
                                                       18. * a2 * b0 +
                                                       a0 * (-3. * std::pow(a1, 2) + 12. * a1 * a2 -
                                                             3. * std::pow(a2, 2) + 9. * b0 - 18. * b1) +
                                                       9. * a2 * b1 + a1 * (-3. * std::pow(a2, 2) + 9. * b0 + 9. * b1),
                                                   2)),
                                  0.3333333333333333)),
                 2) /
                 (b0 * b1))));

    diagonalize_tridiagonal_symmetric_matrix(diag, off_diag, eigvecs);

    std::vector<double> sorted_eig = {eig_1, eig_2, eig_3};
    std::sort(sorted_eig.begin(), sorted_eig.end());

    REQUIRE(diag[0] == Catch::Approx(sorted_eig[0]));
    REQUIRE(diag[1] == Catch::Approx(sorted_eig[1]));
    REQUIRE(diag[2] == Catch::Approx(sorted_eig[2]));

    std::vector<double> sorted_abs_eigvecs = {abs(eigvec_1), abs(eigvec_2), abs(eigvec_3)};
    std::sort(sorted_abs_eigvecs.begin(), sorted_abs_eigvecs.end());

    std::vector<double> eigvecs_abs = {abs(eigvecs[0]), abs(eigvecs[1]), abs(eigvecs[2])};
    std::sort(eigvecs_abs.begin(), eigvecs_abs.end());

    REQUIRE(eigvecs_abs[0] == Catch::Approx(sorted_abs_eigvecs[0]));
    REQUIRE(eigvecs_abs[1] == Catch::Approx(sorted_abs_eigvecs[1]));
    REQUIRE(eigvecs_abs[2] == Catch::Approx(sorted_abs_eigvecs[2]));
  }
}
