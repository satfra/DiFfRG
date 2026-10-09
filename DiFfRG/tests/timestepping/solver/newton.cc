#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/vector.h>

#include <array>

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/timestepping/solver/newton.hh>

//--------------------------------------------
// Forward declarations
//--------------------------------------------

using MatrixType = dealii::FullMatrix<double>;
using VectorType = dealii::Vector<double>;

bool linear_test(std::array<double, 4> A_data, std::array<double, 2> b_data);
template <typename FUNf, typename FUNdf>
bool nonlinear_test(FUNf &f, FUNdf &df, VectorType &x0, const VectorType &x_solution);

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test Newton with linear system", "[linear]")
{
  REQUIRE(linear_test({{1, 2, 3, 4}}, {{1, 2}}) == true);
  REQUIRE(linear_test({{-1, 2, -3, 4}}, {{1, -1}}) == true);
  REQUIRE(linear_test({{1, 2, 3, 4}}, {{10, 2}}) == true);
  REQUIRE(linear_test({{1, -2, 30, 4}}, {{1, 2}}) == true);
}

TEST_CASE("Test Newton with nonlinear system", "[nonlinear]")
{
  auto f = [](const VectorType &x) {
    VectorType res(2);
    res(0) = x(0) * x(0) * x(1) - 1;
    res(1) = x(0) * x(0) - x(1) - 1;
    return res;
  };
  auto df = [](const VectorType &x) {
    MatrixType res(2, 2);
    res(0, 0) = 2 * x(0) * x(1);
    res(0, 1) = x(0) * x(0);
    res(1, 0) = 2 * x(0);
    res(1, 1) = -x(1);
    return res;
  };

  VectorType x0{1., 1.};
  VectorType x_solution{std::sqrt((1. + std::sqrt(5)) / 2.), (-1. + std::sqrt(5)) / 2.};
  REQUIRE(nonlinear_test(f, df, x0, x_solution) == true);

  x0 = VectorType{-1., 1.};
  x_solution = VectorType{-std::sqrt((1. + std::sqrt(5)) / 2.), (-1. + std::sqrt(5)) / 2.};
  REQUIRE(nonlinear_test(f, df, x0, x_solution) == true);
}

//--------------------------------------------
// Helper functions
//--------------------------------------------

bool linear_test(std::array<double, 4> A_data, std::array<double, 2> b_data)
{
  MatrixType A(2, 2);
  A(0, 0) = A_data[0];
  A(0, 1) = A_data[1];
  A(1, 0) = A_data[2];
  A(1, 1) = A_data[3];

  MatrixType A_inv = A;
  A_inv.invert(A);

  VectorType b(2);
  b(0) = b_data[0];
  b(1) = b_data[1];

  DiFfRG::Newton<VectorType> newton(1e-14, 1e-12, 5e-1, 11, 21);

  // The residual is Ax - b
  newton.residual = [&](VectorType &res, const VectorType &u) {
    res = 0;
    A.vmult(res, u);
    res -= b;
  };
  newton.update_jacobian = [&](const VectorType & /*u*/) {};
  newton.lin_solve = [&](VectorType &Du, const VectorType &res) { A_inv.vmult(Du, res); };

  VectorType x(2);
  newton(x);

  VectorType x_solution(2);
  A_inv.vmult(x_solution, b);

  return DiFfRG::is_close(x(0), x_solution(0)) && DiFfRG::is_close(x(1), x_solution(1));
}

template <typename FUNf, typename FUNdf>
bool nonlinear_test(FUNf &f, FUNdf &df, VectorType &x0, const VectorType &x_solution)
{
  MatrixType J(2, 2);
  MatrixType J_inv(2, 2);

  DiFfRG::Newton<VectorType> newton(1e-14, 1e-12, 1e-1, 21, 21);

  // The residual is Ax - b
  newton.residual = [&](VectorType &res, const VectorType &u) { res = f(u); };
  newton.update_jacobian = [&](const VectorType &u) {
    J = df(u);
    J_inv.invert(J);
  };
  newton.lin_solve = [&](VectorType &Du, const VectorType &res) { J_inv.vmult(Du, res); };
  newton.reinit(x0);

  newton(x0);
  return DiFfRG::is_close(x0(0), x_solution(0), 1e-12) && DiFfRG::is_close(x0(1), x_solution(1), 1e-12);
}