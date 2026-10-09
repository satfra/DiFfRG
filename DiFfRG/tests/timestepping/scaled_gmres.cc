#define CATCH_CONFIG_MAIN
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <DiFfRG/timestepping/jacobian_diagnostics.hh>
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/ScaledGMRES.hh>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <cmath>
#include <type_traits>

using namespace DiFfRG;
using namespace dealii;

namespace
{
  struct SparseSystem {
    SparsityPattern sparsity;
    SparseMatrix<double> matrix;
  };

  Vector<double> badly_scaled_expected_solution()
  {
    Vector<double> expected_solution(3);
    expected_solution[0] = 1.25;
    expected_solution[1] = -2.5e-4;
    expected_solution[2] = 3.e-3;
    return expected_solution;
  }

  template <typename MatrixType> Vector<double> matrix_vector_product(const MatrixType &matrix, const Vector<double> &x)
  {
    Vector<double> result(matrix.m());
    result = 0.;
    for (std::size_t row = 0; row < matrix.m(); ++row)
      for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
        result[row] += entry->value() * x[entry->column()];
    return result;
  }

  SparseSystem make_badly_scaled_matrix()
  {
    DynamicSparsityPattern dsp(3, 3);
    dsp.add(0, 0);
    dsp.add(0, 1);
    dsp.add(1, 0);
    dsp.add(1, 1);
    dsp.add(1, 2);
    dsp.add(2, 1);
    dsp.add(2, 2);

    SparseSystem system;
    system.sparsity.copy_from(dsp);

    system.matrix.reinit(system.sparsity);
    system.matrix.set(0, 0, 1.e-8);
    system.matrix.set(0, 1, 2.e4);
    system.matrix.set(1, 0, -3.e-5);
    system.matrix.set(1, 1, 4.);
    system.matrix.set(1, 2, 5.e5);
    system.matrix.set(2, 1, 6.e-6);
    system.matrix.set(2, 2, -7.e8);

    return system;
  }

  void build_badly_scaled_block_matrix(BlockSparsityPattern &sparsity, BlockSparseMatrix<double> &matrix)
  {
    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit(1, 1);
    dsp.block(0, 1).reinit(1, 2);
    dsp.block(1, 0).reinit(2, 1);
    dsp.block(1, 1).reinit(2, 2);

    dsp.block(0, 0).add(0, 0);
    dsp.block(0, 1).add(0, 0);
    dsp.block(1, 0).add(0, 0);
    dsp.block(1, 1).add(0, 0);
    dsp.block(1, 1).add(0, 1);
    dsp.block(1, 1).add(1, 0);
    dsp.block(1, 1).add(1, 1);
    dsp.collect_sizes();

    sparsity.copy_from(dsp);

    matrix.reinit(sparsity);
    matrix.set(0, 0, 1.e-8);
    matrix.set(0, 1, 2.e4);
    matrix.set(1, 0, -3.e-5);
    matrix.set(1, 1, 4.);
    matrix.set(1, 2, 5.e5);
    matrix.set(2, 1, 6.e-6);
    matrix.set(2, 2, -7.e8);
  }

  SparseSystem make_banded_scaled_matrix(const unsigned int n)
  {
    DynamicSparsityPattern dsp(n, n);
    for (unsigned int i = 0; i < n; ++i) {
      dsp.add(i, i);
      if (i > 0) dsp.add(i, i - 1);
      if (i + 1 < n) dsp.add(i, i + 1);
      if (i + 2 < n) dsp.add(i, i + 2);
      if (i > 2) dsp.add(i, i - 3);
    }

    SparseSystem system;
    system.sparsity.copy_from(dsp);

    system.matrix.reinit(system.sparsity);
    const std::array<double, 6> row_exponents = {-6., -3., -1., 1., 3., 6.};
    const std::array<double, 5> col_exponents = {5., -4., 2., -2., 0.};

    for (unsigned int i = 0; i < n; ++i) {
      const double row_scale = std::pow(10., row_exponents[i % row_exponents.size()]);
      const auto add_entry = [&](const unsigned int j, const double value) {
        const double col_scale = std::pow(10., col_exponents[j % col_exponents.size()]);
        system.matrix.set(i, j, row_scale * value * col_scale);
      };

      add_entry(i, 4.5 + 0.01 * double(i % 7));
      if (i > 0) add_entry(i - 1, -1.1);
      if (i + 1 < n) add_entry(i + 1, -0.9);
      if (i + 2 < n) add_entry(i + 2, 0.35);
      if (i > 2) add_entry(i - 3, -0.18);
    }

    return system;
  }

  template <typename MatrixType>
  void check_badly_scaled_solution(const MatrixType &matrix, const Vector<double> &rhs, const Vector<double> &solution)
  {
    const Vector<double> expected_solution = badly_scaled_expected_solution();
    for (unsigned int i = 0; i < solution.size(); ++i) {
      CHECK(std::isfinite(solution[i]));
      CHECK(solution[i] == Catch::Approx(expected_solution[i]).epsilon(1.e-8).margin(1.e-10));
    }

    const Vector<double> residual_check = matrix_vector_product(matrix, solution);
    for (unsigned int i = 0; i < residual_check.size(); ++i)
      CHECK(residual_check[i] == Catch::Approx(rhs[i]).epsilon(1.e-8).margin(1.e-8));
  }
} // namespace

static_assert(std::is_same_v<GMRES<SparseMatrix<double>, Vector<double>>,
                             GMRES<SparseMatrix<double>, Vector<double>, PreconditionJacobi<SparseMatrix<double>>>>);
static_assert(std::is_same_v<ScaledLinearSolver<SparseMatrix<double>, Vector<double>>,
                             ScaledLinearSolver<SparseMatrix<double>, Vector<double>,
                                                GMRES<SparseMatrix<double>, Vector<double>, PreconditionIdentity>>>);
static_assert(std::is_same_v<ScaledGMRES<SparseMatrix<double>, Vector<double>>,
                             ScaledGMRES<SparseMatrix<double>, Vector<double>, PreconditionIdentity>>);

TEST_CASE("ScaledLinearSolver defaults to scaled GMRES", "[timestepping][linear_solver][scaled_gmres]")
{
  const SparseSystem system = make_badly_scaled_matrix();
  const auto &matrix = system.matrix;
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  ScaledLinearSolver<SparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CHECK(iterations >= 0);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("ScaledGMRES solves a badly scaled sparse system", "[timestepping][linear_solver][scaled_gmres]")
{
  const SparseSystem system = make_badly_scaled_matrix();
  const auto &matrix = system.matrix;
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  GMRES<SparseMatrix<double>, Vector<double>> unscaled_solver;
  unscaled_solver.init(matrix);
  Vector<double> unscaled_solution(3);
  const int unscaled_iterations = unscaled_solver.solve(rhs, unscaled_solution, 1.e-12);

  ScaledGMRES<SparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CAPTURE(unscaled_iterations, iterations);
  CHECK(unscaled_iterations >= 0);
  CHECK(iterations >= 0);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("ScaledUMFPack solves a badly scaled sparse system", "[timestepping][linear_solver][scaled_umfpack]")
{
  const SparseSystem system = make_badly_scaled_matrix();
  const auto &matrix = system.matrix;
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  ScaledUMFPack<SparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);
  CHECK(solver.invert());

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CHECK(iterations == -1);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("Jacobian diagnostics use assembled values and ratio dominance", "[timestepping][jacobian_diagnostics]")
{
  DynamicSparsityPattern dsp(3, 3);
  dsp.add(0, 0);
  dsp.add(0, 1);
  dsp.add(0, 2); // allocated but assembled to zero
  dsp.add(1, 0);
  dsp.add(1, 1);
  dsp.add(2, 2);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> matrix(sparsity);
  matrix.set(0, 0, 4.);
  matrix.set(0, 1, -1.);
  matrix.set(1, 0, 2.);
  matrix.set(1, 1, 3.);
  matrix.set(2, 2, 0.5);

  const auto diagnostics = analyze_jacobian_matrix(matrix);
  CHECK(diagnostics.n_rows == 3.);
  CHECK(diagnostics.nnz == 5.);
  CHECK(diagnostics.max_abs_entry == 4.);
  CHECK(diagnostics.frobenius_norm == Catch::Approx(5.5));
  CHECK(diagnostics.one_norm == 6.);
  CHECK(diagnostics.infinity_norm == 5.);
  CHECK(diagnostics.min_abs_diagonal == 0.5);
  CHECK(diagnostics.max_abs_diagonal == 4.);
  CHECK(diagnostics.zero_diagonal_count == 0.);
  CHECK(diagnostics.min_diagonal_dominance == Catch::Approx(1.5));
  CHECK(diagnostics.min_row_max_abs == 0.5);
  CHECK(diagnostics.max_row_max_abs == 4.);
  CHECK(diagnostics.min_column_max_abs == 0.5);
  CHECK(diagnostics.max_column_max_abs == 4.);
}

TEST_CASE("Jacobian diagnostics handle zero and diagonal-only rows", "[timestepping][jacobian_diagnostics]")
{
  FullMatrix<double> matrix(3, 3);
  matrix(0, 0) = 2.;
  matrix(2, 1) = -3.;

  const auto diagnostics = analyze_jacobian_matrix(matrix);
  CHECK(diagnostics.nnz == 2.);
  CHECK(diagnostics.zero_diagonal_count == 2.);
  CHECK(diagnostics.min_abs_diagonal == 0.);
  CHECK(diagnostics.min_diagonal_dominance == 0.);
  CHECK(diagnostics.min_row_max_abs == 0.);
  CHECK(diagnostics.min_column_max_abs == 0.);
}

TEST_CASE("Timestepper Jacobian state tracks accepted steps and retries", "[timestepping][jacobian_diagnostics]")
{
  TimestepperJacobianDiagnosticsState state;

  const auto initial =
      state.make_build(7, ImplicitTimestepperKind::implicit_euler, ImplicitTimestepperStage::main, 0.25, 1., 1., 0.25);
  CHECK(initial.jacobian_build_id == 7);
  CHECK(initial.step == 0.);
  CHECK(initial.retry_index == 0.);
  CHECK(std::isnan(initial.last_h));

  state.finish_attempt(false, 0.25);
  const auto retry =
      state.make_build(8, ImplicitTimestepperKind::implicit_euler, ImplicitTimestepperStage::main, 0.1, 1., 1., 0.1);
  CHECK(retry.step == 0.);
  CHECK(retry.retry_index == 1.);
  CHECK(std::isnan(retry.last_h));

  state.finish_attempt(true, 0.1);
  const auto next =
      state.make_build(9, ImplicitTimestepperKind::trbdf2, ImplicitTimestepperStage::trapezoidal, 0.2, 1., 1., 0.1);
  CHECK(next.step == 1.);
  CHECK(next.retry_index == 0.);
  CHECK(next.last_h == 0.1);
  CHECK(next.current_h == 0.2);
}

TEST_CASE("Jacobian diagnostics support block sparse matrices", "[timestepping][jacobian_diagnostics]")
{
  BlockSparsityPattern sparsity;
  BlockSparseMatrix<double> matrix;
  build_badly_scaled_block_matrix(sparsity, matrix);

  const auto diagnostics = analyze_jacobian_matrix(matrix);
  CHECK(diagnostics.n_rows == 3.);
  CHECK(diagnostics.nnz == 7.);
  CHECK(diagnostics.max_abs_entry == 7.e8);
  CHECK(diagnostics.one_norm == Catch::Approx(7.005e8));
  CHECK(diagnostics.infinity_norm == Catch::Approx(7.e8 + 6.e-6));
  CHECK(diagnostics.zero_diagonal_count == 0.);
}

TEST_CASE("GMRES leaves factorization-only diagnostics unavailable", "[timestepping][jacobian_diagnostics][gmres]")
{
  const SparseSystem system = make_badly_scaled_matrix();
  GMRES<SparseMatrix<double>, Vector<double>> solver;
  solver.init(system.matrix);
  JacobianFactorizationDiagnostics diagnostics;

  factorize_with_diagnostics(solver, system.matrix, diagnostics, true);

  CHECK(std::isnan(diagnostics.factorization_ms));
  CHECK(std::isnan(diagnostics.factorization_success));
  CHECK(std::isnan(diagnostics.scaled_rcond_estimate));
}

TEST_CASE("Failed UMFPack factorization is recorded", "[timestepping][jacobian_diagnostics][umfpack]")
{
  DynamicSparsityPattern dsp(2, 2);
  for (unsigned int row = 0; row < 2; ++row)
    for (unsigned int column = 0; column < 2; ++column)
      dsp.add(row, column);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> matrix(sparsity);
  matrix.set(0, 0, 1.);
  matrix.set(0, 1, 1.);
  matrix.set(1, 0, 2.);
  matrix.set(1, 1, 2.);

  UMFPack<SparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);
  JacobianFactorizationDiagnostics diagnostics;

  CHECK_THROWS(factorize_with_diagnostics(solver, matrix, diagnostics, true));
  CHECK(diagnostics.factorization_ms >= 0.);
  CHECK(diagnostics.factorization_success == 0.);
  CHECK(std::isnan(diagnostics.scaled_rcond_estimate));
}

TEST_CASE("UMFPack estimates the condition of the equilibrated matrix",
          "[timestepping][jacobian_diagnostics][scaled_umfpack]")
{
  DynamicSparsityPattern dsp(3, 3);
  for (unsigned int i = 0; i < 3; ++i)
    dsp.add(i, i);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);
  SparseMatrix<double> matrix(sparsity);
  matrix.set(0, 0, 1.e-12);
  matrix.set(1, 1, 1.);
  matrix.set(2, 2, 1.e12);

  UMFPack<SparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);
  REQUIRE(solver.invert());
  CHECK(solver.estimate_scaled_rcond(matrix, 5) == Catch::Approx(1.).epsilon(1.e-12));

  ScaledUMFPack<SparseMatrix<double>, Vector<double>> scaled_solver;
  scaled_solver.init(matrix);
  REQUIRE(scaled_solver.invert());
  CHECK(scaled_solver.estimate_scaled_rcond(matrix, 5) == Catch::Approx(1.).epsilon(1.e-12));
}

TEST_CASE("ScaledLinearSolver wraps UMFPack explicitly", "[timestepping][linear_solver][scaled_umfpack]")
{
  const SparseSystem system = make_badly_scaled_matrix();
  const auto &matrix = system.matrix;
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  using DirectSolver = UMFPack<SparseMatrix<double>, Vector<double>>;
  ScaledLinearSolver<SparseMatrix<double>, Vector<double>, DirectSolver> solver;
  solver.init(matrix);
  CHECK(solver.invert());

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CHECK(iterations == -1);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("ScaledUMFPack solves a badly scaled block sparse system", "[timestepping][linear_solver][scaled_umfpack]")
{
  BlockSparsityPattern sparsity;
  BlockSparseMatrix<double> matrix;
  build_badly_scaled_block_matrix(sparsity, matrix);
  const Vector<double> rhs = matrix_vector_product(matrix, badly_scaled_expected_solution());

  ScaledUMFPack<BlockSparseMatrix<double>, Vector<double>> solver;
  solver.init(matrix);
  CHECK(solver.invert());

  Vector<double> solution(3);
  const int iterations = solver.solve(rhs, solution, 1.e-12);

  CHECK(iterations == -1);
  check_badly_scaled_solution(matrix, rhs, solution);
}

TEST_CASE("ScaledGMRES preconditioner choice is explicit on a badly scaled sparse system",
          "[timestepping][linear_solver][scaled_gmres][preconditioner]")
{
  const SparseSystem badly_scaled_system = make_badly_scaled_matrix();
  ScaledGMRES<SparseMatrix<double>, Vector<double>, PreconditionIdentity> identity_solver;
  identity_solver.init(badly_scaled_system.matrix);
  const Vector<double> identity_rhs =
      matrix_vector_product(badly_scaled_system.matrix, badly_scaled_expected_solution());
  Vector<double> identity_solution(3);
  const int identity_iterations = identity_solver.solve(identity_rhs, identity_solution, 1.e-12);

  CAPTURE(identity_iterations);
  CHECK(identity_iterations >= 0);
  check_badly_scaled_solution(badly_scaled_system.matrix, identity_rhs, identity_solution);

  const SparseSystem banded_system = make_banded_scaled_matrix(120);
  const auto &banded_matrix = banded_system.matrix;

  Vector<double> expected_solution(banded_matrix.n());
  for (unsigned int i = 0; i < expected_solution.size(); ++i)
    expected_solution[i] = std::sin(0.13 * double(i + 1)) + 0.25 * std::cos(0.07 * double(i));

  const Vector<double> rhs = matrix_vector_product(banded_matrix, expected_solution);

  ScaledGMRES<SparseMatrix<double>, Vector<double>, PreconditionJacobi<SparseMatrix<double>>> jacobi_solver;
  jacobi_solver.init(banded_matrix);
  Vector<double> jacobi_solution(banded_matrix.n());
  const int jacobi_iterations = jacobi_solver.solve(rhs, jacobi_solution, 1.e-10);

  const Vector<double> jacobi_residual = matrix_vector_product(banded_matrix, jacobi_solution);
  CAPTURE(identity_iterations, jacobi_iterations);
  CHECK(jacobi_iterations >= 0);
  for (unsigned int i = 0; i < jacobi_solution.size(); ++i) {
    CHECK(std::isfinite(jacobi_solution[i]));
    CHECK(jacobi_residual[i] == Catch::Approx(rhs[i]).epsilon(1.e-8).margin(1.e-7));
  }
}
