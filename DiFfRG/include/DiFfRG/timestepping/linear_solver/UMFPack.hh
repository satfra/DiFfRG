#pragma once

// external libraries
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <umfpack.h>

// standard library
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// DiFfRG
#include <DiFfRG/timestepping/linear_solver/abstract_linear_solver.hh>
#include <DiFfRG/timestepping/linear_solver/condition_estimate.hh>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief UMFPACK LU of a dealii::SparseMatrix<double> without the entries that are exactly zero.
     *
     * dealii::SparseDirectUMFPACK factorizes every entry of the sparsity pattern. Assemblers reserve their pattern
     * for every state, so for a given state many stored entries are zero (an FV jacobian of a pure-diffusion model:
     * a third), and each costs fill-in. Dropping them is exact; the pivot order, and so the roundoff, may change.
     * At 400^2 cells of a 2D KT jacobian this halves the LU (101M -> 53M entries) and the factorization time.
     * Same interface as dealii::SparseDirectUMFPACK, as far as UMFPack uses it.
     */
    class PrunedSparseDirectUMFPACK
    {
    public:
      PrunedSparseDirectUMFPACK() { umfpack_dl_defaults(control); }

      /// UMFPACK's iterative refinement of the solves: on (its default, up to two steps) or off.
      void set_iterative_refinement(const bool on)
      {
        double defaults[UMFPACK_CONTROL];
        umfpack_dl_defaults(defaults);
        control[UMFPACK_IRSTEP] = on ? defaults[UMFPACK_IRSTEP] : 0;
      }
      PrunedSparseDirectUMFPACK(const PrunedSparseDirectUMFPACK &) = delete;
      PrunedSparseDirectUMFPACK &operator=(const PrunedSparseDirectUMFPACK &) = delete;
      ~PrunedSparseDirectUMFPACK() { clear(); }

      void initialize(const dealii::SparseMatrix<double> &matrix)
      {
        clear();
        const dealii::types::suitesparse_index n = matrix.m();
        // The rows of A, handed to UMFPACK as columns: it factorizes A^T, and solve() asks for the transpose.
        Ap.assign(n + 1, 0);
        Ai.clear();
        Ax.clear();
        std::vector<std::pair<dealii::types::suitesparse_index, double>> row;
        for (dealii::types::suitesparse_index r = 0; r < n; ++r) {
          row.clear();
          for (auto it = matrix.begin(r); it != matrix.end(r); ++it)
            if (it->value() != 0.) row.emplace_back(it->column(), it->value());
          std::sort(row.begin(), row.end()); // deal.II stores the diagonal first
          for (const auto &[column, value] : row) {
            Ai.push_back(column);
            Ax.push_back(value);
          }
          Ap[r + 1] = Ai.size();
        }
        int status = umfpack_dl_symbolic(n, n, Ap.data(), Ai.data(), Ax.data(), &symbolic, control, nullptr);
        if (status != UMFPACK_OK) fail("umfpack_dl_symbolic", status);
        status = umfpack_dl_numeric(Ap.data(), Ai.data(), Ax.data(), symbolic, &numeric, control, nullptr);
        umfpack_dl_free_symbolic(&symbolic);
        if (status == UMFPACK_WARNING_singular_matrix)
          throw std::runtime_error("UMFPACK reports that the matrix is singular.");
        if (status != UMFPACK_OK) fail("umfpack_dl_numeric", status);
      }

      /// dst = A^{-1} src
      void vmult(dealii::Vector<double> &dst, const dealii::Vector<double> &src) const
      {
        dst.reinit(src.size());
        solve_into(dst, src, UMFPACK_At);
      }

      /// rhs_and_solution := A^{-1} rhs_and_solution, or A^{-T} for @p transpose
      void solve(dealii::Vector<double> &rhs_and_solution, const bool transpose = false) const
      {
        const dealii::Vector<double> rhs(rhs_and_solution);
        solve_into(rhs_and_solution, rhs, transpose ? UMFPACK_A : UMFPACK_At);
      }

    private:
      void solve_into(dealii::Vector<double> &x, const dealii::Vector<double> &b, const int system) const
      {
        if (!numeric) throw std::runtime_error("UMFPACK: solve before a factorization.");
        const int status =
            umfpack_dl_solve(system, Ap.data(), Ai.data(), Ax.data(), x.begin(), b.begin(), numeric, control, nullptr);
        if (status != UMFPACK_OK) fail("umfpack_dl_solve", status);
      }
      static void fail(const char *call, const int status)
      {
        throw std::runtime_error(std::string("UMFPACK: ") + call + " failed with status " + std::to_string(status));
      }
      void clear()
      {
        if (symbolic) umfpack_dl_free_symbolic(&symbolic);
        if (numeric) umfpack_dl_free_numeric(&numeric);
      }

      std::vector<dealii::types::suitesparse_index> Ap, Ai;
      std::vector<double> Ax;
      double control[UMFPACK_CONTROL];
      void *symbolic = nullptr, *numeric = nullptr;
    };
  } // namespace internal

  template <typename SparseMatrixType, typename VectorType>
  class UMFPack : public AbstractLinearSolver<SparseMatrixType, VectorType>
  {
  public:
    static constexpr bool performs_factorization = true;

    UMFPack() : matrix(nullptr) {}

    void init(const SparseMatrixType &matrix) { this->matrix = &matrix; }

    /// @see TimeStepperSUNDIALS_IDA's /timestepping/implicit/iterative_refinement. Only for the zero-dropping
    /// factorization of a SparseMatrix<double>; deal.II's (block matrices) always refines.
    void set_iterative_refinement(const bool on)
    {
      if constexpr (requires { solver.set_iterative_refinement(on); }) solver.set_iterative_refinement(on);
    }

    bool invert()
    {
      if (!matrix) throw std::runtime_error("UMFPack::invert: matrix not initialized");
      solver.initialize(*matrix);
      return true;
    }

    int solve(const VectorType &src, VectorType &dst, const double)
    {
      if (!matrix) throw std::runtime_error("UMFPack::solve: matrix not initialized");
      solver.vmult(dst, src);
      return -1;
    }

    void solve_transpose(const VectorType &src, VectorType &dst) const
    {
      if (!matrix) throw std::runtime_error("UMFPack::solve_transpose: matrix not initialized");
      dst = src;
      solver.solve(dst, true);
    }

    double estimate_rcond(const SparseMatrixType &input_matrix, const unsigned int max_iterations = 5) const
    {
      if (!matrix) throw std::runtime_error("UMFPack::estimate_rcond: matrix not initialized");
      const double one_norm = internal::matrix_one_norm(input_matrix);
      if (!(one_norm > 0.) || !std::isfinite(one_norm)) return std::numeric_limits<double>::quiet_NaN();

      const auto solve_direct = [&](const VectorType &src, VectorType &dst) { solver.vmult(dst, src); };
      const auto solve_direct_transpose = [&](const VectorType &src, VectorType &dst) {
        dst = src;
        solver.solve(dst, true);
      };
      const double inverse_one_norm = internal::estimate_inverse_one_norm<VectorType>(
          input_matrix.m(), solve_direct, solve_direct_transpose, max_iterations);
      if (!(inverse_one_norm > 0.) || !std::isfinite(inverse_one_norm)) return std::numeric_limits<double>::quiet_NaN();
      return std::clamp(1. / (one_norm * inverse_one_norm), 0., 1.);
    }

    double estimate_scaled_rcond(const SparseMatrixType &input_matrix, const unsigned int max_iterations = 5) const
    {
      if (!matrix) throw std::runtime_error("UMFPack::estimate_scaled_rcond: matrix not initialized");

      std::vector<double> row_scale, column_scale;
      double scaled_one_norm = 0.;
      internal::build_maximum_equilibration(input_matrix, row_scale, column_scale, scaled_one_norm);
      if (!(scaled_one_norm > 0.) || !std::isfinite(scaled_one_norm)) return std::numeric_limits<double>::quiet_NaN();

      const auto solve_scaled = [&](const VectorType &src, VectorType &dst) {
        VectorType rhs(src), solution(src);
        for (std::size_t i = 0; i < rhs.size(); ++i)
          rhs[i] /= row_scale[i];
        solver.vmult(solution, rhs);
        dst.reinit(solution);
        for (std::size_t i = 0; i < solution.size(); ++i)
          dst[i] = solution[i] / column_scale[i];
      };
      const auto solve_scaled_transpose = [&](const VectorType &src, VectorType &dst) {
        VectorType rhs(src), solution(src);
        for (std::size_t i = 0; i < rhs.size(); ++i)
          rhs[i] /= column_scale[i];
        solution = rhs;
        solver.solve(solution, true);
        dst.reinit(solution);
        for (std::size_t i = 0; i < solution.size(); ++i)
          dst[i] = solution[i] / row_scale[i];
      };

      const double inverse_one_norm = internal::estimate_inverse_one_norm<VectorType>(
          input_matrix.m(), solve_scaled, solve_scaled_transpose, max_iterations);
      if (!(inverse_one_norm > 0.) || !std::isfinite(inverse_one_norm)) return std::numeric_limits<double>::quiet_NaN();
      return std::clamp(1. / (scaled_one_norm * inverse_one_norm), 0., 1.);
    }

  private:
    const SparseMatrixType *matrix;
    /// Block matrices (LDG) keep deal.II's factorization of the full pattern.
    std::conditional_t<std::is_same_v<SparseMatrixType, dealii::SparseMatrix<double>> &&
                           std::is_same_v<VectorType, dealii::Vector<double>>,
                       internal::PrunedSparseDirectUMFPACK, dealii::SparseDirectUMFPACK>
        solver;
  };
} // namespace DiFfRG
