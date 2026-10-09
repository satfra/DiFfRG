#pragma once

// standard library
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// external libraries
#include <deal.II/lac/precondition.h>

// DiFfRG
#include <DiFfRG/timestepping/linear_solver/GMRES.hh>
#include <DiFfRG/timestepping/linear_solver/UMFPack.hh>
#include <DiFfRG/timestepping/linear_solver/abstract_linear_solver.hh>

namespace DiFfRG
{
  template <typename SparseMatrixType, typename VectorType,
            typename InnerSolver = GMRES<SparseMatrixType, VectorType, dealii::PreconditionIdentity>>
  class ScaledLinearSolver : public AbstractLinearSolver<SparseMatrixType, VectorType>
  {
  public:
    static constexpr bool performs_factorization = InnerSolver::performs_factorization;

    ScaledLinearSolver() = default;

    void init(const SparseMatrixType &matrix)
    {
      initialized = false;
      build_scaling(matrix);
      build_scaled_matrix(matrix);
      inner_solver.init(scaled_matrix);
      initialized = true;
    }

    /// Forwarded to the inner solver, if it has the switch; see UMFPack::set_iterative_refinement.
    void set_iterative_refinement(const bool on)
    {
      if constexpr (requires { inner_solver.set_iterative_refinement(on); }) inner_solver.set_iterative_refinement(on);
    }

    bool invert()
    {
      if (!initialized) throw std::runtime_error("ScaledLinearSolver::invert: solver not initialized");
      return inner_solver.invert();
    }

    int solve(const VectorType &src, VectorType &dst, const double tol)
    {
      if (!initialized) throw std::runtime_error("ScaledLinearSolver::solve: solver not initialized");
      if (src.size() != row_scale.size())
        throw std::runtime_error("ScaledLinearSolver::solve: source vector size does not match matrix rows");

      VectorType scaled_src(src);
      VectorType scaled_dst(src);
      scaled_dst = 0.;

      for (std::size_t i = 0; i < scaled_src.size(); ++i)
        scaled_src[i] *= row_scale[i];

      const int solver_result = inner_solver.solve(scaled_src, scaled_dst, tol);

      dst.reinit(src);
      for (std::size_t i = 0; i < dst.size(); ++i)
        dst[i] = col_scale[i] * scaled_dst[i];

      return solver_result;
    }

    double estimate_scaled_rcond(const SparseMatrixType &, const unsigned int max_iterations = 5) const
      requires(InnerSolver::performs_factorization)
    {
      return inner_solver.estimate_rcond(scaled_matrix, max_iterations);
    }

  private:
    static double safe_inverse_sqrt(const double norm)
    {
      constexpr double min_norm = 1.e-300;
      if (!std::isfinite(norm) || norm <= min_norm) return 1.;
      return 1. / std::sqrt(norm);
    }

    void build_scaling(const SparseMatrixType &matrix)
    {
      row_scale.assign(matrix.m(), 0.);
      col_scale.assign(matrix.n(), 0.);

      for (std::size_t row = 0; row < matrix.m(); ++row) {
        for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry) {
          const auto column = entry->column();
          const double value = std::abs(entry->value());
          row_scale[row] = std::max(row_scale[row], value);
          col_scale[column] = std::max(col_scale[column], value);
        }
      }

      std::transform(row_scale.begin(), row_scale.end(), row_scale.begin(), safe_inverse_sqrt);
      std::transform(col_scale.begin(), col_scale.end(), col_scale.begin(), safe_inverse_sqrt);
    }

    void build_scaled_matrix(const SparseMatrixType &matrix)
    {
      scaled_matrix.reinit(matrix.get_sparsity_pattern());
      for (std::size_t row = 0; row < matrix.m(); ++row)
        for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
          scaled_matrix.add(row, entry->column(), row_scale[row] * entry->value() * col_scale[entry->column()]);
    }

    bool initialized = false;
    SparseMatrixType scaled_matrix;
    std::vector<double> row_scale;
    std::vector<double> col_scale;
    InnerSolver inner_solver;
  };

  template <typename SparseMatrixType, typename VectorType, typename PreconditionerType = dealii::PreconditionIdentity>
  using ScaledGMRES =
      ScaledLinearSolver<SparseMatrixType, VectorType, GMRES<SparseMatrixType, VectorType, PreconditionerType>>;

  template <typename SparseMatrixType, typename VectorType>
  using ScaledUMFPack = ScaledLinearSolver<SparseMatrixType, VectorType, UMFPack<SparseMatrixType, VectorType>>;
} // namespace DiFfRG
