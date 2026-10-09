#pragma once

#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/discretization/data/diagnostic_port.hh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace DiFfRG
{
  struct JacobianMatrixDiagnostics {
    double n_rows = 0.;
    double nnz = 0.;
    double max_abs_entry = 0.;
    double frobenius_norm = 0.;
    double one_norm = 0.;
    double infinity_norm = 0.;
    double min_abs_diagonal = 0.;
    double max_abs_diagonal = 0.;
    double zero_diagonal_count = 0.;
    double min_diagonal_dominance = 0.;
    double min_row_max_abs = 0.;
    double max_row_max_abs = 0.;
    double min_column_max_abs = 0.;
    double max_column_max_abs = 0.;
  };

  enum class ImplicitTimestepperKind : unsigned int {
    ida = 0,
    ida_boost_rk = 1,
    ida_boost_abm = 2,
    implicit_euler = 3,
    trbdf2 = 4
  };

  enum class ImplicitTimestepperStage : unsigned int { main = 0, trapezoidal = 1, bdf2 = 2 };

  struct TimestepperJacobianBuildDiagnostics {
    std::size_t jacobian_build_id = 0;
    ImplicitTimestepperKind stepper_kind = ImplicitTimestepperKind::ida;
    ImplicitTimestepperStage stage = ImplicitTimestepperStage::main;
    double step = std::numeric_limits<double>::quiet_NaN();
    double retry_index = std::numeric_limits<double>::quiet_NaN();
    double ida_step = std::numeric_limits<double>::quiet_NaN();
    double alpha = std::numeric_limits<double>::quiet_NaN();
    double beta = std::numeric_limits<double>::quiet_NaN();
    double residual_weight = std::numeric_limits<double>::quiet_NaN();
    double last_h = std::numeric_limits<double>::quiet_NaN();
    double current_h = std::numeric_limits<double>::quiet_NaN();
  };

  class TimestepperJacobianDiagnosticsState
  {
  public:
    TimestepperJacobianBuildDiagnostics make_build(const std::size_t build_id,
                                                   const ImplicitTimestepperKind stepper_kind,
                                                   const ImplicitTimestepperStage stage, const double current_h,
                                                   const double alpha, const double beta,
                                                   const double residual_weight) const
    {
      TimestepperJacobianBuildDiagnostics diagnostics;
      diagnostics.jacobian_build_id = build_id;
      diagnostics.stepper_kind = stepper_kind;
      diagnostics.stage = stage;
      diagnostics.step = static_cast<double>(accepted_steps);
      diagnostics.retry_index = static_cast<double>(retry_index);
      diagnostics.alpha = alpha;
      diagnostics.beta = beta;
      diagnostics.residual_weight = residual_weight;
      diagnostics.last_h = last_accepted_h;
      diagnostics.current_h = current_h;
      return diagnostics;
    }

    void finish_attempt(const bool accepted, const double attempted_h)
    {
      if (accepted) {
        ++accepted_steps;
        retry_index = 0;
        last_accepted_h = attempted_h;
      } else {
        ++retry_index;
      }
    }

  private:
    std::size_t accepted_steps = 0;
    std::size_t retry_index = 0;
    double last_accepted_h = std::numeric_limits<double>::quiet_NaN();
  };

  struct JacobianFactorizationDiagnostics {
    double factorization_ms = std::numeric_limits<double>::quiet_NaN();
    double factorization_success = std::numeric_limits<double>::quiet_NaN();
    double scaled_rcond_estimate = std::numeric_limits<double>::quiet_NaN();
  };

  namespace internal
  {
    /**
     * @brief The global rows this rank may read.
     *
     * PETSc's MatGetRow works only for locally owned rows -- reading someone else's row is an
     * error, not a slow path -- so a diagnostic that walks 0..n unconditionally aborts every rank
     * but the one owning row 0. Distributed matrices expose local_range(); serial ones do not, and
     * for them the whole matrix is local.
     */
    template <typename MatrixType> std::pair<std::size_t, std::size_t> owned_row_range(const MatrixType &matrix)
    {
      if constexpr (requires { matrix.local_range(); }) {
        const auto range = matrix.local_range();
        return {static_cast<std::size_t>(range.first), static_cast<std::size_t>(range.second)};
      } else {
        return {std::size_t(0), static_cast<std::size_t>(matrix.m())};
      }
    }

    template <typename MatrixType> bool is_distributed_matrix(const MatrixType &)
    {
      return requires(const MatrixType &m) { m.local_range(); };
    }

    template <typename MatrixType, typename Visitor>
    void visit_matrix_row(const MatrixType &matrix, const std::size_t row, Visitor &&visitor)
    {
      if constexpr (requires {
                      matrix.begin(row);
                      matrix.end(row);
                    }) {
        for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
          visitor(static_cast<std::size_t>(entry->column()), static_cast<double>(entry->value()));
      } else {
        for (std::size_t column = 0; column < matrix.n(); ++column)
          visitor(column, static_cast<double>(matrix(row, column)));
      }
    }
  } // namespace internal

  template <typename MatrixType> JacobianMatrixDiagnostics analyze_jacobian_matrix(const MatrixType &matrix)
  {
    if (matrix.m() != matrix.n()) throw std::invalid_argument("analyze_jacobian_matrix: Jacobian must be square");

    JacobianMatrixDiagnostics result;
    const std::size_t n = matrix.m();
    result.n_rows = static_cast<double>(n);
    if (n == 0) return result;

    std::vector<double> column_sums(n, 0.);
    std::vector<double> column_maxima(n, 0.);
    double squared_frobenius_norm = 0.;
    result.min_abs_diagonal = std::numeric_limits<double>::infinity();
    result.min_diagonal_dominance = std::numeric_limits<double>::infinity();
    result.min_row_max_abs = std::numeric_limits<double>::infinity();

    // Only this rank's rows; the partial results are reduced below. On a serial matrix the range
    // is the whole matrix and the reduction is a no-op.
    const auto [row_begin, row_end] = internal::owned_row_range(matrix);
    for (std::size_t row = row_begin; row < row_end; ++row) {
      double diagonal = 0.;
      double row_sum = 0.;
      double row_maximum = 0.;
      double off_diagonal_sum = 0.;

      internal::visit_matrix_row(matrix, row, [&](const std::size_t column, const double value) {
        const double abs_value = std::abs(value);
        if (value != 0.) result.nnz += 1.;
        result.max_abs_entry = std::max(result.max_abs_entry, abs_value);
        squared_frobenius_norm += abs_value * abs_value;
        row_sum += abs_value;
        row_maximum = std::max(row_maximum, abs_value);
        column_sums[column] += abs_value;
        column_maxima[column] = std::max(column_maxima[column], abs_value);
        if (column == row)
          diagonal = abs_value;
        else
          off_diagonal_sum += abs_value;
      });

      result.infinity_norm = std::max(result.infinity_norm, row_sum);
      result.min_row_max_abs = std::min(result.min_row_max_abs, row_maximum);
      result.max_row_max_abs = std::max(result.max_row_max_abs, row_maximum);
      result.min_abs_diagonal = std::min(result.min_abs_diagonal, diagonal);
      result.max_abs_diagonal = std::max(result.max_abs_diagonal, diagonal);
      if (diagonal == 0.) result.zero_diagonal_count += 1.;

      const double dominance = off_diagonal_sum > 0. ? diagonal / off_diagonal_sum
                                                     : (diagonal > 0. ? std::numeric_limits<double>::infinity() : 0.);
      result.min_diagonal_dominance = std::min(result.min_diagonal_dominance, dominance);
    }

    if constexpr (requires { matrix.get_mpi_communicator(); }) {
      // Combine the per-rank partials. Column quantities are accumulated across ranks first,
      // because a column is generally touched by rows several ranks own -- taking the extremum of
      // per-rank column sums would understate the one-norm.
      const auto comm = matrix.get_mpi_communicator();
      MPI::sum_reduce(comm, column_sums.data(), static_cast<int>(column_sums.size()));
      MPI::max_reduce(comm, column_maxima.data(), static_cast<int>(column_maxima.size()));

      squared_frobenius_norm = MPI::sum_reduce(comm, squared_frobenius_norm);
      result.nnz = MPI::sum_reduce(comm, result.nnz);
      result.zero_diagonal_count = MPI::sum_reduce(comm, result.zero_diagonal_count);

      result.max_abs_entry = MPI::max_reduce(comm, result.max_abs_entry);
      result.infinity_norm = MPI::max_reduce(comm, result.infinity_norm);
      result.max_row_max_abs = MPI::max_reduce(comm, result.max_row_max_abs);
      result.max_abs_diagonal = MPI::max_reduce(comm, result.max_abs_diagonal);
      result.min_row_max_abs = MPI::min_reduce(comm, result.min_row_max_abs);
      result.min_abs_diagonal = MPI::min_reduce(comm, result.min_abs_diagonal);
      result.min_diagonal_dominance = MPI::min_reduce(comm, result.min_diagonal_dominance);
    }

    result.frobenius_norm = std::sqrt(squared_frobenius_norm);
    result.min_column_max_abs = *std::min_element(column_maxima.begin(), column_maxima.end());
    result.max_column_max_abs = *std::max_element(column_maxima.begin(), column_maxima.end());
    result.one_norm = *std::max_element(column_sums.begin(), column_sums.end());
    return result;
  }

  inline void record_jacobian_diagnostics(const DiagnosticPort &diagnostics, const std::string &table, const double t,
                                          const TimestepperJacobianBuildDiagnostics &build,
                                          const JacobianMatrixDiagnostics &matrix,
                                          const JacobianFactorizationDiagnostics &factorization)
  {
    diagnostics.record(table, t,
                       {{"jacobian_build_id", static_cast<double>(build.jacobian_build_id)},
                        {"stepper_kind", static_cast<double>(build.stepper_kind)},
                        {"stage", static_cast<double>(build.stage)},
                        {"step", build.step},
                        {"retry_index", build.retry_index},
                        {"ida_step", build.ida_step},
                        {"alpha", build.alpha},
                        {"beta", build.beta},
                        {"residual_weight", build.residual_weight},
                        {"last_h", build.last_h},
                        {"current_h", build.current_h},
                        {"n_rows", matrix.n_rows},
                        {"nnz", matrix.nnz},
                        {"max_abs_entry", matrix.max_abs_entry},
                        {"frobenius_norm", matrix.frobenius_norm},
                        {"one_norm", matrix.one_norm},
                        {"infinity_norm", matrix.infinity_norm},
                        {"min_abs_diagonal", matrix.min_abs_diagonal},
                        {"max_abs_diagonal", matrix.max_abs_diagonal},
                        {"zero_diagonal_count", matrix.zero_diagonal_count},
                        {"min_diagonal_dominance", matrix.min_diagonal_dominance},
                        {"min_row_max_abs", matrix.min_row_max_abs},
                        {"max_row_max_abs", matrix.max_row_max_abs},
                        {"min_column_max_abs", matrix.min_column_max_abs},
                        {"max_column_max_abs", matrix.max_column_max_abs},
                        {"factorization_ms", factorization.factorization_ms},
                        {"factorization_success", factorization.factorization_success},
                        {"scaled_rcond_estimate", factorization.scaled_rcond_estimate}});
  }

  /** Factorizes and records how it went. The condition estimate costs several extra triangular
   * solves per call, so `estimate_condition` switches it off when the diagnostics are not written;
   * the success flag and the timing are always filled, as callers branch on them. */
  template <typename LinearSolver, typename MatrixType>
  void factorize_with_diagnostics(LinearSolver &solver, const MatrixType &matrix,
                                  JacobianFactorizationDiagnostics &diagnostics, const bool estimate_condition)
  {
    if constexpr (LinearSolver::performs_factorization) {
      const auto start = std::chrono::steady_clock::now();
      try {
        diagnostics.factorization_success = solver.invert() ? 1. : 0.;
      } catch (...) {
        diagnostics.factorization_success = 0.;
        diagnostics.factorization_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        throw;
      }
      diagnostics.factorization_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      if (estimate_condition && diagnostics.factorization_success == 1.) {
        try {
          diagnostics.scaled_rcond_estimate = solver.estimate_scaled_rcond(matrix, 5);
        } catch (...) {
          diagnostics.scaled_rcond_estimate = std::numeric_limits<double>::quiet_NaN();
        }
      }
    } else {
      solver.invert();
    }
  }
} // namespace DiFfRG
