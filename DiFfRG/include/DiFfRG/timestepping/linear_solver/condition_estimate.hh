#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace DiFfRG::internal
{
  template <typename MatrixType> double matrix_one_norm(const MatrixType &matrix)
  {
    std::vector<double> column_sums(matrix.n(), 0.);
    for (std::size_t row = 0; row < matrix.m(); ++row)
      for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
        column_sums[entry->column()] += std::abs(static_cast<double>(entry->value()));
    return column_sums.empty() ? 0. : *std::max_element(column_sums.begin(), column_sums.end());
  }

  template <typename VectorType, typename Solve, typename SolveTranspose>
  double estimate_inverse_one_norm(const std::size_t n, Solve &&solve, SolveTranspose &&solve_transpose,
                                   const unsigned int max_iterations = 5)
  {
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();

    VectorType x(n), y(n), signs(n), z(n);
    x = 1. / static_cast<double>(n);
    double estimate = 0.;
    std::size_t previous_index = n;

    for (unsigned int iteration = 0; iteration < max_iterations; ++iteration) {
      solve(x, y);
      const double next_estimate = y.l1_norm();
      if (!std::isfinite(next_estimate)) return std::numeric_limits<double>::quiet_NaN();
      estimate = std::max(estimate, next_estimate);

      for (std::size_t i = 0; i < n; ++i)
        signs[i] = y[i] >= 0. ? 1. : -1.;
      solve_transpose(signs, z);

      std::size_t index = 0;
      double maximum = 0.;
      for (std::size_t i = 0; i < n; ++i) {
        const double value = std::abs(z[i]);
        if (value > maximum) {
          maximum = value;
          index = i;
        }
      }

      if (index == previous_index) break;
      previous_index = index;
      x = 0.;
      x[index] = 1.;
    }

    return estimate;
  }

  template <typename MatrixType>
  void build_maximum_equilibration(const MatrixType &matrix, std::vector<double> &row_scale,
                                   std::vector<double> &column_scale, double &scaled_one_norm)
  {
    constexpr double min_norm = 1.e-300;
    row_scale.assign(matrix.m(), 0.);
    column_scale.assign(matrix.n(), 0.);

    for (std::size_t row = 0; row < matrix.m(); ++row)
      for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry) {
        const double value = std::abs(static_cast<double>(entry->value()));
        row_scale[row] = std::max(row_scale[row], value);
        column_scale[entry->column()] = std::max(column_scale[entry->column()], value);
      }

    const auto inverse_sqrt = [](const double norm) {
      if (!std::isfinite(norm) || norm <= min_norm) return 1.;
      return 1. / std::sqrt(norm);
    };
    std::transform(row_scale.begin(), row_scale.end(), row_scale.begin(), inverse_sqrt);
    std::transform(column_scale.begin(), column_scale.end(), column_scale.begin(), inverse_sqrt);

    std::vector<double> column_sums(matrix.n(), 0.);
    for (std::size_t row = 0; row < matrix.m(); ++row)
      for (auto entry = matrix.begin(row); entry != matrix.end(row); ++entry)
        column_sums[entry->column()] +=
            std::abs(row_scale[row] * static_cast<double>(entry->value()) * column_scale[entry->column()]);
    scaled_one_norm = column_sums.empty() ? 0. : *std::max_element(column_sums.begin(), column_sums.end());
  }
} // namespace DiFfRG::internal
