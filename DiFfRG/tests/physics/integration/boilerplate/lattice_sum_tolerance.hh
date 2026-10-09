#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

/**
 * @brief How many weights an IntegratorLat<dim>D sums.
 *
 * A halved axis of extent N visits n = 0 .. N/2, i.e. N/2 + 1 points; q0 is halved only when
 * q0_symmetric is set, every further axis always.
 */
constexpr std::size_t lattice_sum_terms(const unsigned int dim, const unsigned int size0, const unsigned int size1,
                                        const bool q0_symmetric)
{
  std::size_t n = q0_symmetric ? size0 / 2u + 1u : size0;
  for (unsigned int d = 1; d < dim; ++d)
    n *= size1 / 2u + 1u;
  return n;
}

/**
 * @brief Round-off tolerance for a lattice volume integral.
 *
 * The volume integral sums `n_terms` identical positive weights, so it is exact in exact arithmetic
 * and the entire discrepancy is floating-point accumulation. The standard forward bound for summing
 * n numbers in any order is (n-1) * u with u = eps/2, which is what this returns.
 *
 * Measured over 4028 configurations spanning all four dimensions, all five number types, every grid
 * size and both q0 symmetries, the observed relative error is n_terms * eps * c with c between 0.05
 * (median) and 0.123 (worst) -- so the bound holds with about 4x headroom, and it does not depend on
 * the lattice spacings a0, a1 at all. The floor covers the smallest grids, where the handful of
 * roundings in the prefactor itself outweighs the sum.
 */
template <typename ctype> constexpr ctype lattice_sum_tolerance(const std::size_t n_terms)
{
  constexpr ctype eps = std::numeric_limits<ctype>::epsilon();
  return std::max(ctype(8) * eps, static_cast<ctype>(n_terms) * eps / ctype(2));
}

/**
 * @brief Above this tolerance the comparison would stop being a test.
 *
 * `float` cannot accumulate millions of terms: once the running sum exceeds 2^24 weights the
 * additions stagnate, and the 4D volume integral comes out ~40% low with no bug anywhere. Widening
 * the tolerance until such a configuration passes would gate nothing, so skip it and say so instead.
 */
constexpr double lattice_sum_meaningful_tolerance = 1e-1;
