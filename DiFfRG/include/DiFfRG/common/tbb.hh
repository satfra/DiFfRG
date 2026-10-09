#pragma once

#include <DiFfRG/common/kokkos.hh>

// kept: several headers rely on this pulling in tbb::parallel_for as well.
#include <tbb/tbb.h>

#include <cstddef>
#include <cstdlib>
#include <functional>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Number of grid points below which a reduction is carried out serially.
     *
     * Momentum grids in DiFfRG are small - a radial 1D integral typically has 32 Gauss-Legendre
     * nodes - and TBBReduction is almost always called from inside an already parallel region:
     * an assembler's parallel cell loop, or the outer tbb::parallel_for of
     * Integrator::map. Spawning a task tree for such a reduction is pure overhead; measured on a
     * 32-node O(N) kernel it costs a factor 16 when called serially and a factor 5 when nested.
     *
     * Deliberately a compile-time constant and *not* derived from the available concurrency: the
     * summation order must not depend on the thread count.
     */
    inline constexpr std::size_t reduction_serial_cutoff = 4096;

    /**
     * @brief Leaf size of the deterministic reduction tree.
     *
     * blocked_range + simple_partitioner splits at the midpoint until a chunk is <= grainsize, so
     * the shape of the tree - and hence the order of the floating point additions - is a pure
     * function of (number of points, grainsize).
     */
    inline constexpr std::size_t reduction_grainsize = 256;

    /// Turn a row-major flat index into a multi-index.
    template <int dim>
    void unflatten_index(std::size_t flat, const device::array<size_t, dim> &grid_size,
                         device::array<size_t, dim> &idx)
    {
      for (int d = dim - 1; d >= 0; --d) {
        idx[d] = flat % grid_size[d];
        flat /= grid_size[d];
      }
    }

    /// Advance a row-major multi-index by one (odometer increment).
    template <int dim> void advance_index(const device::array<size_t, dim> &grid_size, device::array<size_t, dim> &idx)
    {
      for (int d = dim - 1; d >= 0; --d) {
        if (++idx[d] < grid_size[d]) return;
        idx[d] = 0;
      }
    }

    /// Strictly row-major serial sum of functor over the flat index range [begin, end).
    template <int dim, typename NT, typename FUN>
    NT serial_sum(const device::array<size_t, dim> &grid_size, const FUN &functor, const std::size_t begin,
                  const std::size_t end)
    {
      device::array<size_t, dim> idx{{}};
      unflatten_index<dim>(begin, grid_size, idx);
      NT value(0);
      for (std::size_t flat = begin; flat < end; ++flat) {
        value += functor(idx);
        advance_index<dim>(grid_size, idx);
      }
      return value;
    }
  } // namespace internal

  /**
   * @brief Bitwise reproducible reduction of @p functor over a dim-dimensional index grid.
   *
   * The result depends only on grid_size and functor - never on the number of TBB worker threads,
   * on work stealing, or on whether the call is nested inside another parallel region. This is a
   * hard requirement: the flow kernels feed a stiff DAE solver, so a last-bit difference in the
   * residual changes the accepted step sequence and thus the whole trajectory.
   *
   * tbb::parallel_reduce must never be used here: its range splitting depends on which worker
   * happens to steal which subrange, which makes the summation order - and therefore the result -
   * vary from call to call.
   */
  template <int dim, typename NT, typename FUN>
  NT TBBReduction(const device::array<size_t, dim> &grid_size, const FUN &functor)
  {
    std::size_t total = 1;
    for (int d = 0; d < dim; ++d)
      total *= grid_size[d];
    if (total == 0) return NT(0);

    if (total <= internal::reduction_serial_cutoff)
      return internal::serial_sum<dim, NT, FUN>(grid_size, functor, 0, total);

    return tbb::parallel_deterministic_reduce(
        tbb::blocked_range<std::size_t>(0, total, internal::reduction_grainsize), NT(0),
        [&](const tbb::blocked_range<std::size_t> &r, NT value) -> NT {
          return value + internal::serial_sum<dim, NT, FUN>(grid_size, functor, r.begin(), r.end());
        },
        std::plus<NT>(), tbb::simple_partitioner());
  }
} // namespace DiFfRG
