#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>

// std
#include <array>
#include <cstddef>
#include <string>

/**
 * @file tab.hh
 * @brief Kernels with a per-run table (kernel_has_tab), and the table the quadrature integrators keep for them.
 *
 * A NumTracer trace table (MakeNTKernelDiFfRG with "TraceTables"): every trace of the integrand is a short polynomial
 * in the momentum scales (the radial loop variable and the external scale) whose coefficients depend only on the
 * dimensionless kinematics. Those coefficients are the table: computed once per run at the nodes of the variables they
 * depend on, and read by the kernel instead of evaluating the traces.
 */

namespace DiFfRG
{
  /**
   * @brief Whether `K` provides its integrand with a per-run table.
   *
   * Such a kernel declares
   * - `static constexpr bool has_tab` (true) and `static constexpr std::size_t tab_size`, the entries per table node;
   * - `static constexpr std::array<bool, dim> tab_box_mask` and `std::array<bool, cdim> tab_ext_mask`: which
   *   integration variables and which external coordinates the table depends on. The table has one node per
   *   combination of their grid points;
   * - `tab(double *out, std::size_t stride, a...)`, device-callable and taking kernel()'s parameters, which writes the
   *   node's entries to `out[e * stride]`. It may read only the masked variables: the integrator calls it once per
   *   run, with the other parameters at arbitrary values;
   * - `kernel_tab(row, stride, a...)` (and `post_tab(P, row, stride, a...)` with kernel_has_pre), returning
   *   kernel(a...) from the node's entries `row[e * stride]`, of the integrator's ctype.
   *
   * The external coordinates are kernel()'s parameters right after the integration variables: the grid point in
   * map(), the first `cdim` arguments in map_points() and get(). The integrator builds a table on first use and keeps
   * it while the grids of the masked variables and the masked external values are unchanged; it never depends on the
   * other arguments, on k or on the interpolators. get() reads a table only if tab_ext_mask is all false.
   */
  template <class K>
  inline constexpr bool kernel_has_tab = requires {
    requires K::has_tab;
    K::tab_size;
    K::tab_box_mask.size();
    K::tab_ext_mask.size();
  };

  namespace internal
  {
    /**
     * @brief The table of a kernel_has_tab kernel and the maps from the integrator's indices into it.
     *
     * A shape is an external point restricted to tab_ext_mask; a box node a node of the masked integration variables.
     * map() stores entry e of (box node b, shape s) at `data[(b * tab_size + e) * n_shape + s]`, shapes fastest, since
     * Kokkos walks the external index fastest within a GPU warp. map_points() and get() store it at
     * `data[(s * tab_size + e) * n_box + b]`, box nodes fastest, since there a team walks the nodes of one point.
     */
    template <typename ctype, typename ExecutionSpace> struct TabTable {
      using memory_space = typename ExecutionSpace::memory_space;
      /**
       * @brief The entries, `n_box * tab_size * n_shape` of them.
       */
      Kokkos::View<ctype *, memory_space> data;
      /**
       * @brief The shape of each external point, by its linear index (map()) or point index (map_points()).
       */
      Kokkos::View<unsigned *, memory_space> shape_of;
      /**
       * @brief The representative point of each shape, and the double-precision entries tab() writes (point tables).
       */
      Kokkos::View<unsigned *, memory_space> rep;
      Kokkos::View<double *, memory_space> fp64;
      /**
       * @brief Number of shapes.
       */
      std::size_t n_shape = 0;
      /**
       * @brief Number of box nodes.
       */
      std::size_t n_box = 0;
      /**
       * @brief The grids the table was built for; empty until it is built.
       */
      std::string key;
    };

    /**
     * @brief `K::tab_box_mask`, or an empty mask if `K` has no table. For adapters that forward a kernel's table form.
     */
    template <class K> constexpr auto tab_box_mask_of()
    {
      if constexpr (kernel_has_tab<K>)
        return K::tab_box_mask;
      else
        return std::array<bool, 0>{};
    }
    /**
     * @brief `K::tab_ext_mask`, or an empty mask if `K` has no table.
     */
    template <class K> constexpr auto tab_ext_mask_of()
    {
      if constexpr (kernel_has_tab<K>)
        return K::tab_ext_mask;
      else
        return std::array<bool, 0>{};
    }
    /**
     * @brief `K::tab_size`, or 0 if `K` has no table.
     */
    template <class K> constexpr std::size_t tab_size_of()
    {
      if constexpr (kernel_has_tab<K>)
        return K::tab_size;
      else
        return 0;
    }
  } // namespace internal
} // namespace DiFfRG
