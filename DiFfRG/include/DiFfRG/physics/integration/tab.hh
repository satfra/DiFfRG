#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>

// std
#include <array>
#include <cstddef>
#include <cstdlib>
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
   * - `kernel_tab(row, stride, a...)` (and `post_tab(P, row, stride, a...)` with kernel_has_pre), returning kernel(a...)
   *   from the node's entries `row[e * stride]`, of the integrator's ctype.
   *
   * The integrator builds the table on the first map() and keeps it while the grids of the masked variables are
   * unchanged; it never depends on the arguments, on k or on the interpolators.
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
    /// The table of a kernel_has_tab kernel and the maps from the integrator's indices into it.
    ///
    /// Entry e of the node (box node b, shape s) sits at `data[(b * tab_size + e) * n_shape + s]`: the shapes (the
    /// external points restricted to tab_ext_mask) are fastest because Kokkos walks the external index fastest within
    /// a GPU warp, so a warp's loads of one entry are contiguous.
    template <typename ctype, typename ExecutionSpace> struct TabTable {
      using memory_space = typename ExecutionSpace::memory_space;
      Kokkos::View<ctype *, memory_space> data;
      Kokkos::View<unsigned *, memory_space> shape_of; ///< external linear index -> shape
      std::size_t n_shape = 0, n_box = 0;
      std::string key;

      /// Tables are used unless DIFFRG_NO_TRACE_TABLES is set (an A/B switch needing no code change).
      static bool env_enabled()
      {
        static const bool on = std::getenv("DIFFRG_NO_TRACE_TABLES") == nullptr;
        return on;
      }
    };

    /// The `tab_box_mask` / `tab_ext_mask` of `K`, or empty ones; for adapters that forward a kernel's table form.
    template <class K> constexpr auto tab_box_mask_of()
    {
      if constexpr (kernel_has_tab<K>)
        return K::tab_box_mask;
      else
        return std::array<bool, 0>{};
    }
    template <class K> constexpr auto tab_ext_mask_of()
    {
      if constexpr (kernel_has_tab<K>)
        return K::tab_ext_mask;
      else
        return std::array<bool, 0>{};
    }
    template <class K> constexpr std::size_t tab_size_of()
    {
      if constexpr (kernel_has_tab<K>)
        return K::tab_size;
      else
        return 0;
    }
  } // namespace internal
} // namespace DiFfRG
