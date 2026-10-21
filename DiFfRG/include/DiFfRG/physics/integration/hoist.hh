#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/physics/interpolation/data_epoch.hh>

// std
#include <array>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

/**
 * @file hoist.hh
 * @brief Kernels in hoisted form (kernel_has_pre) and the pre() table the quadrature integrators keep for them.
 */

namespace DiFfRG
{
  /**
   * @brief Whether `K` provides its integrand in hoisted form.
   *
   * Such a kernel declares `static constexpr bool has_pre = true`, a `static constexpr std::array<bool, N> pre_reads`
   * with one flag per kernel() parameter (in kernel()'s order: box variables, frequency for a finite-temperature
   * kernel, external coordinates, then the rest), and
   * - `pre(a...)`, taking kernel()'s parameters and returning (as a trivially copyable value, e.g. a device::tuple)
   *   the part of the integrand that depends only on the parameters flagged in `pre_reads`;
   * - `post(P, a...)` with `P = pre(a...)`, which returns `kernel(a...)`.
   *
   * The integrator then evaluates pre() once per node of the first box variable (QuadratureIntegrator) or per
   * (first box node, frequency node) (QuadratureIntegrator_fT), and only post() at every quadrature point. That gives
   * the same result with less work whenever the integrand has a part that depends on the radial loop momentum (and
   * the frequency) alone. It does so only where pre() reads no box variable but the first, no external coordinate
   * (map()) and no argument given per point (map_points()); otherwise it uses kernel().
   */
  template <class K>
  inline constexpr bool kernel_has_pre = requires {
    requires K::has_pre;
    K::pre_reads.size();
  };

  namespace internal
  {
    /// Whether the pre() of `K` reads kernel() parameter @p i; true for an index beyond `K::pre_reads`.
    template <class K> constexpr bool pre_reads(const size_t i)
    {
      if constexpr (kernel_has_pre<K>)
        return i >= K::pre_reads.size() || K::pre_reads[i];
      else
        return true;
    }

    /// Whether the pre() of `K` reads any of the kernel() parameters [begin, end).
    template <class K> constexpr bool pre_reads_any(const size_t begin, const size_t end)
    {
      for (size_t i = begin; i < end; ++i)
        if (pre_reads<K>(i)) return true;
      return false;
    }

    /// The `pre_reads` mask of `K`, or an empty one; for adapters that forward a kernel's hoisted form.
    template <class K> constexpr auto pre_reads_of()
    {
      if constexpr (kernel_has_pre<K>)
        return K::pre_reads;
      else
        return std::array<bool, 0>{};
    }

    /**
     * @brief Append an identity of @p v to @p key; false if @p v has none, in which case nothing built from it is
     * reused.
     *
     * An interpolator is identified by its data pointer and grid; changes of its data are tracked by the interpolator
     * data epoch (data_epoch.hh), which PreTable::key adds. Any other trivially copyable value by its bytes, which is
     * conservative: padding may make two equal values look different, never two different values look equal.
     */
    template <typename T> bool append_identity(std::string &key, const T &v)
    {
      if constexpr (requires {
                      v.data();
                      v.get_coordinates().to_string();
                    }) {
        const void *p = v.data();
        key.append(reinterpret_cast<const char *>(&p), sizeof(p));
        key += v.get_coordinates().to_string();
        return true;
      } else if constexpr (std::is_trivially_copyable_v<T>) {
        key.append(reinterpret_cast<const char *>(&v), sizeof(T));
        return true;
      } else
        return false;
    }

    /// Append the bytes of @p v to @p key.
    template <typename T> void append_bytes(std::string &key, const T &v)
    {
      key.append(reinterpret_cast<const char *>(&v), sizeof(v));
    }

    /**
     * @brief The pre() table of a hoisted kernel on @p ExecutionSpace: device storage, what it was computed from, and
     * the stream it was filled on.
     *
     * A table is reused while its key is unchanged (see key()); an empty key means "cannot tell" and always rebuilds.
     */
    template <typename ExecutionSpace> class PreTable
    {
    public:
      /// Storage for @p n_entries entries of type PT; forgets the table if it had to grow.
      template <typename PT> PT *storage(const size_t n_entries) const
      {
        static_assert(std::is_trivially_copyable_v<PT>, "pre() must return a trivially copyable value.");
        const size_t bytes = n_entries * sizeof(PT);
        if (m_data.extent(0) < bytes) {
          m_data = Kokkos::View<char *, typename ExecutionSpace::memory_space>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, "pre_table"), bytes);
          m_key.clear();
        }
        return reinterpret_cast<PT *>(m_data.data());
      }

      /**
       * @brief What a table with entries PT of kernel `K` is computed from: the entry type, the interpolator data
       * epoch, whatever @p add_axes appends (the integrator's node sets) and the kernel() parameters @p rest that
       * pre() reads, the first being parameter @p first. Empty if one of them has no identity or reuse is off.
       */
      template <class K, typename PT, typename AddAxes, typename... A>
      std::string key(const AddAxes &add_axes, const size_t first, const A &...rest) const
      {
        if (!m_reuse) return {};
        std::string key = typeid(PT).name();
        append_bytes(key, current_interpolator_data_epoch());
        add_axes(key);
        bool known = true;
        size_t i = first;
        ((known = known && (!pre_reads<K>(i) || append_identity(key, rest)), ++i), ...);
        return known ? key : std::string();
      }

      /**
       * @brief Make the table current for @p key: call @p fill (which launches on @p space) if it is stale, otherwise
       * wait for the earlier fill if that ran on another stream than @p space. @p own is the integrator's own stream.
       */
      template <typename Fill>
      void refresh(std::string key, const ExecutionSpace &space, const ExecutionSpace &own, const Fill &fill) const
      {
        if (key.empty() || key != m_key) {
          fill();
          m_key = std::move(key);
          m_fill_space = space;
          m_fill_elsewhere = &space != &own;
        } else if (m_fill_elsewhere || &space != &own)
          m_fill_space.fence();
      }

      /// Turn reuse across calls on or off (default on).
      void set_reuse(const bool reuse)
      {
        m_reuse = reuse;
        m_key.clear();
      }

    private:
      mutable Kokkos::View<char *, typename ExecutionSpace::memory_space> m_data;
      mutable std::string m_key;
      mutable ExecutionSpace m_fill_space;
      mutable bool m_fill_elsewhere = false;
      bool m_reuse = true;
    };
  } // namespace internal
} // namespace DiFfRG
