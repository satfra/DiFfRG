#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/mpi.hh>
#include <DiFfRG/common/quadrature/quadrature_provider.hh>
#include <DiFfRG/common/tbb.hh>
#include <DiFfRG/common/tuples.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/coordinates/coordinates.hh>
#include <DiFfRG/physics/integration/abstract_integrator.hh>
#include <DiFfRG/physics/integration/map_completion.hh>
#include <DiFfRG/physics/integration/map_distribution.hh>
// for has_cacheable_positions_v, shared with the vacuum integrator
#include <DiFfRG/physics/integration/quadrature_integrator.hh>
#include <DiFfRG/physics/interpolation/data_epoch.hh>

// std
#include <cstdio>
#include <string>
#include <typeinfo>

namespace DiFfRG
{
  /**
   * @brief Whether `K` declares `static constexpr bool matsubara_even = true`.
   *
   * Promises that the kernel is even in the frequency \f$q_0\f$. The integrator then evaluates it once per
   * frequency pair \f$\pm q_0\f$ instead of twice. Without the trait both signs are evaluated, which is always
   * correct.
   */
  template <class K> inline constexpr bool kernel_is_matsubara_even = requires { requires K::matsubara_even; };

  /**
   * @brief Whether `K` declares `static constexpr bool matsubara_finite_extent = true`.
   *
   * Promises that the kernel vanishes for \f$|q_0|\f$ above the frequency cutoff (see
   * QuadratureIntegrator_fT::set_frequency_cutoff), e.g. because every term carries a \f$\partial_t R\f$ of a
   * regulator with compact support. The integrator may then sum the finitely many Matsubara modes exactly when that
   * is cheaper than the approximate rule. Without the trait the approximate rule is used, which is always correct.
   */
  template <class K>
  inline constexpr bool kernel_has_finite_matsubara_extent = requires { requires K::matsubara_finite_extent; };

  /**
   * @brief Whether `K` declares `static constexpr bool matsubara_split = true`.
   *
   * Such a kernel also provides `kernel_finite_extent` and `kernel_tail`, with the same signature as `kernel` and
   * summing to it. `kernel_finite_extent` must vanish above the frequency cutoff (as for
   * kernel_has_finite_matsubara_extent); `kernel_tail` is everything else. The integrator sums the first exactly
   * where that is cheaper and the second with the approximate rule, in one launch.
   */
  // Where the split pays: a term's cost is dominated by the trace it calls and the generator's CSE
  // is per function, so each half computes only its own traces. On QCD_Nf2's ZA4 the one unbounded
  // term carries a 313-line trace and the five confined ones 2918 + 609 + ... lines, so the cheap
  // half runs the full Gaussian rule and the expensive half a handful of exact modes.
  template <class K> inline constexpr bool kernel_has_matsubara_split = requires { requires K::matsubara_split; };

  // DIFFRG_MATSUBARA_SIGN_LOOP (compile definition, default off): evaluate the pair f(+q0) + f(-q0) of every
  // frequency node as a two-iteration loop instead of two inlined copies of the integrand. Halves the code of the
  // map/get kernels (measured on QuarkMeson full_p: same results, -0.3..-2 % time on an RTX 4070). Meant for GPUs
  // whose large kernels stall on instruction fetch, as measured on A100; to be verified there before it becomes the
  // default.

  /**
   * @brief Whether `K` provides its integrand in hoisted form.
   *
   * Such a kernel declares `static constexpr bool has_pre = true`, a `static constexpr std::array<bool, N> pre_reads`
   * with one flag per kernel() parameter (in kernel()'s order: box variables, frequency, then the rest), and
   * - `pre(a...)`, taking kernel()'s parameters and returning (as a trivially copyable value, e.g. a device::tuple)
   *   the part of the integrand that depends only on the parameters flagged in `pre_reads`;
   * - `post(P, a...)` with `P = pre(a...)`, which returns `kernel(a...)`.
   *
   * QuadratureIntegrator_fT then evaluates pre() once per (first box node, frequency node) and only post() at every
   * quadrature point, which gives the same result with less work whenever the integrand has a part that depends on
   * the radial loop momentum and the frequency alone. It does so only where pre() reads no box variable but the
   * first, no external coordinate (map()) and no argument given per point (map_points()); otherwise it uses
   * kernel(). Not combined with kernel_has_matsubara_split.
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

    /**
     * @brief Append an identity of @p v to @p key; false if @p v has none, in which case nothing built from it is
     * reused.
     *
     * An interpolator is identified by its data pointer and grid; changes of its data are tracked by the interpolator
     * data epoch (data_epoch.hh), which the caller adds. Any other trivially copyable value by its bytes, which is
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
  } // namespace internal

  /**
   * @brief Like QuadratureIntegrator, with the last integration variable replaced by a sum over bosonic Matsubara
   * frequencies.
   *
   * \f[
   *   I(\text{args}) = \texttt{constant}(\text{args}) + \sum_{\text{box}} w\, \Delta \;
   *     T \sum_{n \in \mathbb{Z}} \text{kernel}(x_1, \ldots, x_\text{sdim}, 2\pi n T, \text{args}) \,,
   * \f]
   * where the first `sdim` = `dim - 1` variables are integrated over a box exactly as in QuadratureIntegrator, and
   * \f$T\f$ is the temperature. The frequency is passed after the box variables: `kernel(x..., q0, args...)` for
   * get(), `kernel(x..., q0, p..., args...)` for map() with the external point \f$p\f$. `constant` is as in
   * QuadratureIntegrator.
   *
   * Only bosonic frequencies \f$2\pi n T\f$ are summed; a fermionic loop must shift \f$q_0\f$ by \f$\pi T\f$
   * inside the kernel. The sum is approximated by a Gaussian rule for sums sized from the scale of the summand
   * (\f$k\f$, or set_typical_E()). At \f$T = 0\f$, or when that rule would need more than
   * `/integration/max_matsubara_size` nodes, the sum is replaced by the integral \f$\int dq_0/(2\pi)\f$, its
   * \f$T \to 0\f$ limit. For kernels with the matsubara_finite_extent or matsubara_split traits the modes below the
   * frequency cutoff may instead be summed exactly.
   *
   * @tparam dim number of box variables plus one for the frequency
   * @tparam NT type of the result
   * @tparam KERNEL kernel class, see QuadratureIntegrator
   * @tparam ExecutionSpace where the integrand is evaluated: GPU_exec, TBB_exec or KokkosHost_exec
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim > 0)
  class QuadratureIntegrator_fT : public AbstractIntegrator
  {
  public:
    /// Type of the integration variables: `NT` with complex and autodiff parts stripped, e.g. `double`.
    using ctype = typename get_type::ctype<NT>;
    /// The execution space the integrand is evaluated in.
    using execution_space = ExecutionSpace;

    /// Number of integration variables besides the frequency (not necessarily the physical spatial dimension).
    static constexpr int sdim = dim - 1;

    /**
     * @param quadrature_provider source of quadrature nodes and weights, shared between integrators
     * @param _grid_size number of quadrature points per box axis
     * @param grid_min lower end of each box axis
     * @param grid_max upper end of each box axis
     * @param quadrature_type quadrature rule per box axis
     * @param T temperature
     * @param k initial RG scale, see set_k()
     */
    QuadratureIntegrator_fT(QuadratureProvider &quadrature_provider, const std::array<size_t, sdim> _grid_size,
                            std::array<ctype, sdim> grid_min, std::array<ctype, sdim> grid_max,
                            const std::array<QuadratureType, sdim> quadrature_type, const ctype T = 1,
                            const ctype k = 1)
        : space(quadrature_provider.template next_execution_space<ExecutionSpace>()),
          quadrature_provider(quadrature_provider), T(T), m_k(k),
          // A SPLIT kernel does not carry `matsubara_finite_extent` -- only its finite-extent HALF
          // has that property -- so it has to opt in here too, or its expensive half would keep
          // running the Gaussian rule and the split would be pure overhead.
          m_allow_exact(kernel_has_finite_matsubara_extent<KERNEL> || kernel_has_matsubara_split<KERNEL>)
    {
      for (int d = 0; d < sdim; ++d)
        grid_size[d] = _grid_size[d];
      for (int i = 0; i < sdim; ++i) {
        nodes[i] = quadrature_provider.template nodes<ctype, typename ExecutionSpace::memory_space>(grid_size[i],
                                                                                                    quadrature_type[i]);
        weights[i] = quadrature_provider.template weights<ctype, typename ExecutionSpace::memory_space>(
            grid_size[i], quadrature_type[i]);
      }
      set_grid_extents(grid_min, grid_max);
      refresh_matsubara();
    }

    /// Change the box, keeping the quadrature rules; see QuadratureIntegrator::set_grid_extents.
    void set_grid_extents(const std::array<ctype, sdim> grid_min, const std::array<ctype, sdim> grid_max)
    {
      for (int d = 0; d < sdim; ++d) {
        grid_extents[0][d] = grid_min[d];
        grid_extents[1][d] = grid_max[d];
      }
      for (int i = 0; i < sdim; ++i) {
        grid_start[i] = grid_extents[0][i];
        grid_scale[i] = (grid_extents[1][i] - grid_extents[0][i]);
      }
    }

    /// Set the temperature and rebuild the frequency rule.
    void set_T(const ctype T)
    {
      this->T = T;
      refresh_matsubara();
    }

    /**
     * @brief Set the RG scale \f$k\f$, the scale the frequency rule is sized for unless set_typical_E() was given one.
     *
     * Call it every RG step (all_set_k() does). Changes below a relative \f$10^{-6}\f$ are ignored.
     */
    void set_k(const ctype k)
    {
      if (is_close(m_k, k, 1e-6 * std::fabs(k))) return;
      m_k = k;
      refresh_matsubara();
    }

    /**
     * @brief Set the energy scale the frequency rule is sized for, replacing \f$k\f$. Zero (default) means \f$k\f$.
     *
     * It decides how far the Matsubara rule must reach and where the \f$T = 0\f$ integral places its nodes, and so
     * also whether the sum or the integral is used.
     *
     * Report the scale whose thermal content matters, not simply the largest one present: a
     * summand carrying a scale below `typical_E` has its thermal content silently discarded when
     * the rule hands over to the integral, and nothing here can know it was there.
     *
     * Set it from replicated or all-reduced data only. It sizes the frequency rule, so a value
     * derived from rank-local data on a distributed grid makes the node count differ between
     * ranks and the result stop being reproducible across decompositions.
     */
    void set_typical_E(const ctype typical_E)
    {
      // Relative to the quantity compared, and matched to the tolerance the provider's cache uses
      // to decide two rules are the same.
      if (is_close(m_typical_E_user, typical_E, 1e-6 * std::fabs(typical_E))) return;

      m_typical_E_user = typical_E;
      refresh_matsubara();
    }

    /**
     * @brief Set the frequency \f$\Lambda_0\f$ above which the summand vanishes, enabling the exact sum.
     *
     * Only used for kernels with the matsubara_finite_extent or matsubara_split trait. For a 4D regulator cut off at
     * \f$q_0^2 + \vec q^{\,2} \le x_\text{extent}\, k^2\f$ this is \f$\Lambda_0 = \sqrt{x_\text{extent}}\, k\f$, the
     * same value as the spatial cutoff; the wrappers set it that way.
     *
     * Passing zero (the default) disables the exact sum and keeps the approximate rule.
     */
    void set_frequency_cutoff(const ctype freq_cutoff)
    {
      if (is_close(m_freq_cutoff, freq_cutoff, 1e-10 * std::fabs(freq_cutoff))) return;
      m_freq_cutoff = freq_cutoff;
      refresh_matsubara();
    }

    /**
     * @brief Allow (or forbid) the exact sum regardless of the kernel's traits.
     *
     * The default follows the traits. Allowing it for a kernel that does not vanish above the frequency cutoff
     * silently truncates the sum.
     */
    void set_allow_exact_matsubara_sum(const bool allow)
    {
      if (m_allow_exact == allow) return;
      m_allow_exact = allow;
      refresh_matsubara();
    }

    /**
     * @brief Sum exactly up to `margin` times the frequency cutoff, plus one mode \f$2\pi T\f$. Default 1.
     *
     * Values \f$\le 0\f$ are ignored.
     */
    void set_matsubara_extent_margin(const ctype margin)
    {
      if (!(margin > ctype(0)) || is_close(m_extent_margin, margin)) return;
      m_extent_margin = margin;
      refresh_matsubara();
    }

    /// Nodes on the frequency axis, INCLUDING the zero mode.
    size_t get_matsubara_size() const { return matsubara_nodes.size(); }

    /// True if the frequency axis is currently the exact sum rather than a Gaussian quadrature rule.
    bool uses_exact_matsubara_sum() const { return m_using_exact; }

    /**
     * @brief One Matsubara node's contribution at one spatial point, weight excluded.
     *
     * Factored out of get()/map() so the +-frequency and zero-mode logic exists once, and --
     * importantly -- so the `if constexpr` on matsubara_even lives in an ordinary __device__
     * function rather than inside an extended lambda, where nvcc's transformation is fragile.
     *
     * The zero mode is NOT special-cased here: it arrives as an ordinary node `(0, T/2)` at the
     * end of the node list (see MatsubaraQuadrature::sum_nodes), and `w * (f(+0) + f(-0))`
     * reproduces `T f(0)` exactly. The branch this replaces cost the whole warp a full extra
     * kernel evaluation for one active lane.
     *
     * The three packs are applied as NESTED lvalue tuples. Do not tuple_cat them: tuple_cat builds
     * a by-value object, and `args` holds every interpolator by value, so the concatenated tuple
     * becomes a full per-thread copy of all of them in local memory. On QCD_Nf2 that was 3576 B of
     * stack frame and 2.3x of runtime in the vacuum integrator (see the comment in
     * QuadratureIntegrator::map). Applying an lvalue tuple binds references instead.
     */
    template <typename XArr, typename PosArr, typename ArgTuple>
    KOKKOS_INLINE_FUNCTION static NT node_value(const XArr &x, const PosArr &pos, const ArgTuple &args, const ctype xt,
                                                const ctype wt, const bool is_tail)
    {
      NT out{};
      device::apply(
          [&](const auto &...xargs) {
            device::apply(
                [&](const auto &...pargs) {
                  device::apply(
                      [&](const auto &...iargs) {
                        // Six near-identical lines rather than the obvious factoring of the
                        // +-frequency rule into a lambda taking the entry point. That factoring was
                        // written and reverted: expanding an OUTER generic lambda's parameter pack
                        // inside a nested lambda makes nvcc drop the pack, and it fails as "too few
                        // arguments" only for the instantiation where the outer packs are EMPTY
                        // (dim == 1, no spatial variables) -- i.e. it would have compiled here and
                        // broken a 1-D flow somewhere else.
                        NT msum;
                        if constexpr (kernel_has_matsubara_split<KERNEL>) {
                          // Runtime branch on the node index, not on data: within a frequency row
                          // there is exactly ONE boundary, so at most one warp per row straddles it
                          // and runs both bodies -- which together cost what the unsplit kernel
                          // costs today. The split can therefore not be slower than not splitting.
                          if (is_tail) {
                            if constexpr (kernel_is_matsubara_even<KERNEL>)
                              msum = ctype(2) * KERNEL::kernel_tail(xargs..., xt, pargs..., iargs...);
                            else
                              msum = KERNEL::kernel_tail(xargs..., xt, pargs..., iargs...) +
                                     KERNEL::kernel_tail(xargs..., -xt, pargs..., iargs...);
                          } else {
                            if constexpr (kernel_is_matsubara_even<KERNEL>)
                              msum = ctype(2) * KERNEL::kernel_finite_extent(xargs..., xt, pargs..., iargs...);
                            else
                              msum = KERNEL::kernel_finite_extent(xargs..., xt, pargs..., iargs...) +
                                     KERNEL::kernel_finite_extent(xargs..., -xt, pargs..., iargs...);
                          }
                        } else {
                          if constexpr (kernel_is_matsubara_even<KERNEL>)
                            // even kernel: kernel(+xt)+kernel(-xt) == 2*kernel(xt) (one evaluation)
                            msum = ctype(2) * KERNEL::kernel(xargs..., xt, pargs..., iargs...);
                          else {
                            // positive and negative Matsubara frequencies
#ifdef DIFFRG_MATSUBARA_SIGN_LOOP
                            msum = NT{};
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
                            for (int s = 0; s < 2; ++s)
                              msum += KERNEL::kernel(xargs..., s == 0 ? xt : -xt, pargs..., iargs...);
#else
                            msum = KERNEL::kernel(xargs..., xt, pargs..., iargs...) +
                                   KERNEL::kernel(xargs..., -xt, pargs..., iargs...);
#endif
                          }
                        }
                        out = wt * msum;
                      },
                      args);
                },
                pos);
          },
          x);
      return out;
    }

    // ---- Hoisting (kernel_has_pre) ---------------------------------------------------------------
    // The pre() table holds, per (first box node, frequency node), pre() at +q0 and at -q0: entry 2e and 2e + 1 for
    // e = first box index * n_frequencies + frequency index.

    /// Whether this kernel can be hoisted at all: pre() reads no box variable but the first.
    static constexpr bool can_hoist =
        kernel_has_pre<KERNEL> && !kernel_has_matsubara_split<KERNEL> && !internal::pre_reads_any<KERNEL>(1, sdim);

    /// pre() at box point @p x and frequency @p q0; @p pos and @p args as in node_value().
    template <typename XArr, typename PosArr, typename ArgTuple>
    KOKKOS_INLINE_FUNCTION static auto pre_value(const XArr &x, const ctype q0, const PosArr &pos, const ArgTuple &args)
    {
      return device::apply(
          [&](const auto &...xargs) {
            return device::apply(
                [&](const auto &...pargs) {
                  return device::apply(
                      [&](const auto &...iargs) { return KERNEL::pre(xargs..., q0, pargs..., iargs...); }, args);
                },
                pos);
          },
          x);
    }

    /// pre_value() with the arguments of map_points() point 0, whose per-point values pre() does not read.
    template <typename XArr, typename DeviceArgs>
    KOKKOS_INLINE_FUNCTION static auto pre_value_points(const XArr &x, const ctype q0, const DeviceArgs &device_args)
    {
      return device::apply([&](const auto &...a) { return pre_value(x, q0, device::tuple<>{}, device::tie(a(0)...)); },
                           device_args);
    }

    /// As node_value(), from the pre() table entries @p entry[0] (at +xt) and @p entry[1] (at -xt).
    template <typename PT, typename XArr, typename PosArr, typename ArgTuple>
    KOKKOS_INLINE_FUNCTION static NT hoisted_node_value(const PT *entry, const XArr &x, const PosArr &pos,
                                                        const ArgTuple &args, const ctype xt, const ctype wt)
    {
      NT out{};
      device::apply(
          [&](const auto &...xargs) {
            device::apply(
                [&](const auto &...pargs) {
                  device::apply(
                      [&](const auto &...iargs) {
                        NT msum;
                        if constexpr (kernel_is_matsubara_even<KERNEL>)
                          msum = ctype(2) * KERNEL::post(entry[0], xargs..., xt, pargs..., iargs...);
                        else {
#ifdef DIFFRG_MATSUBARA_SIGN_LOOP
                          msum = NT{};
#ifdef __CUDA_ARCH__
#pragma unroll 1
#endif
                          for (int s = 0; s < 2; ++s)
                            msum += KERNEL::post(entry[s], xargs..., s == 0 ? xt : -xt, pargs..., iargs...);
#else
                          msum = KERNEL::post(entry[0], xargs..., xt, pargs..., iargs...) +
                                 KERNEL::post(entry[1], xargs..., -xt, pargs..., iargs...);
#endif
                        }
                        out = wt * msum;
                      },
                      args);
                },
                pos);
          },
          x);
      return out;
    }

    /// Box point with the first variable at node @p i0 and all others zero (pre() reads only the first).
    template <typename NodeArr, typename StartArr>
    KOKKOS_INLINE_FUNCTION static device::array<ctype, sdim>
    first_box_point(const size_t i0, const NodeArr &n, const StartArr &start, const StartArr &scale)
    {
      device::array<ctype, sdim> x{};
      if constexpr (sdim > 0) x[0] = Kokkos::fma(scale[0], n[0][i0], start[0]);
      return x;
    }

    /**
     * @brief Turn reuse of the pre() table across calls on or off (default on); see kernel_has_pre.
     *
     * A table is reused while the frequency rule, the first box axis, every argument pre() reads and the data of
     * every interpolator (data_epoch.hh) are unchanged; otherwise it is rebuilt.
     */
    void set_pre_table_reuse(const bool reuse)
    {
      m_pre_reuse = reuse;
      m_pre_key.clear();
    }

    /// Fill the pre() table on @p space. Public only because nvcc rejects extended device lambdas inside non-public
    /// member functions.
    template <typename PT, typename PosArr, typename ArgTuple>
    void fill_pre_table(ExecutionSpace &space, PT *table, const PosArr &pos, const ArgTuple &args) const
    {
      const auto &n = nodes;
      const auto &m_n = matsubara_nodes;
      const auto &start = grid_start;
      const auto &scale = grid_scale;
      const size_t n_m = grid_size[dim - 1];
      Kokkos::parallel_for(
          "QuadratureIntegrator_fT_pre_table", Kokkos::RangePolicy<ExecutionSpace>(space, 0, pre_table_entries()),
          KOKKOS_LAMBDA(const size_t e) {
            const auto x = first_box_point(e / n_m, n, start, scale);
            const ctype q0 = m_n[e % n_m];
            table[2 * e] = pre_value(x, q0, pos, args);
            table[2 * e + 1] = pre_value(x, -q0, pos, args);
          });
    }

    /// fill_pre_table() for map_points(). Public for the same reason.
    template <typename PT, typename DeviceArgs> void fill_pre_table_points(PT *table, const DeviceArgs &device_args)
    {
      const auto &n = nodes;
      const auto &m_n = matsubara_nodes;
      const auto &start = grid_start;
      const auto &scale = grid_scale;
      const size_t n_m = grid_size[dim - 1];
      Kokkos::parallel_for(
          "QuadratureIntegrator_fT_pre_table", Kokkos::RangePolicy<ExecutionSpace>(space, 0, pre_table_entries()),
          KOKKOS_LAMBDA(const size_t e) {
            const auto x = first_box_point(e / n_m, n, start, scale);
            const ctype q0 = m_n[e % n_m];
            table[2 * e] = pre_value_points(x, q0, device_args);
            table[2 * e + 1] = pre_value_points(x, -q0, device_args);
          });
    }

    /**
     * @brief Compute the integral for the extra kernel arguments @p t and write it to @p dest.
     *
     * Blocks until the result is available.
     */
    template <typename... T> void get(NT &dest, const T &...t) const
    {
      if (!m_result_views_initialized) {
        m_result_view = Kokkos::View<NT, typename ExecutionSpace::memory_space>("result");
        m_result_host = Kokkos::create_mirror_view(m_result_view);
        m_result_views_initialized = true;
      }
      get(space, m_result_view, t...);
      Kokkos::deep_copy(space, m_result_host, m_result_view);
      space.fence();
      dest = m_result_host();
    }

    /// Single-precision integration handing back a double result.
    template <typename OT, typename... T>
      requires internal::is_widened_result<OT, NT>
    void get(OT &dest, const T &...t) const
    {
      NT result;
      get(result, t...);
      dest = OT(result);
    }

    /// Compute the integral into a Kokkos reduction target @p dest (e.g. a rank-0 View) without waiting.
    template <typename OT, typename... T>
      requires(!std::is_same_v<OT, NT> && !internal::is_widened_result<OT, NT>)
    void get(OT &dest, const T &...t) const
    {
      get(space, dest, t...);
    }

    /// Launch the integral on @p space into a Kokkos reduction target @p dest without waiting.
    template <typename OT, typename... Args>
      requires(!std::is_same_v<OT, NT> && !internal::is_widened_result<OT, NT>)
    void get(ExecutionSpace &space, OT &dest, const Args &...t) const
    {
      constexpr bool handles = internal::takes_handles<NT, KERNEL, ctype, dim, Args...>;
      internal::check_kernel_precision<handles, ctype, Args...>();
      const auto args = device::make_tuple(internal::to_kernel_arg<handles, ctype>(t)...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      // Nodes before the boundary of the concatenated axis; 0 for an unsplit kernel, where the
      // is_tail flag is dead code that the `if constexpr` in node_value() removes anyway.
      const size_t n_tail = m_n_tail;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      if constexpr (can_hoist) {
        using PT = decltype(pre_value(device::array<ctype, sdim>{}, ctype(0), device::tuple<>{}, args));
        const PT *table = prepare_pre_table<PT>(space, device::tuple<>{}, args, sdim + 1, t...);
        const size_t n_m = grid_size[dim - 1];
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, dim> &idx, NT &update)
        {
          device::array<ctype, sdim> x;
          ctype weight = 1;
          bool is_first = true;
          for (int i = 0; i < sdim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
            weight *= w[i][idx[i]] * scale[i];
            is_first &= idx[i] == 0;
          }
          is_first &= idx[dim - 1] == 0;
          const size_t jt = idx[dim - 1];
          const size_t e = (sdim > 0 ? idx[0] : 0) * n_m + jt;
          update += weight * hoisted_node_value(table + 2 * e, x, device::tuple<>{}, args, m_n[jt], m_w[jt]);
          device::apply([&](const auto &...iargs) { update += is_first ? KERNEL::constant(iargs...) : NT(0); }, args);
        };
        Kokkos::parallel_reduce("QuadratureIntegral_fT_hoisted_" + std::to_string(dim) + "D",
                                make_kokkos_nd_range<dim, ExecutionSpace>(space, {0}, grid_size),
                                internal::KokkosNDLambdaWrapperReduction<dim, decltype(functor)>(functor), dest);
      } else {
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, dim> &idx, NT &update)
        {
          device::array<ctype, sdim> x;
          ctype weight = 1;
          bool is_first = true;
          for (int i = 0; i < sdim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
            weight *= w[i][idx[i]] * scale[i];
            is_first &= idx[i] == 0;
          }
          is_first &= idx[dim - 1] == 0;
          const size_t jt = idx[dim - 1];
          // Empty position pack: this overload's caller passes the external position inside `args`.
          update += weight * node_value(x, device::tuple<>{}, args, m_n[jt], m_w[jt], jt < n_tail);
          device::apply([&](const auto &...iargs) { update += is_first ? KERNEL::constant(iargs...) : NT(0); }, args);
        };

        Kokkos::parallel_reduce("QuadratureIntegral_fT_" + std::to_string(dim) + "D", // name of the kernel
                                make_kokkos_nd_range<dim, ExecutionSpace>(space, {0}, grid_size),
                                internal::KokkosNDLambdaWrapperReduction<dim, decltype(functor)>(functor), dest);
      }
    }

    /// Launch the integrals for all points of @p coordinates into a device view, without waiting or MPI
    /// distribution. `integral_view(i)` receives the result at `coordinates.forward(coordinates.from_linear_index(i))`.
    template <typename view_type, typename Coordinates, typename... Args>
    void map(ExecutionSpace &space, const view_type integral_view, const Coordinates &coordinates, const Args &...args)
    {
      device::array<size_t, 1 + dim> extents;
      extents[0] = integral_view.size();
      for (int i = 0; i < dim; ++i)
        extents[1 + i] = grid_size[i];

      // Reuse cached view if large enough, otherwise reallocate (grow-only)
      {
        bool needs_realloc = false;
        for (size_t i = 0; i < 1 + dim; ++i)
          needs_realloc |= (extents[i] > m_cache_extents[i]);
        if (needs_realloc) {
          for (size_t i = 0; i < 1 + dim; ++i)
            m_cache_extents[i] = std::max(m_cache_extents[i], extents[i]);
          m_cache = make_kokkos_nd_view<1 + dim, NT, ExecutionSpace>("cache", m_cache_extents);
        }
      }
      // Create a Restrict-tagged alias of the cache for no-alias optimization
      const auto cache = KokkosNDViewRestrict<1 + dim, NT, ExecutionSpace>(m_cache);

      constexpr bool handles = internal::map_takes_handles<NT, KERNEL, ctype, dim, Coordinates, Args...>;
      internal::check_kernel_precision<handles, ctype, Args...>();
      const auto m_args = device::make_tuple(internal::to_kernel_arg<handles, ctype>(args)...);

      const auto &n = nodes;
      const auto &w = weights;
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      // Nodes before the boundary of the concatenated axis; 0 for an unsplit kernel, where the
      // is_tail flag is dead code that the `if constexpr` in node_value() removes anyway.
      const size_t n_tail = m_n_tail;
      const auto &start = grid_start;
      const auto &scale = grid_scale;

      // The external position is a function of idx[0] alone, but this functor runs
      // integral_view.size() * prod(grid_size) threads -- and grid_size carries the Matsubara axis,
      // so the per-thread forward() (a fp64 expm1/sinh+exp on the logarithmic coordinate classes)
      // is paid tens of times more often here than in the vacuum integrator. Precompute the
      // positions once per coordinate system into a device view, exactly as
      // QuadratureIntegrator::map does; see the comments there for why the key is built this way
      // and why coordinates without a to_string() identity keep the per-thread computation.
      constexpr size_t cdim = Coordinates::dim;
      if constexpr (has_cacheable_positions_v<Coordinates>) {
        std::string key = coordinates.to_string() + "|" + std::to_string(integral_view.size());
        {
          char buf[64];
          const auto first = coordinates.forward(coordinates.from_linear_index(size_t(0)));
          const auto last = coordinates.forward(coordinates.from_linear_index(integral_view.size() - 1));
          for (size_t d = 0; d < cdim; ++d) {
            std::snprintf(buf, sizeof(buf), "|%la|%la", double(first[d]), double(last[d]));
            key += buf;
          }
        }
        const size_t need = integral_view.size() * cdim;
        if (m_positions_key != key || m_positions.extent(0) < need) {
          if (m_positions.extent(0) < need)
            m_positions = Kokkos::View<ctype *, typename ExecutionSpace::memory_space>(
                Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "QuadratureIntegrator_fT_positions"), need);
          const auto pos_fill = m_positions;
          const auto coords = coordinates;
          Kokkos::parallel_for(
              "QuadratureIntegrator_fT_fill_positions",
              Kokkos::RangePolicy<ExecutionSpace>(space, 0, integral_view.size()), KOKKOS_LAMBDA(const size_t i) {
                const auto p = coords.forward(coords.from_linear_index(i));
                for (size_t d = 0; d < cdim; ++d)
                  pos_fill(i * cdim + d) = p[d];
              });
          m_positions_key = key;
        }
      }
      const auto pos_view = m_positions;
      using pos_ctype = typename Coordinates::ctype;
      // Runtime copy for the team lambda below: its constant() evaluation runs once per team, so a
      // plain branch there costs nothing and avoids nvcc's fragile handling of if-constexpr inside
      // extended class lambdas.
      const bool pos_cached_rt = has_cacheable_positions_v<Coordinates>;

      // Complete functors, selected by a HOST-level if constexpr -- nvcc miscompiles an
      // `if constexpr` inside the extended lambda body. See QuadratureIntegrator::map.
      // Hoisted: pre() must not read the external position, which sits right after the frequency.
      if constexpr (can_hoist && has_cacheable_positions_v<Coordinates> &&
                    !internal::pre_reads_any<KERNEL>(sdim + 1, sdim + 1 + cdim)) {
        const device::array<pos_ctype, cdim> pos0{};
        using PT = decltype(pre_value(device::array<ctype, sdim>{}, ctype(0), pos0, m_args));
        const PT *table = prepare_pre_table<PT>(space, pos0, m_args, sdim + 1 + cdim, args...);
        const size_t n_m = grid_size[dim - 1];
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, 1 + dim> &idx)
        {
          auto subview = device::apply([&](const auto &...i) { return Kokkos::subview(cache, i...); }, idx);

          device::array<pos_ctype, cdim> pos;
          for (size_t d = 0; d < cdim; ++d)
            pos[d] = static_cast<pos_ctype>(pos_view(idx[0] * cdim + d));

          device::array<ctype, sdim> x;
          ctype weight = 1;
          for (int i = 0; i < sdim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[1 + i]], start[i]);
            weight *= w[i][idx[1 + i]] * scale[i];
          }
          const size_t jt = idx[1 + dim - 1];
          const size_t e = (sdim > 0 ? idx[1] : 0) * n_m + jt;
          subview() = weight * hoisted_node_value(table + 2 * e, x, pos, m_args, m_n[jt], m_w[jt]);
        };
        Kokkos::parallel_for(make_kokkos_nd_range_divisible<1 + dim, ExecutionSpace>(space, {0}, extents),
                             internal::KokkosNDLambdaWrapper<1 + dim, decltype(functor)>(functor));
      } else if constexpr (has_cacheable_positions_v<Coordinates>) {
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, 1 + dim> &idx)
        {
          // make subview
          auto subview = device::apply([&](const auto &...i) { return Kokkos::subview(cache, i...); }, idx);

          // get the (precomputed) position for the current index
          device::array<pos_ctype, cdim> pos;
          for (size_t d = 0; d < cdim; ++d)
            pos[d] = static_cast<pos_ctype>(pos_view(idx[0] * cdim + d));

          device::array<ctype, sdim> x;
          ctype weight = 1;
          for (int i = 0; i < sdim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[1 + i]], start[i]);
            weight *= w[i][idx[1 + i]] * scale[i];
          }
          const size_t jt = idx[1 + dim - 1];
          subview() = weight * node_value(x, pos, m_args, m_n[jt], m_w[jt], jt < n_tail);
        };
        Kokkos::parallel_for(make_kokkos_nd_range_divisible<1 + dim, ExecutionSpace>(space, {0}, extents),
                             internal::KokkosNDLambdaWrapper<1 + dim, decltype(functor)>(functor));
      } else {
        auto functor = KOKKOS_LAMBDA(const device::array<size_t, 1 + dim> &idx)
        {
          // make subview
          auto subview = device::apply([&](const auto &...i) { return Kokkos::subview(cache, i...); }, idx);

          // get the position for the current index
          const auto pos = coordinates.forward(coordinates.from_linear_index(idx[0]));

          device::array<ctype, sdim> x;
          ctype weight = 1;
          for (int i = 0; i < sdim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[1 + i]], start[i]);
            weight *= w[i][idx[1 + i]] * scale[i];
          }
          const size_t jt = idx[1 + dim - 1];
          subview() = weight * node_value(x, pos, m_args, m_n[jt], m_w[jt], jt < n_tail);
        };
        Kokkos::parallel_for(make_kokkos_nd_range_divisible<1 + dim, ExecutionSpace>(space, {0}, extents),
                             internal::KokkosNDLambdaWrapper<1 + dim, decltype(functor)>(functor));
      }

      using TeamType = Kokkos::TeamPolicy<ExecutionSpace>::member_type;
      // reduction with vector lanes for warp-level parallelism
      constexpr int vector_width = 32;
      Kokkos::parallel_for(
          Kokkos::TeamPolicy(space, integral_view.size(), Kokkos::AUTO, vector_width),
          KOKKOS_CLASS_LAMBDA(const TeamType &team) {
            // get the current (continuous) index
            const uint k = team.league_rank();

            if (k >= integral_view.size()) return;

            // no-ops to capture
            (void)cache;
            (void)grid_size;

            // Flatten grid_size into total element count for thread+vector splitting
            size_t total_elements = 1;
            for (int d = 0; d < dim; ++d)
              total_elements *= grid_size[d];

            // Pre-compute stride array for index decomposition (avoids modulo in the inner loop;
            // matches the vacuum QuadratureIntegrator::map reduction)
            device::array<size_t, dim> strides;
            strides[dim - 1] = 1;
            for (int d = dim - 2; d >= 0; --d)
              strides[d] = strides[d + 1] * grid_size[d + 1];

            NT res{};
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, (total_elements + vector_width - 1) / vector_width),
                [&](const size_t outer, NT &team_update) {
                  NT vec_sum{};
                  Kokkos::parallel_reduce(
                      Kokkos::ThreadVectorRange(team, vector_width),
                      [&](const size_t inner, NT &vec_update) {
                        const size_t flat = outer * vector_width + inner;
                        if (flat < total_elements) {
                          // Convert flat index back to multi-dimensional using pre-computed strides
                          device::array<size_t, dim> ridx;
                          size_t remainder = flat;
                          for (int d = 0; d < dim; ++d) {
                            ridx[d] = remainder / strides[d];
                            remainder -= ridx[d] * strides[d];
                          }
                          device::apply([&](const auto &...iargs) { vec_update += cache(k, iargs...); }, ridx);
                        }
                      },
                      vec_sum);
                  team_update += vec_sum;
                },
                res);

            // add the constant value (skip coordinate computation if kernel has no constant)
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              device::array<pos_ctype, cdim> pos;
              if (pos_cached_rt) {
                for (size_t d = 0; d < cdim; ++d)
                  pos[d] = static_cast<pos_ctype>(pos_view(size_t(k) * cdim + d));
              } else {
                pos = coordinates.forward(coordinates.from_linear_index(k));
              }
              // Nested packs, not tuple_cat -- same reason as node_value(). This kernel does almost
              // no arithmetic but carried a full per-thread copy of every interpolator.
              integral_view(k) =
                  res + device::apply(
                            [&](const auto &...pargs) {
                              return device::apply(
                                  [&](const auto &...iargs) { return KERNEL::constant(pargs..., iargs...); }, m_args);
                            },
                            pos);
            });
          });
    }

    /// Number of quadrature points, i.e. kernel evaluations per integral (frequency nodes included).
    size_t quadrature_volume() const
    {
      size_t volume = 1;
      for (int i = 0; i < dim; ++i)
        volume *= grid_size[i];
      return volume;
    }

    /**
     * @brief Evaluate the integral at dest.size() points in a single launch; see QuadratureIntegrator::map_points.
     * The frequency rule (Matsubara sum, exact sum or vacuum integral) is the one get() uses, i.e. the one chosen
     * for the current T, k and typical_E: it is shared by all points.
     */
    template <typename OT, typename... A>
      requires(internal::is_map_result<OT, NT> &&
               internal::accepts_args<NT, KERNEL, ctype, dim, internal::point_arg_value_t<A>...>)
    void map_points(const PointSpan<OT> dest, const A &...args)
    {
      run_map_points(dest, PointArg<internal::point_arg_value_t<A>>(args)...);
    }

    /// map_points() for a kernel returning MultiValue<T, N>: component j of every integral lands in `dest[j]`.
    template <typename OT, size_t N, typename... A>
      requires(internal::is_map_result<MultiValue<OT, N>, NT> &&
               internal::accepts_args<NT, KERNEL, ctype, dim, internal::point_arg_value_t<A>...>)
    void map_points(const std::array<PointSpan<OT>, N> &dest, const A &...args)
    {
      run_map_points(dest, PointArg<internal::point_arg_value_t<A>>(args)...);
    }

    /// How a device map_points() distributes its work; see MapPointsPolicy.
    void set_map_points_policy(const MapPointsPolicy policy) { m_map_points_policy = policy; }

    /// The kernel launch of map_points() for a hoisted kernel, from the pre() table @p table. Public for the same
    /// reason as launch_map_points().
    template <typename ResultView, typename DeviceArgs, typename PT>
    void launch_map_points_hoisted(const ResultView &result, const size_t n, const DeviceArgs &device_args,
                                   const PT *table)
    {
      using OT = typename ResultView::value_type;

      const auto &nd = nodes;
      const auto &w = weights;
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      const auto &start = grid_start;
      const auto &scale = grid_scale;
      const auto gs = grid_size;
      const auto args = device_args;
      const size_t n_m = grid_size[dim - 1];

      size_t total = 1;
      for (int d = 0; d < dim; ++d)
        total *= grid_size[d];
      device::array<size_t, dim> strides;
      strides[dim - 1] = 1;
      for (int d = dim - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * gs[d + 1];

      // As launch_map_points(), with every node taking its pre() from the table. Always a team per point: the
      // remaining work per node is small, so the points alone rarely fill the device.
      using TeamType = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;
      constexpr int vector_width = 32;
      const size_t n_outer = (total + vector_width - 1) / vector_width;
      Kokkos::parallel_for(
          "QuadratureIntegrator_fT_map_points_hoisted",
          Kokkos::TeamPolicy<ExecutionSpace>(space, n, Kokkos::AUTO, vector_width),
          KOKKOS_LAMBDA(const TeamType &team) {
            const size_t i = team.league_rank();
            // References, not copies: a shared argument may be an interpolator (see node_value).
            const auto point_args = device::apply([&](const auto &...a) { return device::tie(a(i)...); }, args);
            NT integral{};
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, n_outer),
                [&](const size_t outer, NT &team_update) {
                  NT vec_sum{};
                  Kokkos::parallel_reduce(
                      Kokkos::ThreadVectorRange(team, vector_width),
                      [&](const size_t inner, NT &vec_update) {
                        const size_t flat = outer * vector_width + inner;
                        if (flat < total) {
                          device::array<ctype, sdim> x;
                          ctype weight = 1;
                          size_t remainder = flat;
                          for (int d = 0; d < sdim; ++d) {
                            const size_t id = remainder / strides[d];
                            remainder -= id * strides[d];
                            x[d] = Kokkos::fma(scale[d], nd[d][id], start[d]);
                            weight *= w[d][id] * scale[d];
                          }
                          const size_t jt = remainder;
                          const size_t e = (sdim > 0 ? flat / strides[0] : 0) * n_m + jt;
                          vec_update += weight * hoisted_node_value(table + 2 * e, x, device::tuple<>{}, point_args,
                                                                    m_n[jt], m_w[jt]);
                        }
                      },
                      vec_sum);
                  team_update += vec_sum;
                },
                integral);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              NT sum = device::apply([&](const auto &...a) { return NT(KERNEL::constant(a(i)...)); }, args);
              sum += integral;
              result(i) = static_cast<OT>(sum);
            });
          });
    }

    /// The kernel launch of map_points(). Public only because nvcc rejects extended device lambdas
    /// inside non-public member functions.
    template <typename ResultView, typename DeviceArgs>
    void launch_map_points(const ResultView &result, const size_t n, const DeviceArgs &device_args)
    {
      using OT = typename ResultView::value_type;

      const auto &nd = nodes;
      const auto &w = weights;
      const auto &m_n = matsubara_nodes;
      const auto &m_w = matsubara_weights;
      const size_t n_tail = m_n_tail;
      const auto &start = grid_start;
      const auto &scale = grid_scale;
      const auto gs = grid_size;
      const auto args = device_args;

      size_t total = 1;
      for (int d = 0; d < dim; ++d)
        total *= grid_size[d];
      device::array<size_t, dim> strides;
      strides[dim - 1] = 1;
      for (int d = dim - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * gs[d + 1];

      // The integrand at grid node `flat` (row-major, the frequency axis last) for the arguments of point i.
      const auto node = KOKKOS_LAMBDA(const size_t flat, const size_t i)->NT
      {
        device::array<ctype, sdim> x;
        ctype weight = 1;
        size_t remainder = flat;
        for (int d = 0; d < sdim; ++d) {
          const size_t id = remainder / strides[d];
          remainder -= id * strides[d];
          x[d] = Kokkos::fma(scale[d], nd[d][id], start[d]);
          weight *= w[d][id] * scale[d];
        }
        const size_t jt = remainder;
        // References, not copies: a shared argument may be an interpolator (see node_value).
        const auto point_args = device::apply([&](const auto &...a) { return device::tie(a(i)...); }, args);
        return weight * node_value(x, device::tuple<>{}, point_args, m_n[jt], m_w[jt], jt < n_tail);
      };

      if (internal::resolve_policy(m_map_points_policy, n, total) == MapPointsPolicy::thread_per_point) {
        Kokkos::parallel_for(
            "QuadratureIntegrator_fT_map_points", Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
            KOKKOS_LAMBDA(const size_t i) {
              NT sum = device::apply([&](const auto &...a) { return NT(KERNEL::constant(a(i)...)); }, args);
              NT integral{};
              for (size_t flat = 0; flat < total; ++flat)
                integral += node(flat, i);
              sum += integral;
              result(i) = static_cast<OT>(sum);
            });
      } else {
        using TeamType = typename Kokkos::TeamPolicy<ExecutionSpace>::member_type;
        constexpr int vector_width = 32;
        const size_t n_outer = (total + vector_width - 1) / vector_width;
        Kokkos::parallel_for(
            "QuadratureIntegrator_fT_map_points_team",
            Kokkos::TeamPolicy<ExecutionSpace>(space, n, Kokkos::AUTO, vector_width),
            KOKKOS_LAMBDA(const TeamType &team) {
              const size_t i = team.league_rank();
              NT integral{};
              Kokkos::parallel_reduce(
                  Kokkos::TeamThreadRange(team, n_outer),
                  [&](const size_t outer, NT &team_update) {
                    NT vec_sum{};
                    Kokkos::parallel_reduce(
                        Kokkos::ThreadVectorRange(team, vector_width),
                        [&](const size_t inner, NT &vec_update) {
                          const size_t flat = outer * vector_width + inner;
                          if (flat < total) vec_update += node(flat, i);
                        },
                        vec_sum);
                    team_update += vec_sum;
                  },
                  integral);
              Kokkos::single(Kokkos::PerTeam(team), [&]() {
                NT sum = device::apply([&](const auto &...a) { return NT(KERNEL::constant(a(i)...)); }, args);
                sum += integral;
                result(i) = static_cast<OT>(sum);
              });
            });
      }
    }

    /**
     * @brief Compute the integral at every point of a grid of external values; see QuadratureIntegrator::map.
     *
     * The kernel receives `(x..., q0, p..., args...)`, with \f$p\f$ the external point.
     */
    template <typename OT, typename Coordinates, typename... Args>
      requires internal::is_map_result<OT, NT>
    auto map(OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::scheduled_map<ExecutionSpace>(integrator_id(), quadrature_volume(), dest, coordinates,
                                              [&](auto *d, const auto &c) { this->map_dist(d, c, args...); });
      return space;
    }

    /// map() for a kernel returning MultiValue<T, N>: component j of every integral lands in `dest[j]`.
    template <typename OT, size_t N, typename Coordinates, typename... Args>
      requires internal::is_map_result<MultiValue<OT, N>, NT>
    auto map(const std::array<OT *, N> &dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::scheduled_map<ExecutionSpace>(integrator_id(), quadrature_volume(), dest, coordinates,
                                              [&](const auto &d, const auto &c) { this->map_dist(d, c, args...); });
      return space;
    }

    /// map() into MultiValue destinations given as a braced list: `map({&a[0], &b[0]}, coordinates, args...)`.
    template <typename OT, size_t N, typename Coordinates, typename... Args>
      requires internal::is_map_result<MultiValue<OT, N>, NT>
    auto map(OT *const (&dest)[N], const Coordinates &coordinates, const Args &...args)
    {
      return map(std::to_array(dest), coordinates, args...);
    }

    /// map() without the MapScheduler: computes the whole of `coordinates` on this rank.
    template <typename OT, typename Coordinates, typename... Args>
    auto map_dist(OT *dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::staged_map(
          space, m_staging.template get<OT>(), dest, coordinates.size(),
          [this, coordinates, args...](const auto &view) { this->map(this->space, view, coordinates, args...); });
      return space;
    }

    /// map_dist() for MultiValue destinations.
    template <typename OT, size_t N, typename Coordinates, typename... Args>
    auto map_dist(const std::array<OT *, N> &dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::staged_map(
          space, m_staging.template get<MultiValue<OT, N>>(), dest, coordinates.size(),
          [this, coordinates, args...](const auto &view) { this->map(this->space, view, coordinates, args...); });
      return space;
    }

  private:
    template <typename Dest, typename... T> void run_map_points(const Dest &dest, const PointArg<T> &...args)
    {
      const size_t n = internal::point_count(dest);
      if (n == 0) return;
      constexpr bool handles = internal::takes_handles<NT, KERNEL, ctype, dim, T...>;
      internal::check_kernel_precision<handles, ctype, T...>();
      const auto device_args = m_map_points.template stage<ctype, handles>(space, n, args...);
      if constexpr (can_hoist) {
        // Hoisted unless pre() reads an argument given per point.
        bool hoist = true;
        size_t i = sdim + 1;
        ((hoist = hoist && !(args.per_point() && internal::pre_reads<KERNEL>(i)), ++i), ...);
        if (hoist) {
          using PT = decltype(pre_value_points(device::array<ctype, sdim>{}, ctype(0), device_args));
          PT *table = pre_table_storage<PT>();
          std::string key = pre_table_key<PT>(sdim + 1, args.value...);
          if (key.empty() || key != m_pre_key) {
            fill_pre_table_points(table, device_args);
            m_pre_key = std::move(key);
            note_pre_fill(space);
          } else
            wait_for_pre_fill(space);
          m_map_points.run(space, dest,
                           [&](const auto &result) { launch_map_points_hoisted(result, n, device_args, table); });
          return;
        }
      }
      m_map_points.run(space, dest, [&](const auto &result) { launch_map_points(result, n, device_args); });
    }

    /**
     * @brief Re-select and re-fetch the frequency rule after T, k, typical_E or the cutoff changed.
     *
     * Three rules are on offer, and every choice between them is made here rather than inside
     * MatsubaraQuadrature, because they are questions about the summand's SPECTRUM and only this
     * class is told about it (through `k` and `typical_E`).
     *
     * 1. Sum or integral, and with how many nodes -- one question, decided on cost alone. Size the
     *    Monien rule to reach past `typical_E`; if that fits `max_matsubara_size`, sum. If it does
     *    not, a clamped sum would no longer reach the structure it was sized for, so integrate
     *    instead. The error that switch commits is then bounded by a budget the user sets, rather
     *    than by a constant claiming the thermal content has died away -- a claim that was wrong
     *    by nine orders for a 4D-regulated summand, which has no pole for the `4 exp(-E/T)` law to
     *    apply to (Part 13 of the convergence study: 3.1e-4 measured against 1.9e-13 claimed).
     *
     *    What the switch still cannot see is a scale in the summand LIGHTER than `typical_E`: the
     *    integral discards its thermal content and nothing here knows it is there. That is the
     *    reason `typical_E` is worth setting, and the reason it should be the scale that matters
     *    rather than the largest one present.
     *
     * 2. Which scale to size against. The reach must cover `typical_E` with a multiplier well
     *    above one, or the rule places no nodes where the structure lives. This is not a tail
     *    multiplier a compactly supported summand could do without: sizing a rule to merely SPAN
     *    the support of a 4D-regulated kernel is wrong by 4e-5 at k/T = 30 and by 2.5e-2 at
     *    k/T = 300 (Part 13). Reach buys node density. See `monien_reach`.
     *
     * 3. Exact or Gaussian, for a summand of finite extent. Purely cost: the exact sum is never
     *    *less* accurate than the Gaussian rule -- it is the sum itself -- so whichever has fewer
     *    nodes wins. The crossover is crossed during a flow (the exact sum shrinks with k while
     *    the Monien rule grows), which is why this is decided per RG step.
     */
    void refresh_matsubara()
    {
      // The kernel must be COMPLETE here, because this function branches on its traits. Every
      // `requires { requires K::trait; }` detector reads false on an incomplete K -- a substitution
      // failure, not an error -- so a translation unit that has not seen the kernel definition
      // would build a different integrator than one that has, pick a different Matsubara rule, and
      // produce a wrong right-hand side with no diagnostic anywhere. That is not hypothetical: the
      // generated flow scaffolds used to forward-declare their kernel in <Flow>.hh, and flows.cc
      // (which instantiates set_T -> here) disagreed with the CT_*.cc translation units (which
      // instantiate map()). Fail loudly instead.
      static_assert(sizeof(KERNEL) > 0, "QuadratureIntegrator_fT: the KERNEL type must be complete here. Include the "
                                        "flow's kernel.hh before its <Flow>.hh -- a forward declaration silently turns "
                                        "every kernel trait off in this translation unit and yields a wrong RHS.");

      using mem_space = typename ExecutionSpace::memory_space;

      // The one scale the frequency rule is built from: what the model said, or k if it never
      // said anything. `k` is the DEFAULT, not a floor -- a model that reports a scale is reporting
      // the summand's spectrum, and the integrator must not second-guess it. `max(k, typical_E)`
      // was tried and is wrong: it makes the budget test below fire on a scale the summand does
      // not have, sending a genuinely thermal sum to the vacuum rule.
      const ctype E = m_typical_E_user > ctype(0) ? m_typical_E_user : m_k;

      // Choice 1: build the Monien rule if it fits the budget, otherwise integrate. A T of zero is
      // how the provider is asked for the vacuum rule; it takes the same scale, since its tangent
      // map spends its nodes around E and covers the algebraic tail by construction.
      const bool vacuum = !(T > ctype(0)) || quadrature_provider.template matsubara_predicted_size<ctype>(T, E) >
                                                 quadrature_provider.max_matsubara_size();

      const auto &standard = quadrature_provider.template matsubara_rule<ctype>(vacuum ? ctype(0) : T, E);
      m_using_exact = false;

      // Which rule the finite-extent side should run on. Never *less* accurate than the Gaussian
      // rule -- it is the sum itself -- so the choice is purely one of cost, and the crossover is
      // crossed during a flow (the exact sum shrinks with k while the Monien rule grows), which is
      // why it is decided per RG step.
      const MatsubaraQuadrature<ctype> *fe_rule = &standard;
      if (m_allow_exact && T > ctype(0) && m_freq_cutoff > ctype(0)) {
        // One mode of margin. A FERMIONIC insertion confines p0 to an interval centred at -pi T,
        // not at zero, so its support can reach one bosonic mode further in the negative direction
        // than a symmetric list built from the cutoff alone would cover. One node is a cheap price
        // for not having to know which species a given kernel's insertion belongs to.
        const ctype cutoff = m_extent_margin * m_freq_cutoff + ctype(2 * M_PI) * T;
        // Price it BEFORE building it. modes_below() is three flops; the rule itself is an O(N)
        // table and six Kokkos views, and in the UV the answer is thousands of modes that would be
        // discarded on the very next line. Measured on QCD_Nf2 at T = 0.1: every step from k = 1000
        // down to k = 150 built and threw away a rule, 1963 modes at the top and still 294 at the
        // bottom -- and the provider's map never evicts, so each one leaked for the process.
        const size_t n_exact = size_t(MatsubaraQuadrature<ctype>::modes_below(T, cutoff)) + 1;
        if (n_exact < standard.sum_size()) {
          fe_rule = &quadrature_provider.template matsubara_exact_sum<ctype>(T, cutoff);
          m_using_exact = true;
        } else if (is_close(standard.get_T(), ctype(0))) {
          // Above the crossover the exact sum is the expensive rule -- but `standard` is then the
          // T=0 TANGENT MAP, which spends 43% of its nodes past this summand's support evaluating
          // an exact zero, and weights those most heavily. Over a finite interval there is nothing
          // to reach for, so plain Gauss-Legendre at the SPATIAL order is both cheaper and more
          // accurate. (If `standard` is the Monien rule instead, the sum is genuinely thermal and
          // must not be replaced by an integral, so leave it alone.) The order is the SPATIAL one:
          // for a 4D regulator p0 and |q| enter the support p0^2 + |q|^2 <= x_extent k^2 on the
          // same footing, so the order that resolves the radius resolves the frequency too. Without a spatial axis
          // (a pure frequency sum), keep the size of the tangent map.
          size_t order = standard.sum_size();
          if constexpr (sdim > 0) order = grid_size[0];
          fe_rule =
              &quadrature_provider.template matsubara_finite_interval<ctype>(m_extent_margin * m_freq_cutoff, order);
        }
      }

      // sum_nodes(), not nodes(): the zero mode is an ordinary node on this axis, so the hot
      // functor has no branch and an exact sum with no positive modes at all still evaluates it.
      if constexpr (kernel_has_matsubara_split<KERNEL>) {
        // Concatenate: [ tail half on the Gaussian rule | finite-extent half on fe_rule ]. Each
        // half carries its own zero mode, because each is a complete rule for its own summand.
        //
        // The axis is built this way even ABOVE the crossover, where fe_rule IS the Gaussian rule
        // and the concatenation is just that rule twice. That looks wasteful and is very nearly
        // free -- each node evaluates half a kernel, so the arithmetic is the same as one pass of
        // the full body -- and it buys something worth more: the integrator never has to call
        // KERNEL::kernel, so only TWO bodies are ever inlined into the launch instead of three.
        // On kernels already sitting at REG=255 with spills, a third inlined body is not free.
        const auto tail_n = standard.template sum_nodes<mem_space>();
        const auto tail_w = standard.template sum_weights<mem_space>();
        const auto fe_n = fe_rule->template sum_nodes<mem_space>();
        const auto fe_w = fe_rule->template sum_weights<mem_space>();

        m_n_tail = tail_n.size();
        const size_t n = m_n_tail + fe_n.size();

        if (m_split_nodes.extent(0) < n) {
          m_split_nodes = Kokkos::View<ctype *, mem_space>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, "QuadratureIntegrator_fT_split_nodes"), n);
          m_split_weights = Kokkos::View<ctype *, mem_space>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing, "QuadratureIntegrator_fT_split_weights"), n);
        }
        const auto head = Kokkos::make_pair(size_t(0), m_n_tail);
        const auto tail = Kokkos::make_pair(m_n_tail, n);
        Kokkos::deep_copy(Kokkos::subview(m_split_nodes, head), tail_n);
        Kokkos::deep_copy(Kokkos::subview(m_split_weights, head), tail_w);
        Kokkos::deep_copy(Kokkos::subview(m_split_nodes, tail), fe_n);
        Kokkos::deep_copy(Kokkos::subview(m_split_weights, tail), fe_w);

        matsubara_nodes = Kokkos::View<const ctype *, mem_space>(m_split_nodes, Kokkos::make_pair(size_t(0), n));
        matsubara_weights = Kokkos::View<const ctype *, mem_space>(m_split_weights, Kokkos::make_pair(size_t(0), n));
      } else {
        m_n_tail = 0;
        matsubara_nodes = fe_rule->template sum_nodes<mem_space>();
        matsubara_weights = fe_rule->template sum_weights<mem_space>();
      }
      grid_size[dim - 1] = matsubara_nodes.size();
    }

  protected:
    /// Number of pre() table rows: first box nodes times frequency nodes.
    size_t pre_table_entries() const { return (sdim > 0 ? size_t(grid_size[0]) : size_t(1)) * grid_size[dim - 1]; }

    /// Storage for the pre() table with entries of type PT; invalidates the table if it had to grow.
    template <typename PT> PT *pre_table_storage() const
    {
      static_assert(std::is_trivially_copyable_v<PT>, "pre() must return a trivially copyable value.");
      const size_t bytes = 2 * pre_table_entries() * sizeof(PT);
      if (m_pre_table.extent(0) < bytes) {
        m_pre_table = Kokkos::View<char *, typename ExecutionSpace::memory_space>(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "QuadratureIntegrator_fT_pre_table"), bytes);
        m_pre_key.clear();
      }
      return reinterpret_cast<PT *>(m_pre_table.data());
    }

    /**
     * @brief What a pre() table with entries PT for the kernel() parameters @p rest (the first being parameter
     * @p first) is computed from; empty if that cannot be told or reuse is off, and then the table is always rebuilt.
     */
    template <typename PT, typename... A> std::string pre_table_key(const size_t first, const A &...rest) const
    {
      if (!m_pre_reuse) return {};
      std::string key = typeid(PT).name();
      const auto add = [&](const auto &v) { key.append(reinterpret_cast<const char *>(&v), sizeof(v)); };
      add(internal::current_interpolator_data_epoch());
      add(matsubara_nodes.data());
      add(matsubara_nodes.size());
      if constexpr (sdim > 0) {
        add(nodes[0].data());
        add(nodes[0].size());
        add(grid_start[0]);
        add(grid_scale[0]);
      }
      bool known = true;
      size_t i = first;
      ((known = known && (!internal::pre_reads<KERNEL>(i) || internal::append_identity(key, rest)), ++i), ...);
      return known ? key : std::string();
    }

    /// Record that the pre() table was just filled on @p space.
    void note_pre_fill(const ExecutionSpace &space) const
    {
      m_pre_fill_space = space;
      m_pre_fill_elsewhere = &space != &this->space;
    }

    /// Before reusing the pre() table on @p space: wait for its fill if that ran on another stream instance.
    void wait_for_pre_fill(const ExecutionSpace &space) const
    {
      if (m_pre_fill_elsewhere || &space != &this->space) m_pre_fill_space.fence();
    }

    /// The pre() table for kernel() parameters @p rest (see pre_table_key()), rebuilt on @p space if stale.
    template <typename PT, typename PosArr, typename ArgTuple, typename... A>
    const PT *prepare_pre_table(ExecutionSpace &space, const PosArr &pos, const ArgTuple &args, const size_t first,
                                const A &...rest) const
    {
      PT *table = pre_table_storage<PT>();
      std::string key = pre_table_key<PT>(first, rest...);
      if (key.empty() || key != m_pre_key) {
        fill_pre_table(space, table, pos, args);
        m_pre_key = std::move(key);
        note_pre_fill(space);
      } else
        wait_for_pre_fill(space);
      return table;
    }

    /// Mutable because the const get() overloads issue work on it: which stream instance a launch
    /// goes to is not part of the integrator's logical state.
    mutable ExecutionSpace space;
    QuadratureProvider &quadrature_provider;
    device::array<device::array<ctype, sdim>, 2> grid_extents;
    device::array<ctype, sdim> grid_start;
    device::array<ctype, sdim> grid_scale;

    device::array<size_t, dim> grid_size;

    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, sdim> nodes;
    device::array<Kokkos::View<const ctype *, typename ExecutionSpace::memory_space>, sdim> weights;

    ctype T;
    /// The RG scale, which is the frequency scale unless the model set one; see set_k().
    ctype m_k;
    /// The model's heaviest scale, or zero if it never said; see set_typical_E().
    ctype m_typical_E_user = 0;
    /// Support boundary in frequency; zero means "unknown", which disables the exact sum.
    ctype m_freq_cutoff = 0;
    /// Nodes before the boundary of the concatenated axis of a split kernel; 0 when not split.
    size_t m_n_tail = 0;
    /// Owned concatenation [tail | finite-extent]; the two halves come from different rules, so
    /// unlike every other node list this one cannot be a view into the provider's cache.
    Kokkos::View<ctype *, typename ExecutionSpace::memory_space> m_split_nodes, m_split_weights;
    bool m_allow_exact = false;
    bool m_using_exact = false;
    ctype m_extent_margin = 1;

    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> matsubara_nodes;
    Kokkos::View<const ctype *, typename ExecutionSpace::memory_space> matsubara_weights;

    // Persistent view caches to avoid per-call GPU memory allocation
    mutable KokkosNDView<1 + dim, NT, ExecutionSpace> m_cache;
    mutable device::array<size_t, 1 + dim> m_cache_extents{};
    // Cached external positions for map(): one forward() per grid point instead of per thread.
    // Keyed on the coordinates' to_string() identity; flat layout [grid_point * cdim + d].
    mutable Kokkos::View<ctype *, typename ExecutionSpace::memory_space> m_positions;
    mutable std::string m_positions_key;
    mutable internal::MapStagingSet<NT, ExecutionSpace> m_staging;
    // map_points() buffers, separate from map()'s, whose staging may still be pending in a DeferredMaps scope.
    internal::MapPointsBuffers<NT, ExecutionSpace> m_map_points;
    MapPointsPolicy m_map_points_policy = MapPointsPolicy::automatic;
    mutable Kokkos::View<NT, typename ExecutionSpace::memory_space> m_result_view;
    mutable typename Kokkos::View<NT, typename ExecutionSpace::memory_space>::host_mirror_type m_result_host;
    mutable bool m_result_views_initialized = false;
    // The pre() table of a hoisted kernel (see kernel_has_pre), raw bytes of its entry type; what it was computed
    // from (pre_table_key(), empty: rebuild on next use); the stream it was last filled on.
    mutable Kokkos::View<char *, typename ExecutionSpace::memory_space> m_pre_table;
    mutable std::string m_pre_key;
    mutable ExecutionSpace m_pre_fill_space;
    mutable bool m_pre_fill_elsewhere = false;
    bool m_pre_reuse = true;
  };

  /**
   * @brief QuadratureIntegrator_fT on the host with TBB.
   *
   * Same interface and kernel requirements as the general class. Each integral is summed in a fixed order, so results
   * do not depend on the number of threads.
   */
  template <int dim, typename NT, typename KERNEL>
  class QuadratureIntegrator_fT<dim, NT, KERNEL, TBB_exec>
      : public QuadratureIntegrator_fT<dim, NT, KERNEL, KokkosHost_exec>
  {
    using Base = QuadratureIntegrator_fT<dim, NT, KERNEL, KokkosHost_exec>;

  public:
    /// Type of the integration variables.
    using ctype = typename get_type::ctype<NT>;
    /// The execution space, TBB_exec.
    using execution_space = TBB_exec;

    /// Number of integration variables besides the frequency.
    static constexpr int sdim = dim - 1;

    /// See QuadratureIntegrator_fT::QuadratureIntegrator_fT.
    QuadratureIntegrator_fT(QuadratureProvider &quadrature_provider, const std::array<size_t, sdim> _grid_size,
                            std::array<ctype, sdim> grid_min, std::array<ctype, sdim> grid_max,
                            const std::array<QuadratureType, sdim> quadrature_type, const ctype T = 1,
                            const ctype k = 1)
        : Base(quadrature_provider, _grid_size, grid_min, grid_max, quadrature_type, T, k)
    {
    }

    /// See QuadratureIntegrator_fT::get.
    template <typename... Args>
      requires internal::accepts_args<NT, KERNEL, ctype, dim, Args...>
    void get(NT &dest, const Args &...t) const
    {
      // a kernel that takes handles gets those
      constexpr bool handles = internal::takes_handles<NT, KERNEL, ctype, dim, Args...>;
      internal::check_kernel_precision<handles, ctype, Args...>();
      if constexpr (handles && !(std::is_same_v<Args, kernel_handle_t<Args, ctype>> && ...))
        get(dest, internal::to_kernel_arg<handles, ctype>(t)...);
      else {
        const auto args = device::tie(t...);

        const auto &n = nodes;
        const auto &w = weights;
        const auto &m_n = matsubara_nodes;
        const auto &m_w = matsubara_weights;
        // Nodes before the boundary of the concatenated axis; 0 for an unsplit kernel, where the
        // is_tail flag is dead code that the `if constexpr` in node_value() removes anyway.
        const size_t n_tail = m_n_tail;
        const auto &start = grid_start;
        const auto &scale = grid_scale;

        auto functor = [&](const device::array<size_t, dim> &idx) {
          device::array<ctype, sdim> x;
          ctype weight = 1;
          for (int i = 0; i < sdim; ++i) {
            x[i] = Kokkos::fma(scale[i], n[i][idx[i]], start[i]);
            weight *= w[i][idx[i]] * scale[i];
          }
          const size_t jt = idx[dim - 1];
          // Empty position pack: this overload's caller passes the external position inside `args`.
          return weight * Base::node_value(x, device::tuple<>{}, args, m_n[jt], m_w[jt], jt < n_tail);
        };

        dest = KERNEL::constant(t...) + TBBReduction<dim, NT, decltype(functor)>(grid_size, functor);
      }
    }

    template <typename OT, typename... Args>
      requires(internal::is_widened_result<OT, NT> && internal::accepts_args<NT, KERNEL, ctype, dim, Args...>)
    void get(OT &dest, const Args &...t) const
    {
      NT result;
      get(result, t...);
      dest = OT(result);
    }

    /// See QuadratureIntegrator_fT::map_points. One get() per point inside a flat parallel loop, so every
    /// result is bitwise identical to the corresponding get().
    template <typename OT, typename... A>
      requires(internal::is_map_result<OT, NT> &&
               internal::accepts_args<NT, KERNEL, ctype, dim, internal::point_arg_value_t<A>...>)
    void map_points(const PointSpan<OT> dest, const A &...args) const
    {
      internal::map_points_by_get(*this, dest, PointArg<internal::point_arg_value_t<A>>(args)...);
    }

    /// See QuadratureIntegrator_fT::map_points for MultiValue results.
    template <typename OT, size_t N, typename... A>
      requires(internal::is_map_result<MultiValue<OT, N>, NT> &&
               internal::accepts_args<NT, KERNEL, ctype, dim, internal::point_arg_value_t<A>...>)
    void map_points(const std::array<PointSpan<OT>, N> &dest, const A &...args) const
    {
      internal::map_points_by_get(*this, dest, PointArg<internal::point_arg_value_t<A>>(args)...);
    }

    /**
     * @brief map() without MPI distribution: computes all of @p coordinates on this rank and waits for the result.
     *
     * @p dest is a pointer, or a std::array of pointers for a kernel returning MultiValue.
     */
    template <typename Dest, typename Coordinates, typename... Args>
      requires internal::is_map_result<internal::map_dest_result_t<Dest>, NT>
    void map(execution_space &, const Dest &dest, const Coordinates &coordinates, const Args &...args)
    {
      // Handles once here rather than per point in get().
      constexpr bool handles = internal::map_takes_handles<NT, KERNEL, ctype, dim, Coordinates, Args...>;
      internal::check_kernel_precision<handles, ctype, Args...>();
      const auto m_args = [&] {
        if constexpr (handles)
          return device::make_tuple(DiFfRG::to_kernel_handle<ctype>(args)...);
        else
          return device::tie(args...);
      }();

      tbb::parallel_for(tbb::blocked_range<uint>(0, coordinates.size()), [&](const tbb::blocked_range<uint> &r) {
        for (uint idx = r.begin(); idx != r.end(); ++idx) {
          const auto dis_idx = coordinates.from_linear_index(idx);
          const auto pos = coordinates.forward(dis_idx);
          internal::map_dest_result_t<Dest> result;
          // nested packs rather than tuple_cat, which would copy every argument per point
          device::apply(
              [&](const auto &...pargs) {
                device::apply([&](const auto &...iargs) { get(result, pargs..., iargs...); }, m_args);
              },
              pos);
          internal::store_result(dest, idx, result);
        }
      });
    }

    /// See QuadratureIntegrator_fT::map; @p dest is a pointer, or a std::array of pointers for a MultiValue kernel.
    template <typename Dest, typename Coordinates, typename... Args>
      requires internal::is_map_result<internal::map_dest_result_t<Dest>, NT>
    auto map(const Dest &dest, const Coordinates &coordinates, const Args &...args)
    {
      internal::scheduled_map<execution_space>(this->integrator_id(), Base::quadrature_volume(), dest, coordinates,
                                               [&](const auto &d, const auto &c) {
                                                 // tbb::parallel_for writes straight into `dest`, so there is nothing
                                                 // to stage.
                                                 internal::run_or_queue_host([this, d, c, args...]() {
                                                   auto sp = execution_space();
                                                   this->map(sp, d, c, args...);
                                                 });
                                                 if (!MapCompletion::deferral_enabled()) MapCompletion::flush();
                                               });
      return execution_space();
    }

    /// See QuadratureIntegrator_fT::map with a braced list of MultiValue destinations.
    template <typename OT, size_t N, typename Coordinates, typename... Args>
      requires internal::is_map_result<MultiValue<OT, N>, NT>
    auto map(OT *const (&dest)[N], const Coordinates &coordinates, const Args &...args)
    {
      return map(std::to_array(dest), coordinates, args...);
    }

  protected:
    using Base::grid_extents;
    using Base::grid_scale;
    using Base::grid_size;
    using Base::grid_start;
    using Base::quadrature_provider;

    using Base::m_n_tail;
    using Base::matsubara_nodes;
    using Base::matsubara_weights;
    using Base::nodes;
    using Base::weights;

    using Base::m_k;
    using Base::T;
  };

} // namespace DiFfRG
