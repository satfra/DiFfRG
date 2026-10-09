#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>

// C++ standard library
#include <vector>

namespace DiFfRG
{
  /**
   * @brief A quadrature rule for (bosonic) Matsubara frequencies, based on the method of Monien [1]. This class
   * provides nodes and weights for the summation
   * \f[
   * T \sum_{n=\in \mathbb{Z}} f(2\pi n T) \approx \sum_{n=1}^{N} w_i (f(x_i) + f(-x_i)) + T f(0)
   * \f]
   *
   * [1] H. Monien, "Gaussian quadrature for sums: a rapidly convergent summation scheme", Math. Comp. 79, 857 (2010).
   * doi:10.1090/S0025-5718-09-02289-3
   *
   * @tparam NT numeric type to be used for all calculations
   */
  template <typename NT> class MatsubaraQuadrature
  {
  public:
    /**
     * @brief Nodes the Monien rule needs to reach past `typical_E` at this temperature.
     *
     * The sizing law and nothing else: no ceiling, no floor, and no choice of rule. Which rule a
     * summand deserves depends on its whole spectrum, and `typical_E` is only the top of it --
     * that decision belongs to the caller, which is QuadratureIntegrator_fT::refresh_matsubara.
     *
     * @param T The temperature. Zero returns zero: the vacuum rule is an integral, not a sum.
     * @param typical_E The LARGEST scale the rule has to cover; the reach is a fixed multiple of it.
     * @param precision_factor Multiplies the reach; the user-facing dial.
     * @param step The step size of considered node sizes (e.g. step=2 implies only even numbers of nodes).
     */
    static int predict_size(const NT T, const NT typical_E = 1., const double precision_factor = 1.,
                            const int step = 2);

    /**
     * @brief Create a new quadrature rule for Matsubara frequencies.
     *
     * @param T The temperature. Zero selects the vacuum rule; this is the only thing that does.
     * @param typical_E The largest scale to cover, which determines the number of nodes.
     * @param step The step size of considered node sizes (e.g. step=2 implies only even numbers of nodes).
     * @param min_size Minimum number of nodes.
     * @param max_size Node ceiling. A rule that wants more is clamped to it, and warns once.
     * @param vacuum_quad_size Size of the vacuum rule, which a T of zero selects.
     * @param precision_factor Multiplies the reach; the user-facing dial.
     *
     * @note These defaults are kept equal to the ConfigTree defaults read by QuadratureProvider
     * (/integration/{min,max}_matsubara_size, vacuum_quad_size, matsubara_precision_factor), so
     * that a directly constructed rule behaves like one a production run would build.
     */
    MatsubaraQuadrature(const NT T, const NT typical_E = 1., const int step = 2, const int min_size = 8,
                        const int max_size = 128, const int vacuum_quad_size = 64, const double precision_factor = 1);

    MatsubaraQuadrature();

    /**
     * @brief Update the quadrature rule with new parameters.
     *
     * @param T  The temperature.
     * @param typical_E A typical energy scale, which determines the number of nodes in the quadrature rule.
     * @param step The step size of considered node sizes (e.g. step=2 implies only even numbers of nodes).
     * @param min_size Minimum number of nodes.
     * @param max_size Node ceiling. A rule that wants more is clamped to it, and warns once.
     * @param vacuum_quad_size Size of the vacuum rule, which a T of zero selects.
     * @param precision_factor Multiplies the reach; the user-facing dial.
     *
     * @note These defaults are kept equal to the ConfigTree defaults read by QuadratureProvider
     * (/integration/{min,max}_matsubara_size, vacuum_quad_size, matsubara_precision_factor), so
     * that a directly constructed rule behaves like one a production run would build.
     */
    void reinit(const NT T, const NT typical_E = 1., const int step = 2, const int min_size = 8,
                const int max_size = 128, const int vacuum_quad_size = 64, const double precision_factor = 1);

    /**
     * @brief Build a rule with an explicitly prescribed number of nodes, bypassing predict_size().
     *
     * This is the entry point for convergence studies: it answers "what accuracy does N nodes
     * buy at this (T, typical_E)", which is the question predict_size() encodes an answer to.
     * A T of zero selects the vacuum (T=0) rule with `size` nodes, exactly as reinit() would.
     *
     * @param size The number of nodes to use. Must be positive.
     * @param T The temperature.
     * @param typical_E A typical energy scale. Only sets the scale of the vacuum rule; the
     * Monien rule's nodes depend on T alone once the size is fixed.
     */
    void reinit_with_size(const int size, const NT T, const NT typical_E = 1.);

    /**
     * @brief Number of positive Matsubara modes strictly inside a frequency cutoff.
     *
     * `2 pi n T <= freq_cutoff`, i.e. `floor(freq_cutoff / (2 pi T))`. Zero is a legitimate
     * answer: below `freq_cutoff = 2 pi T` the whole sum is the zero mode.
     */
    static int modes_below(const NT T, const NT freq_cutoff);

    /**
     * @brief Build the EXACT Matsubara sum for a summand of finite extent in the frequency.
     *
     * When every term carries a `dR/dt` insertion whose argument confines the loop frequency, the
     * summand is identically zero for `|p0| > freq_cutoff` and
     * \f[ T \sum_{n\in\mathbb Z} f(2\pi n T) = T f(0) + T \sum_{n=1}^{N} (f(\omega_n)+f(-\omega_n)) \f]
     * with \f$N = \f$ modes_below() is not an approximation but the sum itself. Nodes are the true
     * frequencies and every weight is \f$T\f$, which is what the caller convention of sum()
     * already expects -- so this is a drop-in replacement for the Monien rule, differing only in
     * how many nodes it needs.
     *
     * Below the crossover this is dramatically cheaper (at k/T ~ 10 it is one node against a
     * 44-node Monien rule) AND exact; above it, it is more expensive, so the caller is expected
     * to pick whichever rule is smaller. The cutoff is a property of the regulator's support, not
     * of the temperature: for a 4D regulator with extent `x_extent` in `q^2/k^2` it is
     * `sqrt(x_extent) * k`.
     *
     * @param T The temperature. Must be positive; at T=0 there is nothing to sum.
     * @param freq_cutoff The frequency beyond which the summand vanishes.
     */
    void reinit_exact_sum(const NT T, const NT freq_cutoff);

    /**
     * @brief Gauss-Legendre over the FINITE interval the summand actually lives on.
     *
     * The third rule, for a summand of finite extent in a regime where the caller has decided the
     * sum may be replaced by an integral. The T=0 rule reaches that regime by the
     * tangent map `p0 = E tan(theta)`, which is built to cover an algebraic `1/p0^2` tail out to
     * ~1e15 E -- and for a summand that dies out past `R = sqrt(x_extent) k` that is a
     * poor use of nodes twice over: only `atan(R/E) / (pi/2) = 57%` of the theta range carries any
     * integrand at all (28 of 64 nodes evaluate an exact zero), and the ones that are wasted are
     * the HEAVY ones, since the tangent map's weights `~ 1/cos^2(theta)` grow towards the end it is
     * throwing away. It also converges badly, being asked to resolve a near-discontinuity at the
     * support edge with the ~36 nodes that remain.
     *
     * Over `[-R, R]` there is no tail worth covering, so plain Gauss-Legendre is both cheaper and
     * more accurate, and the frequency direction becomes just another radial direction: for a 4D
     * regulator `p0` and `|q|` enter the support on the same footing, which is why the natural
     * order for this rule is the SPATIAL `x_order` rather than `vacuum_quad_size`.
     *
     * `R` is not a true support boundary but the radius `optimize_x_extent()` converged to, so
     * cutting there is a systematic error in its own right. Measured on the assembled 3+1D
     * integrand (Part 13 of the convergence study): 3.5e-10 for `PolynomialExp<8>`, 3.0e-10 for
     * `Exponential<2>`, at the shipped `x_extent_tolerance`. That is well below the aliasing error
     * this rule exists to avoid, but it is not zero.
     *
     * Caller convention as in sum(): \f$\sum_i w_i (f(x_i)+f(-x_i)) = \frac{1}{2\pi}\int_{-R}^{R} f\,dp_0\f$,
     * i.e. \f$x_i = R u_i\f$ and \f$w_i = R v_i / 2\pi\f$ for Gauss-Legendre nodes/weights
     * \f$(u_i, v_i)\f$ on [0,1]. No zero mode: like the T=0 rule this integrates rather than sums.
     *
     * @param cutoff The frequency R beyond which the summand vanishes. Must be positive.
     * @param gl_nodes Gauss-Legendre nodes on [0,1] -- passed in so the (cached) rule is not
     * rebuilt for every cutoff; only the scaling depends on R.
     * @param gl_weights The matching weights, summing to 1.
     */
    void reinit_finite_interval(const NT cutoff, const Kokkos::View<const NT *, CPU_memory> gl_nodes,
                                const Kokkos::View<const NT *, CPU_memory> gl_weights);

    /**
     * @brief Get the size of the quadrature rule, NOT counting the zero mode.
     */
    size_t size() const;

    /**
     * @brief Size of the node list returned by sum_nodes()/sum_weights(): size() plus the zero
     * mode, where there is one (i.e. everywhere but the T=0 rule).
     */
    size_t sum_size() const;

    /**
     * @brief Get the temperature of the quadrature rule.
     */
    NT get_T() const;

    /**
     * @brief Get the typical energy scale of the quadrature rule.
     */
    NT get_typical_E() const;

    /**
     * @brief Compute a matsubara sum of a given function.
     *
     * @tparam F The function type.
     * @param f The function to be summed. Must have signature NT f(NT x).
     * @return NT The Matsubara sum of the function.
     */
    template <typename F> auto sum(const F &f) const
    {
      auto sum = T * f(static_cast<NT>(0));
      for (int i = 0; i < m_size; ++i)
        sum += host_weights[i] * (f(host_nodes[i]) + f(-host_nodes[i]));
      return sum;
    }

    /**
     * @brief The positive-frequency nodes, without the zero mode. Length size().
     */
    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> nodes() const
    {
      return Kokkos::subview(raw_nodes<MemorySpace>(), Kokkos::make_pair(size_t(0), size_t(m_size)));
    }

    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> weights() const
    {
      return Kokkos::subview(raw_weights<MemorySpace>(), Kokkos::make_pair(size_t(0), size_t(m_size)));
    }

    /**
     * @brief Node list for a device-side sum: the positive frequencies followed by the zero mode.
     *
     * The zero mode is appended as a node like any other -- `x = 0` with weight `T/2` -- so that
     * the `w * (f(+x) + f(-x))` body that evaluates every other node reproduces `T f(0)` exactly,
     * with no branch. That matters twice over: a `idx == 0 ? T*f(0) : 0` guard around a kernel
     * call costs the whole warp a full extra kernel evaluation for one active lane, and the exact
     * exact sum can legitimately have ZERO positive modes in the deep IR, where a rule that carries
     * its zero mode outside the node list would integrate to nothing at all.
     *
     * Empty for the T=0 rule, where there is no zero mode to add (it has weight zero).
     */
    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> sum_nodes() const
    {
      return Kokkos::subview(raw_nodes<MemorySpace>(), Kokkos::make_pair(size_t(0), sum_size()));
    }

    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> sum_weights() const
    {
      return Kokkos::subview(raw_weights<MemorySpace>(), Kokkos::make_pair(size_t(0), sum_size()));
    }

  private:
    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> raw_nodes() const
    {
      if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultExecutionSpace::memory_space>) {
        return device_nodes;
      } else if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultHostExecutionSpace::memory_space>) {
        return host_nodes;
      } else {
        throw std::runtime_error("Invalid memory space");
      }
    }

    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> raw_weights() const
    {
      if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultExecutionSpace::memory_space>) {
        return device_weights;
      } else if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultHostExecutionSpace::memory_space>) {
        return host_weights;
      } else {
        throw std::runtime_error("Invalid memory space");
      }
    }

    NT T, typical_E;

    Kokkos::View<NT *, GPU_memory> device_nodes;
    Kokkos::View<NT *, GPU_memory> device_weights;

    Kokkos::View<NT *, CPU_memory> host_nodes;
    Kokkos::View<NT *, CPU_memory> host_weights;

    /**
     * @brief The number of nodes in the quadrature rule.
     */
    int m_size;

    /**
     * @brief Construct the Monien rule for the current m_size and T.
     */
    void build_monien();

    /**
     * @brief Construct a quadrature rule for T=0.
     */
    void reinit_0();

    /**
     * @brief Store x/w plus the appended zero mode; see sum_nodes(). x and w have m_size entries.
     */
    void write_data(const std::vector<NT> &x, const std::vector<NT> &w);

    int vacuum_quad_size;
    double precision_factor;
  };
} // namespace DiFfRG