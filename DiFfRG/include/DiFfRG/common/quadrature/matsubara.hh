#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/math.hh>

// C++ standard library
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Nodes and weights for bosonic Matsubara sums at temperature \f$T\f$.
   *
   * All rules share one convention: with positive nodes \f$x_i\f$ and weights \f$w_i\f$, \f$i = 1 \dots N\f$,
   * \f[
   *   T \sum_{n \in \mathbb{Z}} f(2\pi n T) \approx T f(0) + \sum_{i=1}^{N} w_i \big(f(x_i) + f(-x_i)\big)\,,
   * \f]
   * the weights include the factor \f$T\f$. sum() evaluates this on the host. For kernels, sum_nodes() and
   * sum_weights() append the zero mode as a node \f$x = 0\f$ with weight \f$T/2\f$, so that the sum is just
   * \f$\sum_i w_i (f(x_i) + f(-x_i))\f$ over that list.
   *
   * Which rule is built:
   * - \f$T > 0\f$, reinit(): the Gauss rule for sums of Monien [1]. Its \f$N\f$ nodes resolve frequencies up to
   *   about \f$\pi^2 T N^2/4\f$; \f$N\f$ is chosen (predict_size()) so that this reaches \f$200\, c\, E\f$, with
   *   \f$E\f$ = `typical_E` the largest scale of the summand and \f$c\f$ = `precision_factor`, then rounded up to a
   *   multiple of `step` and clamped to [`min_size`, `max_size`].
   * - \f$T = 0\f$: the sum becomes \f$\int dp_0/(2\pi)\, f(p_0)\f$, and the same formula holds without the zero mode.
   *   Nodes are \f$x_i = E \tan\theta_i\f$ with Gauss-Legendre points \f$\theta_i \in [0, \pi/2)\f$, weights
   *   \f$w_i = E\, w^\text{GL}_i / (4\cos^2\theta_i)\f$ (\f$w^\text{GL}_i\f$ the Gauss-Legendre weights on
   *   \f$[0, 1]\f$). This integrates \f$1/(p_0^2 + E^2)\f$ exactly and covers \f$1/p_0^2\f$ tails; `vacuum_quad_size`
   *   nodes (at least 16).
   * - reinit_exact_sum(): the exact sum over all modes below a cutoff, for summands that vanish beyond it.
   * - reinit_finite_interval(): Gauss-Legendre on \f$[-P, P]\f$, replacing the sum by an integral.
   *
   * @code
   * MatsubaraQuadrature<double> rule(T, k);
   * const double s = rule.sum([&](double p0) { return 1. / (p0 * p0 + m2); }); // = coth(sqrt(m2)/(2T))/(2 sqrt(m2))
   * @endcode
   *
   * [1] H. Monien, "Gaussian quadrature for sums: a rapidly convergent summation scheme", Math. Comp. 79, 857 (2010).
   *
   * @tparam NT the number type of nodes and weights.
   */
  template <typename NT> class MatsubaraQuadrature
  {
  public:
    /**
     * @brief Number of nodes the Monien rule needs: \f$N = 5 + \lfloor\sqrt{4 \cdot 200\, c E / (\pi^2 T)}\rfloor\f$,
     * rounded up to a multiple of `step`.
     *
     * No clamping to a minimum or maximum. Note the argument order differs from the constructor.
     *
     * @param T the temperature; 0 returns 0.
     * @param typical_E the largest scale \f$E\f$ the summand contains.
     * @param precision_factor the factor \f$c\f$ multiplying the reach.
     * @param step only multiples of `step` are used (e.g. 2: even sizes).
     */
    static int predict_size(const NT T, const NT typical_E = 1., const double precision_factor = 1.,
                            const int step = 2);

    /**
     * @brief Build the rule for temperature `T`; see reinit().
     */
    MatsubaraQuadrature(const NT T, const NT typical_E = 1., const int step = 2, const int min_size = 8,
                        const int max_size = 128, const int vacuum_quad_size = 64, const double precision_factor = 1);

    /// An empty rule at \f$T = 0\f$; call reinit() before use.
    MatsubaraQuadrature();

    /**
     * @brief Rebuild the rule: the Monien rule for \f$T > 0\f$, the vacuum rule for \f$T = 0\f$.
     *
     * The defaults equal those of the parameter file (`/integration/min_matsubara_size`, `max_matsubara_size`,
     * `vacuum_quad_size`, `matsubara_precision_factor`).
     *
     * @param T the temperature; \f$T = 0\f$ (to machine precision) selects the vacuum rule.
     * @param typical_E the largest scale \f$E\f$ of the summand; sets the number of nodes (or, at \f$T = 0\f$, the
     * scale of the tangent map).
     * @param step only multiples of `step` nodes are used.
     * @param min_size minimal number of nodes.
     * @param max_size maximal number of nodes; a larger request is clamped (with a one-time warning), which reduces
     * the frequency reach.
     * @param vacuum_quad_size number of nodes of the vacuum rule; values below 16 are raised to 16.
     * @param precision_factor the factor \f$c\f$ multiplying the reach; non-positive values mean 1.
     */
    void reinit(const NT T, const NT typical_E = 1., const int step = 2, const int min_size = 8,
                const int max_size = 128, const int vacuum_quad_size = 64, const double precision_factor = 1);

    /**
     * @brief Build the rule with exactly `size` nodes instead of predict_size(), e.g. for convergence tests.
     *
     * @param size the number of nodes, positive.
     * @param T the temperature; 0 selects the vacuum rule with `size` nodes.
     * @param typical_E only sets the scale of the vacuum rule; the Monien rule depends on \f$T\f$ alone once its
     * size is fixed.
     */
    void reinit_with_size(const int size, const NT T, const NT typical_E = 1.);

    /**
     * @brief Number of positive Matsubara frequencies \f$2\pi n T \leq\f$ `freq_cutoff`, i.e.
     * \f$\lfloor \mathrm{freq\_cutoff} / (2\pi T) \rfloor\f$.
     *
     * May be 0, in which case only the zero mode contributes.
     */
    static int modes_below(const NT T, const NT freq_cutoff);

    /**
     * @brief The exact Matsubara sum for a summand that vanishes for \f$|p_0| > \omega_c\f$ = `freq_cutoff`.
     *
     * Then
     * \f[ T \sum_{n\in\mathbb Z} f(2\pi n T) = T f(0) + T \sum_{n=1}^{N} \big(f(2\pi n T)+f(-2\pi n T)\big) \f]
     * with \f$N\f$ = modes_below(), i.e. nodes \f$2\pi n T\f$ and weights \f$T\f$. With \f$k\f$ the RG scale, at low
     * \f$k/T\f$ this needs far fewer nodes than the Monien rule; at high \f$k/T\f$ more, so pick the smaller one.
     * Example: a summand that carries \f$\partial_t R_k(p^2)\f$ of a 4D regulator with support
     * \f$p^2 < x_\text{extent} k^2\f$ has \f$\omega_c = \sqrt{x_\text{extent}}\, k\f$.
     *
     * @param T the temperature, positive.
     * @param freq_cutoff the frequency \f$\omega_c\f$ beyond which the summand vanishes.
     */
    void reinit_exact_sum(const NT T, const NT freq_cutoff);

    /**
     * @brief Replace the sum by \f$\frac{1}{2\pi}\int_{-P}^{P} dp_0\, f(p_0)\f$ for a summand that vanishes for
     * \f$|p_0| > P\f$, using Gauss-Legendre on \f$[0, P]\f$.
     *
     * Nodes \f$x_i = P u_i\f$ and weights \f$w_i = P v_i/(2\pi)\f$ for Gauss-Legendre points \f$(u_i, v_i)\f$ on
     * \f$[0, 1]\f$; no zero mode. For such summands this is cheaper and more accurate than the \f$T = 0\f$ tangent
     * map, which spends nodes on a tail that is not there. If \f$P\f$ is a numerically determined support radius
     * (e.g. from optimize_x_extent()), cutting there is a small systematic error.
     *
     * @param cutoff \f$P\f$, positive.
     * @param gl_nodes Gauss-Legendre nodes \f$u_i\f$ on \f$[0, 1]\f$.
     * @param gl_weights the matching weights \f$v_i\f$, summing to 1.
     */
    void reinit_finite_interval(const NT cutoff, const Kokkos::View<const NT *, CPU_memory> gl_nodes,
                                const Kokkos::View<const NT *, CPU_memory> gl_weights);

    /// Number of nodes \f$N\f$, not counting the zero mode.
    size_t size() const;

    /**
     * @brief Length of sum_nodes(): size() plus one for the zero mode, except for the \f$T = 0\f$ and
     * finite-interval rules, which have none.
     */
    size_t sum_size() const;

    /// The temperature \f$T\f$ (0 after reinit_finite_interval()).
    NT get_T() const;

    /// The scale \f$E\f$ the rule was built for; the cutoff after reinit_exact_sum() or reinit_finite_interval().
    NT get_typical_E() const;

    /**
     * @brief \f$T f(0) + \sum_i w_i (f(x_i) + f(-x_i))\f$, evaluated on the host.
     *
     * @param f callable as `f(NT p0)`.
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

    /// The weights \f$w_i\f$ belonging to nodes(). Length size().
    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> weights() const
    {
      return Kokkos::subview(raw_weights<MemorySpace>(), Kokkos::make_pair(size_t(0), size_t(m_size)));
    }

    /**
     * @brief Node list for a device-side sum: the positive frequencies followed by the zero mode.
     *
     * The zero mode is appended as a node \f$x = 0\f$ with weight \f$T/2\f$, so that
     * \f$\sum_i w_i (f(x_i) + f(-x_i))\f$ over this list is the whole sum, without a special case for the zero mode.
     * Length sum_size(); equal to nodes() for the rules without a zero mode (\f$T = 0\f$, finite interval).
     */
    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> sum_nodes() const
    {
      return Kokkos::subview(raw_nodes<MemorySpace>(), Kokkos::make_pair(size_t(0), sum_size()));
    }

    /// The weights belonging to sum_nodes().
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