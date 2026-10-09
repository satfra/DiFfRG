// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>

// Standard library
#include <cmath>
#include <string>

namespace DiFfRG
{
  namespace
  {
    /**
     * @brief How far, in units of typical_E, the Monien rule is asked to reach in frequency.
     *
     * The Monien rule with N nodes resolves frequencies up to x_max ~ pi^2 T N^2 / 4, so this
     * constant is what sets N.
     *
     * It is NOT merely a tail multiplier that a compactly supported summand could do without.
     * Measured on the assembled 3+1D integrand of a 4D-regulated kernel -- which vanishes
     * identically past R = sqrt(x_extent) k -- against the exact sum
     * (tests/common/quadrature/matsubara_convergence.cc, `[.study]`, Part 13, PolynomialExp<8>,
     * m^2 = -0.9 k^2; cell = nodes -> relative error):
     *
     *   k/T     reach R (=1.23k)      reach 20k        reach 100k       reach 400k
     *    30      8 -> 4.1e-05      20 -> 1.8e-16    40 -> 1.8e-16    74 -> 1.8e-16
     *   100     12 -> 3.7e-03      34 -> 2.1e-15    68 -> 4.4e-15   128 -> 1.8e-15
     *   300     18 -> 2.5e-02      54 -> 1.5e-08   116 -> 4.4e-15   128 -> 1.4e-15
     *
     * Sizing the rule to just SPAN the support fails by four to seven orders and gets worse with
     * k/T: at T = k/300 the summand carries 58 significant Matsubara modes and an 18-node rule
     * cannot reproduce them. The reach is what buys node DENSITY, not just extent, so it applies
     * to a compact summand exactly as it does to an algebraic tail.
     *
     * 200 was picked on the real QCD_Nf2/avatars_3Dquark kernels, driven statically across the
     * handover and compared against a converged reference (worst relative error over k/T < 46):
     *
     *   flow      reach 100   reach 200   reach 400   nodes @200 / @400
     *   ZA3        2.6e-15     1.2e-15     2.1e-15         77 / 103
     *   ZAqbq1     7.8e-05     1.1e-05     4.5e-06         77 / 103
     *   Zq         3.6e-06     4.6e-07     1.3e-07         77 / 103
     *
     * The gluon and ghost flows are at machine precision for every reach; the two quark flows are
     * not, because the 3D quark regulator leaves a genuine algebraic tail in p0 with a mass scale
     * above k that the model does not report through set_typical_E. 200 is the smallest reach that
     * stays below what the shipped code achieves on those flows while costing fewer nodes than it.
     *
     * The predecessor 400 was calibrated on Part 8, whose stressor is a mode two decades ABOVE
     * typical_E -- i.e. on the sizing scale being wrong. With the heaviest scale reported honestly,
     * Part 8b finds every reach from 20 to 400 indistinguishable on that stressor; the residual
     * 8.4e-10 at E/T = 78 is the max_matsubara_size clamp, which all of them hit. A model that
     * carries a scale far above k should say so through set_typical_E, which now survives set_k.
     *
     * Reach also decides where the rule stops fitting max_matsubara_size and the vacuum rule takes
     * over: k/T = 187 at 200, against 93 at 400. It does NOT decide the accuracy of that handover,
     * which is the 64-node tangent map's own quadrature error, 2.6e-8 on the assembled 3+1D
     * integrand (Part 5) -- raise /integration/vacuum_quad_size to improve it (96 nodes: 4.4e-10).
     *
     * The user-facing dial is /integration/matsubara_precision_factor, which multiplies this;
     * setting it to 2 restores the pre-2026-08 reach of 400.
     */
    constexpr double monien_reach = 200.;
  } // namespace

  template <typename NT>
  int MatsubaraQuadrature<NT>::predict_size(const NT T, const NT typical_E, const double precision_factor,
                                            const int step)
  {
    if (!(std::fabs(T) > NT(0))) return 0; // no sum to size; the T = 0 rule is an integral

    const NT E_max = monien_reach * precision_factor * std::fabs(typical_E);
    const int size = 5 + int(std::sqrt(4. * E_max / (M_PI * M_PI * std::fabs(T))));
    return (int)std::ceil(size / (double)step) * step;
  }

  template <typename NT>
  MatsubaraQuadrature<NT>::MatsubaraQuadrature(const NT T, const NT typical_E, const int step, const int min_size,
                                               const int max_size, const int vacuum_quad_size,
                                               const double precision_factor)
  {
    reinit(T, typical_E, step, min_size, max_size, vacuum_quad_size, precision_factor);
  }

  // Fully initialised: predict_size() and the rule builders read precision_factor, T and
  // typical_E, so a default-constructed object that leaves them indeterminate answers questions
  // with garbage rather than with the shipped defaults. These match the ConfigTree defaults, as
  // the reinit() overloads' do.
  template <typename NT>
  MatsubaraQuadrature<NT>::MatsubaraQuadrature()
      : T(0), typical_E(1), m_size(0), vacuum_quad_size(64), precision_factor(1)
  {
  }

  template <typename NT>
  void MatsubaraQuadrature<NT>::reinit(const NT T, const NT typical_E, const int step, const int min_size,
                                       const int max_size, const int vacuum_quad_size, const double precision_factor)
  {
    if (precision_factor <= 0)
      this->precision_factor = 1;
    else
      this->precision_factor = precision_factor;
    // Warn on the *effective* factor: a non-positive argument has just been coerced to 1, so
    // testing the argument here warned on every default-constructed QuadratureProvider.
    if (this->precision_factor < 1e-1)
      std::cerr
          << "MatsubaraQuadrature: precision_factor is very small (< 1e-1), which may lead to inaccurate results.";

    // This is ludicrously low, but let's cut somewhere
    if (vacuum_quad_size <= 16)
      this->vacuum_quad_size = 16;
    else
      this->vacuum_quad_size = vacuum_quad_size;

    if (max_size < min_size) throw std::invalid_argument("MatsubaraQuadrature: max_size must be larger than min_size.");

    this->T = T;
    this->typical_E = typical_E;

    // T = 0 is the caller's way of asking for the vacuum rule. Nothing else selects it: which
    // rule a summand deserves is a question about the summand's spectrum, and the only scale
    // handed in here is the one that sizes the reach. See QuadratureIntegrator_fT::refresh_matsubara.
    if (is_close(T, NT{})) {
      m_size = this->vacuum_quad_size;
      reinit_0();
      return;
    }

    const int wanted = predict_size(T, typical_E, this->precision_factor, step);
    // Clamping violates the resolution law E_max ~ pi^2 N^2 T / 4 -- the rule then reaches
    // (max_size/wanted)^2 less far in frequency than it was asked to, with no other symptom.
    //
    // QuadratureIntegrator_fT never gets here: it asks for the vacuum rule as soon as the Monien
    // rule stops fitting the budget, which is the same test. This is the floor for a caller that
    // reaches the storage directly and so makes no such choice. Say so once.
    if (wanted > max_size) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        std::cerr << "MatsubaraQuadrature: requested size " << wanted << " exceeds max_matsubara_size " << max_size
                  << " (T = " << T << ", typical_E = " << typical_E << "); the rule reaches "
                  << (double(wanted) / max_size) * (double(wanted) / max_size)
                  << "x less far in frequency than the sizing asked for. Raise max_matsubara_size, or lower "
                     "matsubara_precision_factor if that reach is not needed. (warned once)\n";
      }
    }
    m_size = std::max(min_size, std::min(max_size, wanted));

    build_monien();
  }

  template <typename NT> void MatsubaraQuadrature<NT>::reinit_with_size(const int size, const NT T, const NT typical_E)
  {
    if (size <= 0) throw std::invalid_argument("MatsubaraQuadrature: size must be positive.");

    this->T = T;
    this->typical_E = typical_E;
    m_size = size;

    if (is_close(T, NT{})) {
      reinit_0();
      return;
    }
    build_monien();
  }

  template <typename NT> int MatsubaraQuadrature<NT>::modes_below(const NT T, const NT freq_cutoff)
  {
    if (!(T > NT(0)) || !(freq_cutoff > NT(0))) return 0;
    const double n = std::floor(double(std::fabs(freq_cutoff)) / (2. * M_PI * double(std::fabs(T))));
    // Nothing here bounds the count: a caller that hands in a cutoff far above 2 pi T is asking
    // for a sum that large, and it is the caller's job (see QuadratureIntegrator_fT) to prefer
    // the Monien rule once it is the cheaper of the two.
    return n < 0. ? 0 : int(n);
  }

  template <typename NT> void MatsubaraQuadrature<NT>::reinit_exact_sum(const NT T, const NT freq_cutoff)
  {
    if (!(T > NT(0))) throw std::invalid_argument("MatsubaraQuadrature: reinit_exact_sum needs a positive T.");

    this->T = T;
    // Not a "typical energy" in the predict_size sense -- it is the hard support boundary. Stored
    // here so get_typical_E() still identifies which rule this is, and so write_data's view names
    // stay distinguishable.
    this->typical_E = freq_cutoff;
    m_size = modes_below(T, freq_cutoff);

    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);
    for (int i = 0; i < m_size; ++i) {
      x[i] = NT(2. * M_PI * double(T) * double(i + 1));
      w[i] = T;
    }

    write_data(x, w);
  }

  template <typename NT>
  void MatsubaraQuadrature<NT>::reinit_finite_interval(const NT cutoff,
                                                       const Kokkos::View<const NT *, CPU_memory> gl_nodes,
                                                       const Kokkos::View<const NT *, CPU_memory> gl_weights)
  {
    if (!(cutoff > NT(0)))
      throw std::invalid_argument("MatsubaraQuadrature: reinit_finite_interval needs a positive cutoff.");

    // Zero, like the T=0 rule: this replaces the sum by an integral, so there is no separate zero
    // mode and sum_size() == size().
    this->T = 0;
    this->typical_E = cutoff;
    m_size = (int)gl_nodes.size();

    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);
    for (int i = 0; i < m_size; ++i) {
      x[i] = cutoff * gl_nodes[i];
      w[i] = cutoff * gl_weights[i] / NT(2 * M_PI);
    }

    write_data(x, w);
  }

  template <typename NT> void MatsubaraQuadrature<NT>::build_monien()
  {
    // construct the recurrence relation for the quadrature rule from [1]
    std::vector<NT> a(m_size, 0.);
    std::vector<NT> b(m_size, 0.);

    for (int j = 0; j < m_size; ++j) {
      const double j1 = j + 1;
      a[j] = 2 * powr<2>(M_PI) / (4 * j + 1) / (4 * j + 5);
      b[j] = powr<4>(M_PI) / ((4 * j1 - 1) * (4 * j1 + 3)) / powr<2>(4 * j1 + 1);
    }
    a[0] = powr<2>(M_PI) / 15.;

    const NT mu0 = powr<2>(M_PI) / 6.;

    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);

    // compute the nodes and weights of the quadrature rule
    make_quadrature(a, b, mu0, x, w);

    // normalize the weights and scale the nodes
    for (int i = 0; i < m_size; ++i) {
      w[i] = T * w[i] / x[i];
      x[i] = 2. * M_PI * T / std::sqrt(x[i]);
    }

    write_data(x, w);
  }

  /**
   * @brief The vacuum (T=0) rule for the frequency integral.
   *
   * At T=0 the Matsubara sum degenerates to \f$\int dp_0/(2\pi) f(p_0)\f$. Substituting
   * \f$p_0 = E \tan\theta\f$, \f$\theta\in[0,\pi/2)\f$, maps the canonical propagator structure
   * to a constant: for \f$f = 1/(p_0^2+E^2)\f$ the transformed integrand is exactly \f$1/E\f$,
   * so Gauss-Legendre in \f$\theta\f$ is near-exact. An algebraic \f$c/p_0^2\f$ tail maps to the
   * finite limit \f$c/E\f$ at \f$\theta\to\pi/2\f$ and is thus *covered* by the quadrature
   * rather than truncated.
   *
   * The previous construction (2/3 of the nodes linear on [0,3E], 1/3 log-spaced on
   * [3E, 1e11 E]) truncated the tail at 1e11 E, which floored the relative error of a
   * 1/p0^2 tail at ~6e-12 regardless of the node count -- going from 81 to 513 nodes did not
   * improve it at all. The tangent map has no such floor.
   *
   * Caller convention (see sum()): \f$\sum_i w_i (f(x_i)+f(-x_i)) = \int dp_0/(2\pi) f\f$, so
   * for even f we need \f$\sum_i w_i f(x_i) = (1/2\pi)\int_0^\infty f\,dp_0\f$, giving
   * \f[ x_i = E\tan\theta_i, \qquad w_i = \frac{E\,w^{GL}_i}{4\cos^2\theta_i}. \f]
   * Everything is linear in E, i.e. this is one dimensionless table times a single scale.
   */
  template <typename NT> void MatsubaraQuadrature<NT>::reinit_0()
  {
    this->T = 0;
    m_size = abs(m_size);
    if (is_close(typical_E, NT(0))) this->typical_E = 1.;

    // gauss-legendre nodes/weights on [0, 1]
    Quadrature<NT> quad(m_size, QuadratureType::legendre);
    const auto gl_nodes = quad.template nodes<CPU_memory>();
    const auto gl_weights = quad.template weights<CPU_memory>();

    std::vector<NT> x(m_size, 0.);
    std::vector<NT> w(m_size, 0.);

    using std::abs, std::cos, std::sin;
    const long double E = abs((long double)typical_E);
    constexpr long double half_pi = (long double)M_PI / 2.0L;

    for (int i = 0; i < m_size; ++i) {
      // Parameterise by the complement delta = pi/2 - theta: the large nodes are the ones
      // that matter for the tail, and cot(delta) is far better conditioned there than
      // tan(theta) with theta -> pi/2.
      const long double delta = half_pi * (1.0L - (long double)gl_nodes[i]);
      const long double s = sin(delta);

      x[i] = (NT)(E * cos(delta) / s);
      w[i] = (NT)(E * (long double)gl_weights[i] / (4.0L * s * s));
    }

    write_data(x, w);
  }

  template <typename NT> void MatsubaraQuadrature<NT>::write_data(const std::vector<NT> &x, const std::vector<NT> &w)
  {
    std::string name = "matsubara_T" + std::to_string(T) + "_typicalE" + std::to_string(typical_E);

    // One slot past the rule itself, holding the zero mode; see sum_nodes(). Allocated
    // unconditionally so that nodes()/sum_nodes() are two subviews of one buffer -- the T=0 rule
    // simply never exposes the extra slot.
    const size_t alloc = size_t(m_size) + 1;

    // copy nodes and weights to std::vector
    device_nodes = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_nodes_" + name + "_" + std::to_string(m_size), alloc);
    device_weights = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_weights_" + name + "_" + std::to_string(m_size), alloc);

    // create a host mirror view
    auto host_mirror_nodes = Kokkos::create_mirror_view(device_nodes);
    auto host_mirror_weights = Kokkos::create_mirror_view(device_weights);

    // copy data from gsl to host mirror view
    for (int i = 0; i < m_size; ++i) {
      host_mirror_nodes(i) = x[i];
      host_mirror_weights(i) = w[i];
    }
    // The zero mode as an ordinary node: a summand evaluated as w * (f(+x) + f(-x)) reproduces
    // T * f(0) exactly at x = 0, w = T/2 (the division by two is exact in binary). Zero-weight and
    // harmless for the T=0 rule, which does not expose it anyway.
    host_mirror_nodes(m_size) = NT(0);
    host_mirror_weights(m_size) = T / NT(2);

    // copy data from host mirror view to device view
    Kokkos::deep_copy(device_nodes, host_mirror_nodes);
    Kokkos::deep_copy(device_weights, host_mirror_weights);

    // copy data from host mirror view to host view
    host_nodes = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_nodes_" + name + "_" + std::to_string(m_size), alloc);
    host_weights = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_weights_" + name + "_" + std::to_string(m_size), alloc);
    Kokkos::deep_copy(host_nodes, host_mirror_nodes);
    Kokkos::deep_copy(host_weights, host_mirror_weights);
  }

  template <typename NT> size_t MatsubaraQuadrature<NT>::size() const { return m_size; }
  template <typename NT> size_t MatsubaraQuadrature<NT>::sum_size() const
  {
    // The T=0 rule integrates rather than sums: its "zero mode" has weight zero, so exposing it
    // would only buy a wasted kernel evaluation per spatial point.
    return size_t(m_size) + (is_close(T, NT{}) ? 0u : 1u);
  }
  template <typename NT> NT MatsubaraQuadrature<NT>::get_T() const { return T; }
  template <typename NT> NT MatsubaraQuadrature<NT>::get_typical_E() const { return typical_E; }

  // explicit instantiation
  template class MatsubaraQuadrature<double>;
  template class MatsubaraQuadrature<float>;
} // namespace DiFfRG