#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>
#include <DiFfRG/physics/integration/optimize.hh>

namespace DiFfRG
{
  namespace internal
  {
    template <int dim, typename NT, typename KERNEL> class Transform_fT_p2_1ang
    {
    public:
      static constexpr int sdim = dim - 1; // spatial dimension
      using ctype = typename get_type::ctype<NT>;

      // The polar angle (spatial loop vs. external momentum) of a 2-point loop lives in the sdim
      // spatial dimensions and carries the zonal measure (1-c^2)^{(sdim-3)/2} dc, supplied by the
      // Gauss-Jacobi(alpha=beta=(sdim-3)/2) angular quadrature (see the class below). The remaining
      // (sdim-2)-sphere gives S_{sdim-1} = S_d_prec(sdim-1).
      static constexpr ctype int_prefactor = S_d_prec<ctype>(sdim - 1) // remaining solid angle after the polar angle
                                             / powr<sdim>(2 * (ctype)M_PI); // fourier factor

      // Forward the kernel's Matsubara traits: QuadratureIntegrator_fT only sees this adapter, never KERNEL. The
      // measure multiplied in here does not depend on the frequency, so parity and support in q0 are unchanged.
      static constexpr bool matsubara_even = kernel_is_matsubara_even<KERNEL>;
      static constexpr bool matsubara_finite_extent = kernel_has_finite_matsubara_extent<KERNEL>;
      static constexpr bool matsubara_split = kernel_has_matsubara_split<KERNEL>;

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel_finite_extent(const ctype q, const ctype cos, const ctype q0,
                                                                 const T &...t)
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<sdim - 1>(q); // from p integral
        return int_prefactor * int_element * KERNEL::kernel_finite_extent(q, cos, q0, t...);
      }

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel_tail(const ctype q, const ctype cos, const ctype q0, const T &...t)
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<sdim - 1>(q); // from p integral
        return int_prefactor * int_element * KERNEL::kernel_tail(q, cos, q0, t...);
      }

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q, const ctype cos, const ctype q0, const T &...t)
        requires provides_kernel<NT, KERNEL, ctype, 3, T...>
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<sdim - 1>(q); // from p integral
        const NT result = KERNEL::kernel(q, cos, q0, t...);

        return int_prefactor * int_element * result;
      }

      template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION NT constant(const T &...t)
        requires provides_constant<NT, KERNEL, T...>
      {
        return KERNEL::constant(t...);
      }
    };

  } // namespace internal

  /**
   * @brief Integrates a kernel at temperature \f$T\f$ when it depends on the frequency, on \f$|q|\f$ of the spatial
   * momentum and on one spatial angle.
   *
   * In \f$d\f$ spacetime dimensions (so \f$d-1\f$ spatial ones),
   * \f[
   *   I = \texttt{constant}(\ldots) + T \sum_{n\in\mathbb{Z}} \frac{S_{d-2}}{(2\pi)^{d-1}}
   *   \int_{-1}^{1} dc\, (1-c^2)^{\frac{d-4}{2}} \int_0^{q_\text{max}} dq\, q^{d-2}\, K(q, c, \omega_n, \ldots)\,,
   * \f]
   * where \f$\omega_n = 2\pi n T\f$ are the bosonic Matsubara frequencies; a fermionic kernel shifts its frequency
   * argument itself. At \f$T = 0\f$, or when the sum would need too many modes, the sum is replaced by the integral
   * \f$\int \frac{dq_0}{2\pi}\f$.
   * \f$c\f$ is the cosine of the angle between the spatial \f$q\f$ and a fixed direction (usually the external
   * momentum), and \f$S_{d-2} = 2\pi^{(d-2)/2}/\Gamma((d-2)/2)\f$ is the surface of the unit sphere in \f$d-2\f$
   * dimensions. The weight \f$(1-c^2)^{(d-4)/2}\f$ is built into the angular (Gauss-Jacobi) quadrature, so the kernel
   * does not include it.
   * Here \f$q_\text{max} = \sqrt{x_\text{extent}}\, k\f$, with \f$x_\text{extent}\f$ in units of \f$k^2\f$ and the RG
   * scale \f$k\f$ set by set_k() (initially the constructor argument `k`, default 1).
   *
   * If the kernel declares `static constexpr bool matsubara_finite_extent = true` (its summand vanishes for
   * \f$|\omega_n| > q_\text{max}\f$), the finitely many contributing modes are summed exactly whenever that is
   * cheaper than the approximate rule.
   *
   * The `_p2` in the name is historical: the kernel receives \f$|q|\f$, not \f$q^2\f$.
   *
   * Kernel interface:
   * - get(): `KERNEL::kernel(q, c, q0, args...)` and `KERNEL::constant(args...)`
   * - map(): `KERNEL::kernel(q, c, q0, pos..., args...)` and `KERNEL::constant(pos..., args...)`
   *
   * @see QuadratureIntegrator for the kernel interface, Integrator_fT_p2 for an example.
   *
   * @tparam dim spacetime dimension \f$d\f$, 3 to 9
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, providing static `kernel` and `constant`
   * @tparam ExecutionSpace GPU_exec, TBB_exec or KokkosHost_exec
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim >= 3)
  class Integrator_fT_p2_1ang
      : public QuadratureIntegrator_fT<3, NT, internal::Transform_fT_p2_1ang<dim, NT, KERNEL>, ExecutionSpace>
  {
    using Base = QuadratureIntegrator_fT<3, NT, internal::Transform_fT_p2_1ang<dim, NT, KERNEL>, ExecutionSpace>;

    static constexpr int sdim = dim - 1; // spatial dimension

    // Gauss-Jacobi quadrature on [-1,1] with weight (1-c^2)^{(sdim-3)/2} for the spatial polar angle.
    // Dimension-agnostic: reduces to Chebyshev-1 (sdim=2), Legendre (sdim=3), Chebyshev-2 (sdim=4), ...
    static QuadratureType polar_quadrature()
    {
      QuadratureType t(QuadratureType::jacobi);
      t.a = static_cast<double>(-1);
      t.b = static_cast<double>(1);
      t.alpha = (static_cast<double>(sdim) - 3.) / 2.;
      t.beta = t.alpha;
      return t;
    }

  public:
    /**
     * @brief Numerical type to be used for integration tasks e.g. the argument or possible jacobians.
     */
    using ctype = typename get_type::ctype<NT>;
    /**
     * @brief Execution space to be used for the integration, e.g. GPU_exec, TBB_exec.
     */
    using execution_space = ExecutionSpace;

    /**
     * @brief Apply the optional Matsubara settings from the parameter file.
     *
     * - `/integration/force_exact_matsubara_sum` (bool): allow (true) or forbid (false) the exact frequency sum,
     *   overriding the kernel's traits; see set_allow_exact_matsubara_sum(). Allowing it for a summand that does not
     *   vanish above \f$q_\text{max}\f$ truncates the sum.
     * - `/integration/matsubara_extent_margin` (double): factor on the frequency cutoff \f$q_\text{max}\f$ of the
     *   exact sum.
     */
    void apply_matsubara_overrides(const ConfigTree &config)
    {
      if (config.contains("/integration/force_exact_matsubara_sum"))
        Base::set_allow_exact_matsubara_sum(config.get_bool("/integration/force_exact_matsubara_sum", false));
      if (config.contains("/integration/matsubara_extent_margin"))
        Base::set_matsubara_extent_margin(config.get_double("/integration/matsubara_extent_margin", 1.));
    }

    /**
     * @brief Construct from the parameter file.
     *
     * Reads the quadrature orders `/integration/x_order` and `/integration/cos1_order`, the temperature `/physical/T`
     * (default 1) and the settings of apply_matsubara_overrides(), and finds \f$x_\text{extent}\f$ with
     * optimize_x_extent() for `KERNEL::Regulator`.
     */
    Integrator_fT_p2_1ang(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_fT_p2_1ang(quadrature_provider, internal::make_int_grid<2, NT>(config, {"x_order", "cos1_order"}),
                                optimize_x_extent<typename KERNEL::Regulator>(config),
                                config.get_double("/physical/T", 1.0))
    {
      apply_matsubara_overrides(config);
    }

    /**
     * @param quadrature_provider source of the quadrature rules
     * @param grid_size number of quadrature points per spatial axis, in the order of the kernel arguments
     * @param x_extent upper limit of the radial integral, \f$q_\text{max}^2 / k^2\f$
     * @param T temperature
     * @param k initial RG scale
     */
    Integrator_fT_p2_1ang(QuadratureProvider &quadrature_provider, const std::array<size_t, 2> grid_size,
                          ctype x_extent = 2., ctype T = 1, ctype k = 1)
        : Base(quadrature_provider, grid_size, {0, 0.}, {std::sqrt(x_extent) * k, 1.},
               {QuadratureType(QuadratureType::legendre), polar_quadrature()}, T, k),
          x_extent(x_extent), k(k)
    {
      // The summand of a finite-extent kernel vanishes outside the same ball in (q0, q) that cuts the spatial grid, so
      // the exact Matsubara sum is cut at the same radius.
      Base::set_frequency_cutoff(std::sqrt(this->x_extent) * this->k);
    }

    /// Set \f$x_\text{extent}\f$, so that \f$q_\text{max} = \sqrt{x_\text{extent}}\, k\f$.
    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0, 0.}, {std::sqrt(x_extent) * k, 1.});
      Base::set_frequency_cutoff(std::sqrt(this->x_extent) * this->k);
    }

    /// Set the RG scale \f$k\f$: moves \f$q_\text{max}\f$ and sets the default energy scale of the Matsubara rule.
    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0, 0.}, {std::sqrt(x_extent) * k, 1.});
      Base::set_k(k);
      Base::set_frequency_cutoff(std::sqrt(this->x_extent) * this->k);
    }

    /// Set the energy scale the Matsubara rule is built around, replacing \f$k\f$. Zero means: use \f$k\f$.
    void set_typical_E(ctype typical_E) { Base::set_typical_E(typical_E); }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
