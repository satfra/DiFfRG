
#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>
#include <DiFfRG/physics/integration/optimize.hh>

namespace DiFfRG
{
  /**
   * @brief Sums a kernel over the Matsubara frequencies at temperature \f$T\f$, with no spatial integral.
   *
   * \f[
   *   I = \texttt{constant}(\ldots) + T \sum_{n\in\mathbb{Z}} K(\omega_n, \ldots)\,,
   * \f]
   * where \f$\omega_n = 2\pi n T\f$ are the bosonic Matsubara frequencies; a fermionic kernel shifts its frequency
   * argument itself. At \f$T = 0\f$, or when the sum would need too many modes, the sum is replaced by the integral
   * \f$\int \frac{dq_0}{2\pi}\f$.
   *
   * If the kernel declares `static constexpr bool matsubara_finite_extent = true` (its summand vanishes for
   * \f$|\omega_n| > \Lambda_0 = \sqrt{x_\text{extent}}\, k\f$), the finitely many contributing modes are summed
   * exactly whenever that is cheaper than the approximate rule. \f$x_\text{extent} = 0\f$ disables this.
   *
   * Kernel interface:
   * - get(): `KERNEL::kernel(q0, args...)` and `KERNEL::constant(args...)`
   * - map(): `KERNEL::kernel(q0, pos..., args...)` and `KERNEL::constant(pos..., args...)`
   *
   * @see QuadratureIntegrator for the kernel interface, QuadratureIntegrator_fT for the frequency rule.
   *
   * @tparam dim must be 1
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be summed, providing static `kernel` and `constant`
   * @tparam ExecutionSpace GPU_exec, TBB_exec or KokkosHost_exec
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim == 1)
  class Integrator_fT : public QuadratureIntegrator_fT<1, NT, KERNEL, ExecutionSpace>
  {
    using Base = QuadratureIntegrator_fT<1, NT, KERNEL, ExecutionSpace>;

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
     *   overriding the kernel's traits. Allowing it for a summand that does not vanish above \f$\Lambda_0\f$ truncates
     *   the sum.
     * - `/integration/matsubara_extent_margin` (double): factor on the frequency cutoff \f$\Lambda_0\f$ of the exact
     *   sum.
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
     * Reads the temperature `/physical/T` (default 1) and the settings of apply_matsubara_overrides(), and finds
     * \f$x_\text{extent}\f$ with optimize_x_extent() for `KERNEL::Regulator`.
     */
    Integrator_fT(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_fT(quadrature_provider, config.get_double("/physical/T", 1.0), 1,
                        optimize_x_extent<typename KERNEL::Regulator, dim>(config))
    {
      apply_matsubara_overrides(config);
    }

    /**
     * @param quadrature_provider source of the quadrature rules
     * @param T temperature
     * @param k initial RG scale, which sets the frequency scale of the Matsubara rule until set_k() is called
     * @param x_extent frequency cutoff of a finite-extent kernel, \f$\Lambda_0^2 / k^2\f$; 0 disables the exact sum
     */
    Integrator_fT(QuadratureProvider &quadrature_provider, ctype T = 1, ctype k = 1, ctype x_extent = 0)
        : Base(quadrature_provider, {}, {}, {}, {}, T, k), x_extent(x_extent), k(k)
    {
      Base::set_frequency_cutoff(std::sqrt(this->x_extent) * this->k);
    }

    /// Set \f$x_\text{extent}\f$, so that \f$\Lambda_0 = \sqrt{x_\text{extent}}\, k\f$.
    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_frequency_cutoff(std::sqrt(this->x_extent) * this->k);
    }

    /// Set the RG scale \f$k\f$: moves \f$\Lambda_0\f$ and sets the default energy scale of the Matsubara rule.
    void set_k(ctype k)
    {
      this->k = k;
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
