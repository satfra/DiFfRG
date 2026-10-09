
#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

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
   * Unlike the Integrator_fT_p2 family, this integrator never sets a frequency cutoff, so the exact
   * (finite) Matsubara sum is never used, even for kernels declaring `matsubara_finite_extent`.
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

    /// Construct from the parameter file. Reads only the temperature, `/physical/T` (default 1).
    Integrator_fT(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_fT(quadrature_provider, config.get_double("/physical/T", 1.0))
    {
    }

    /**
     * @param quadrature_provider source of the quadrature rules
     * @param T temperature
     * @param k initial RG scale, which sets the frequency scale of the Matsubara rule until set_k() is called
     */
    Integrator_fT(QuadratureProvider &quadrature_provider, ctype T = 1, ctype k = 1)
        : Base(quadrature_provider, {}, {}, {}, {}, T, k)
    {
    }

    /// Set the RG scale \f$k\f$, the default energy scale the Matsubara rule is built around.
    void set_k(ctype k) { Base::set_k(k); }

    /// Set the energy scale the Matsubara rule is built around, replacing \f$k\f$. Zero means: use \f$k\f$.
    void set_typical_E(ctype typical_E) { Base::set_typical_E(typical_E); }
  };
} // namespace DiFfRG
