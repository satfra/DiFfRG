#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/optimize.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  namespace internal
  {
    template <typename NT, typename KERNEL> class Transform_p2_4D_3ang
    {
    public:
      using ctype = typename get_type::ctype<NT>;

      static constexpr ctype int_prefactor = powr<-4>((ctype)2 * M_PI); // fourier factor

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q, const ctype cos1, const ctype cos2, const ctype phi,
                                                   const T &...t)
        requires provides_kernel<NT, KERNEL, ctype, 4, T...>
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<4 - 1>(q); // from p integral

        // sqrt(1. - powr<2>(cos1)); // the cos1 integral jacobian is already included
        // in the chebyshev2 quadrature

        const NT result = KERNEL::kernel(q, cos1, cos2, phi, t...);

        return int_prefactor * int_element * result;
      }

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT constant(const T &...t)
        requires provides_constant<NT, KERNEL, T...>
      {
        return KERNEL::constant(t...);
      }
    };

  } // namespace internal

  /**
   * @brief Integrates a kernel over a 4-dimensional momentum when it depends on \f$|q|\f$ and all three angles.
   *
   * \f[
   *   I = \texttt{constant}(\ldots) + \frac{1}{(2\pi)^4} \int_{-1}^{1} dc_1\, \sqrt{1-c_1^2}
   *   \int_{-1}^{1} dc_2 \int_0^{2\pi} d\phi \int_0^{q_\text{max}} dq\, q^3\, K(q, c_1, c_2, \phi, \ldots)\,,
   * \f]
   * where \f$c_1 = \cos\theta_1\f$, \f$c_2 = \cos\theta_2\f$ and \f$\phi\f$ are the angles of 4-dimensional spherical
   * coordinates, \f$\vec q = q\,(c_1,\, \sqrt{1-c_1^2}\, c_2,\, \sqrt{1-c_1^2}\sqrt{1-c_2^2}\cos\phi,\,
   * \sqrt{1-c_1^2}\sqrt{1-c_2^2}\sin\phi)\f$. Here
   * \f$q_\text{max} = \sqrt{x_\text{extent}}\, k\f$, with \f$x_\text{extent}\f$ in units of \f$k^2\f$ and the RG scale
   * \f$k\f$ set by set_k() (initially 1). The weight \f$\sqrt{1-c_1^2}\f$ is built into the \f$c_1\f$ (Gauss-Chebyshev)
   * quadrature, so the kernel does not include it; \f$\phi\f$ uses the trapezoidal rule, which is very accurate for
   * periodic integrands. The `_p2` in the name is historical: the kernel receives \f$|q|\f$, not \f$q^2\f$.
   *
   * Kernel interface:
   * - get(): `KERNEL::kernel(q, c1, c2, phi, args...)` and `KERNEL::constant(args...)`
   * - map(): `KERNEL::kernel(q, c1, c2, phi, pos..., args...)` and `KERNEL::constant(pos..., args...)`
   *
   * @see QuadratureIntegrator for the kernel interface, Integrator_p2 for an example.
   *
   * @tparam dim momentum-space dimension, must be 4
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, providing static `kernel` and `constant`
   * @tparam ExecutionSpace GPU_exec, TBB_exec or KokkosHost_exec
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim == 4)
  class Integrator_p2_4D_3ang
      : public QuadratureIntegrator<4, NT, internal::Transform_p2_4D_3ang<NT, KERNEL>, ExecutionSpace>
  {
    using Base = QuadratureIntegrator<4, NT, internal::Transform_p2_4D_3ang<NT, KERNEL>, ExecutionSpace>;

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
     * @brief Construct from the parameter file.
     *
     * Reads the quadrature orders from `/integration/x_order`, `/integration/cos1_order`, `/integration/cos2_order`
     * and `/integration/phi_order`, and finds \f$x_\text{extent}\f$ with optimize_x_extent() for `KERNEL::Regulator`.
     */
    Integrator_p2_4D_3ang(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_p2_4D_3ang(
              quadrature_provider,
              internal::make_int_grid<4, NT>(config, {"x_order", "cos1_order", "cos2_order", "phi_order"}),
              optimize_x_extent<typename KERNEL::Regulator, dim>(config))
    {
    }

    /**
     * @param quadrature_provider source of the quadrature rules
     * @param grid_size number of quadrature points per axis, in the order of the kernel arguments
     * @param x_extent upper limit of the radial integral, \f$q_\text{max}^2 / k^2\f$
     */
    Integrator_p2_4D_3ang(QuadratureProvider &quadrature_provider, const std::array<size_t, 4> grid_size,
                          ctype x_extent = 2.)
        // The azimuthal angle phi in [0,2pi) has a smooth 2pi-periodic integrand; the periodic trapezoidal
        // rule integrates it with spectral (exponential) accuracy, unlike Gauss-Legendre.
        : Base(quadrature_provider, grid_size, {0, 0, -1, 0}, {std::sqrt(x_extent), 1, 1, 2 * M_PI},
               {QuadratureType::legendre, QuadratureType::chebyshev2, QuadratureType::legendre,
                QuadratureType::trapezoidal}),
          x_extent(x_extent), k(1.)
    {
    }

    /// Set \f$x_\text{extent}\f$, so that \f$q_\text{max} = \sqrt{x_\text{extent}}\, k\f$.
    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0, 0, -1, 0}, {std::sqrt(x_extent) * k, 1, 1, 2 * M_PI});
    }

    /// Set the RG scale \f$k\f$, which moves the upper limit \f$q_\text{max} = \sqrt{x_\text{extent}}\, k\f$.
    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0, 0, -1, 0}, {std::sqrt(x_extent) * k, 1, 1, 2 * M_PI});
    }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
