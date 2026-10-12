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
    template <int dim, typename NT, typename KERNEL> class Transform_p2
    {
    public:
      using ctype = typename get_type::ctype<NT>;

      static constexpr ctype int_prefactor = S_d_prec<ctype>(dim)          // solid nd angle
                                             / powr<dim>(2 * (ctype)M_PI); // fourier factor

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q, const T &...t)
        requires provides_kernel<NT, KERNEL, ctype, 1, T...>
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<dim - 1>(q); // from p integral
        const NT result = KERNEL::kernel(q, t...);

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
   * @brief Integrates a kernel over a \f$d\f$-dimensional momentum when it depends only on the magnitude
   * \f$q = |\vec q|\f$ of the loop momentum.
   *
   * \f[
   *   I = \texttt{constant}(\ldots) + \frac{S_d}{(2\pi)^d} \int_0^{q_\text{max}} dq\, q^{d-1}\, K(q, \ldots)\,,
   *   \qquad q_\text{max} = \sqrt{x_\text{extent}}\, k\,,
   * \f]
   * where \f$S_d = 2\pi^{d/2}/\Gamma(d/2)\f$ is the surface of the unit sphere in \f$d\f$ dimensions. The upper limit
   * follows the RG scale \f$k\f$ (set with set_k(), initially 1), and \f$x_\text{extent}\f$ is given in units of
   * \f$k^2\f$. The radial integral uses Gauss-Legendre quadrature. The `_p2` in the name is historical: the kernel
   * receives \f$|q|\f$, not \f$q^2\f$.
   *
   * Kernel interface:
   * - get(): `KERNEL::kernel(q, args...)` and `KERNEL::constant(args...)`
   * - map(): `KERNEL::kernel(q, pos..., args...)` and `KERNEL::constant(pos..., args...)`, where `pos...` are the
   *   components of the external grid point.
   *
   * @see QuadratureIntegrator for the kernel interface, get() and map().
   *
   * Example:
   * @code
   * struct MyKernel {
   *   using Regulator = DiFfRG::LitimRegulator<>; // needed by the ConfigTree constructor
   *
   *   static KOKKOS_FORCEINLINE_FUNCTION double kernel(const double q, const double k, const double m2)
   *   {
   *     const double q2 = q * q, k2 = k * k;
   *     return Regulator::RBdot(k2, q2) / DiFfRG::powr<2>(q2 + Regulator::RB(k2, q2) + m2);
   *   }
   *   static KOKKOS_FORCEINLINE_FUNCTION double constant(const double, const double) { return 0.; }
   * };
   *
   * DiFfRG::Integrator_p2<4, double, MyKernel, DiFfRG::GPU_exec> integrator(quadrature_provider, config);
   * integrator.set_k(k); // on every RG step
   * double result;
   * integrator.get(result, k, m2);
   * @endcode
   *
   * @tparam dim momentum-space dimension \f$d \geq 1\f$
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, providing static `kernel` and `constant`
   * @tparam ExecutionSpace GPU_exec, TBB_exec or KokkosHost_exec
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim >= 1)
  class Integrator_p2 : public QuadratureIntegrator<1, NT, internal::Transform_p2<dim, NT, KERNEL>, ExecutionSpace>
  {
    using Base = QuadratureIntegrator<1, NT, internal::Transform_p2<dim, NT, KERNEL>, ExecutionSpace>;

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
     * Reads the quadrature order from `/integration/x_order` and finds \f$x_\text{extent}\f$ with
     * optimize_x_extent() for `KERNEL::Regulator`.
     */
    Integrator_p2(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_p2(quadrature_provider, internal::make_int_grid<1, NT>(config, {"x_order"}),
                        optimize_x_extent<typename KERNEL::Regulator, dim>(config))
    {
    }

    /**
     * @param quadrature_provider source of the quadrature rules
     * @param grid_size number of radial quadrature points
     * @param x_extent upper limit of the radial integral, \f$q_\text{max}^2 / k^2\f$
     */
    Integrator_p2(QuadratureProvider &quadrature_provider, const std::array<size_t, 1> grid_size, ctype x_extent = 2.)
        : Base(quadrature_provider, grid_size, {0}, {std::sqrt(x_extent)}, {QuadratureType::legendre}),
          x_extent(x_extent), k(1.)
    {
    }

    /// Set \f$x_\text{extent}\f$, so that \f$q_\text{max} = \sqrt{x_\text{extent}}\, k\f$.
    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0}, {std::sqrt(x_extent) * k});
    }

    /// Set the RG scale \f$k\f$, which moves the upper limit \f$q_\text{max} = \sqrt{x_\text{extent}}\, k\f$.
    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0}, {std::sqrt(x_extent) * k});
    }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
