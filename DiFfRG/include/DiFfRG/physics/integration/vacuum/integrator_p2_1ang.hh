#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/quadrature_integrator.hh>

// standard libraries
#include <array>

namespace DiFfRG
{
  namespace internal
  {
    template <int dim, typename NT, typename KERNEL> class Transform_p2_1ang
    {
    public:
      using ctype = typename get_type::ctype<NT>;

      // The polar angle (loop vs. external momentum) of a 2-point loop in d dims carries the zonal
      // measure (1-c^2)^{(d-3)/2} dc, supplied by the Gauss-Jacobi(alpha=beta=(d-3)/2) angular
      // quadrature (see the class below). The remaining (d-2)-sphere gives S_{d-2} = S_d_prec(dim-1).
      static constexpr ctype int_prefactor = S_d_prec<ctype>(dim - 1) // remaining solid angle after the polar angle
                                             / powr<dim>(2 * (ctype)M_PI); // fourier factor

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q, const ctype cos, const T &...t)
        requires provides_kernel<NT, KERNEL, ctype, 2, T...>
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<dim - 1>(q); // from p integral
        const NT result = KERNEL::kernel(q, cos, t...);

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
   * @brief Integrator_p2_1ang integrates a kernel \f$K(p,\cos,\ldots)\f$ depending on the radial momentum \f$p\f$ and
   * the cosine of the single polar angle between the loop and external momentum as
   * $$
   * \frac{S_{d-1}}{(2\pi)^{d}}\,\int_{-1}^1 dc\,(1-c^2)^{\frac{d-3}{2}}\,\int_0^\infty dp\, p^{d-1} K(p,c,\ldots)
   * $$
   * in \f$d\f$ dimensions, where \f$S_{d-1}\f$ is the solid angle of the \f$(d-2)\f$-sphere remaining after the polar
   * angle is singled out. The zonal measure \f$(1-c^2)^{(d-3)/2}\f$ is supplied exactly by a Gauss-Jacobi angular
   * quadrature, so it does not appear in the kernel.
   *
   * @tparam dim dimension of the momentum space, i.e. $d$ in the above equation
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, which must provide the static methods `kernel` and `constant`
   * @tparam ExecutionSpace can be any execution space, e.g. GPU_exec, TBB_exec, or KokkosHost_exec.
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
  class Integrator_p2_1ang
      : public QuadratureIntegrator<2, NT, internal::Transform_p2_1ang<dim, NT, KERNEL>, ExecutionSpace>
  {
    using Base = QuadratureIntegrator<2, NT, internal::Transform_p2_1ang<dim, NT, KERNEL>, ExecutionSpace>;

    // Gauss-Jacobi quadrature on [-1,1] with weight (1-c^2)^{(d-3)/2} for the polar angle.
    // Dimension-agnostic: reduces to Chebyshev-1 (d=2), Legendre (d=3), Chebyshev-2 (d=4), ...
    static QuadratureType polar_quadrature()
    {
      QuadratureType t(QuadratureType::jacobi);
      t.a = static_cast<double>(-1);
      t.b = static_cast<double>(1);
      t.alpha = (static_cast<double>(dim) - 3.) / 2.;
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

    Integrator_p2_1ang(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_p2_1ang(quadrature_provider, internal::make_int_grid<2, NT>(config, {"x_order", "cos1_order"}),
                             optimize_x_extent<typename KERNEL::Regulator>(config))
    {
    }

    Integrator_p2_1ang(QuadratureProvider &quadrature_provider, const std::array<size_t, 2> grid_size,
                       ctype x_extent = 2.)
        : Base(quadrature_provider, grid_size, {0, 0.}, {std::sqrt(x_extent), 1.},
               {QuadratureType(QuadratureType::legendre), polar_quadrature()}),
          x_extent(x_extent), k(1.)
    {
    }

    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0, 0.}, {std::sqrt(x_extent) * k, 1.});
    }

    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0, 0.}, {std::sqrt(x_extent) * k, 1.});
    }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
