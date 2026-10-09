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
   * @brief Integrator_p2_4D_3ang integrates a kernel \f$K(p,\cos_1,\cos_2,\ldots)\f$ depending on the radial momentum
   * \f$p\f$ and two angles on \f$[0,\pi]\f$ and one angle on \f$[0,2\pi]\f$ as
   * $$
   * \frac{1}{(2\pi)^{d}} \,\int_0^\pi d\cos_1\,\int_0^\pi d\cos_2\,\int_0^{2\pi}d\phi\,\int_0^\infty dp^2 p^{d-2}
   * K(p,\cos_1,\cos_2,\pi,\ldots)
   * $$
   * in \f$d=4\f$ dimensions.
   *
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, which must provide the static methods `kernel` and `constant`
   * @tparam ExecutionSpace can be any execution space, e.g. GPU_exec, TBB_exec.
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

    Integrator_p2_4D_3ang(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_p2_4D_3ang(
              quadrature_provider,
              internal::make_int_grid<4, NT>(config, {"x_order", "cos1_order", "cos2_order", "phi_order"}),
              optimize_x_extent<typename KERNEL::Regulator>(config))
    {
    }

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

    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0, 0, -1, 0}, {std::sqrt(x_extent) * k, 1, 1, 2 * M_PI});
    }

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
