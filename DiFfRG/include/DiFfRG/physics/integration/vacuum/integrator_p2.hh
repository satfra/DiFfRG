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
   * @brief Integrator_p2 integrates a kernel \f$K(p,\ldots)\f$ depending on the radial momentum \f$p\f$ as
   * $$
   * \frac{S_d}{(2\pi)^{d}}\,\int_0^\infty dp^2 p^{d-2} K(p, \ldots)
   * $$
   * where \f$S_d\f$ is the solid angle in $d$ dimensions.
   *
   * @tparam dim dimension of the momentum space, i.e. $d$ in the above equation
   * @tparam NT numerical type of the result
   * @tparam KERNEL kernel to be integrated, which must provide the static methods `kernel` and `constant`
   * @tparam ExecutionSpace can be any execution space, e.g. GPU_exec, TBB_exec, or KokkosHost_exec.
   */
  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
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

    Integrator_p2(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_p2(quadrature_provider, internal::make_int_grid<1, NT>(config, {"x_order"}),
                        optimize_x_extent<typename KERNEL::Regulator>(config))
    {
    }

    Integrator_p2(QuadratureProvider &quadrature_provider, const std::array<size_t, 1> grid_size, ctype x_extent = 2.)
        : Base(quadrature_provider, grid_size, {0}, {std::sqrt(x_extent)}, {QuadratureType::legendre}),
          x_extent(x_extent), k(1.)
    {
    }

    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0}, {std::sqrt(x_extent) * k});
    }

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
