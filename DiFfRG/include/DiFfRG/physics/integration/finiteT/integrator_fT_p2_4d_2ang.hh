#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

namespace DiFfRG
{
  namespace internal
  {
    template <typename NT, typename KERNEL> class Transform_fT_p2_4D_2ang
    {
    public:
      using ctype = typename get_type::ctype<NT>;

      static constexpr ctype int_prefactor = powr<-3>((ctype)2 * M_PI); // fourier factor

      // Forward the kernel's Matsubara traits. The measure this adapter multiplies in is
      // independent of the frequency, so it changes neither the parity in q0 nor the support in
      // q0. Without this forwarding the traits are silently lost: QuadratureIntegrator_fT only
      // ever sees the adapter, never KERNEL, so `matsubara_even` never fired for any
      // Integrator_fT_p2* flow.
      static constexpr bool matsubara_even = kernel_is_matsubara_even<KERNEL>;
      static constexpr bool matsubara_finite_extent = kernel_has_finite_matsubara_extent<KERNEL>;
      static constexpr bool matsubara_split = kernel_has_matsubara_split<KERNEL>;

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel_finite_extent(const ctype q, const ctype cos1, const ctype phi,
                                                                 const ctype q0, const T &...t)
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<3 - 1>(q); // from p integral
        return int_prefactor * int_element * KERNEL::kernel_finite_extent(q, cos1, phi, q0, t...);
      }

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel_tail(const ctype q, const ctype cos1, const ctype phi,
                                                        const ctype q0, const T &...t)
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<3 - 1>(q); // from p integral
        return int_prefactor * int_element * KERNEL::kernel_tail(q, cos1, phi, q0, t...);
      }

      template <typename... T>
      static KOKKOS_FORCEINLINE_FUNCTION NT kernel(const ctype q, const ctype cos1, const ctype phi, const ctype q0,
                                                   const T &...t)
        requires provides_kernel<NT, KERNEL, ctype, 4, T...>
      {
        using namespace DiFfRG::compute;

        const ctype int_element = powr<3 - 1>(q); // from p integral

        const NT result = KERNEL::kernel(q, cos1, phi, q0, t...);

        return int_prefactor * int_element * result;
      }

      template <typename... T> static KOKKOS_FORCEINLINE_FUNCTION NT constant(const T &...t)
        requires provides_constant<NT, KERNEL, T...>
      {
        return KERNEL::constant(t...);
      }
    };
  } // namespace internal

  template <int dim, typename NT, typename KERNEL, typename ExecutionSpace>
    requires(dim == 4)
  class Integrator_fT_p2_4D_2ang
      : public QuadratureIntegrator_fT<4, NT, internal::Transform_fT_p2_4D_2ang<NT, KERNEL>, ExecutionSpace>
  {
    using Base = QuadratureIntegrator_fT<4, NT, internal::Transform_fT_p2_4D_2ang<NT, KERNEL>, ExecutionSpace>;

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
     * @brief Apply the /integration/force_exact_matsubara_sum override, if the model sets it.
     *
     * The per-kernel `matsubara_finite_extent` trait is the right default -- it is derived from the
     * diagram algebra -- but two cases need a dial. A model whose kernels predate the trait can
     * turn the exact sum on for the whole run, and a study can turn it off to price it against
     * the Gaussian rule on the same kernel. Forcing it ON for a summand that does NOT vanish above
     * sqrt(x_extent)*k silently truncates the sum, so it is opt-in and absent by default.
     */
    void apply_matsubara_overrides(const ConfigTree &config)
    {
      if (config.contains("/integration/force_exact_matsubara_sum"))
        Base::set_allow_exact_matsubara_sum(config.get_bool("/integration/force_exact_matsubara_sum", false));
      if (config.contains("/integration/matsubara_extent_margin"))
        Base::set_matsubara_extent_margin(config.get_double("/integration/matsubara_extent_margin", 1.));
    }

    Integrator_fT_p2_4D_2ang(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_fT_p2_4D_2ang(
              quadrature_provider, internal::make_int_grid<3, NT>(config, {"x_order", "cos1_order", "phi_order"}),
              optimize_x_extent<typename KERNEL::Regulator>(config), config.get_double("/physical/T", 1.0))
    {
      apply_matsubara_overrides(config);
    }

    Integrator_fT_p2_4D_2ang(QuadratureProvider &quadrature_provider, const std::array<size_t, 3> grid_size,
                             ctype x_extent = 2., ctype T = 1, ctype typical_E = 1)
        // The azimuthal angle phi in [0,2pi) has a smooth 2pi-periodic integrand; the periodic trapezoidal
        // rule integrates it with spectral (exponential) accuracy, unlike Gauss-Legendre.
        : Base(quadrature_provider, grid_size, {0, -1, 0}, {std::sqrt(x_extent), 1, 2 * M_PI},
               {QuadratureType::legendre, QuadratureType::legendre, QuadratureType::trapezoidal}, T, typical_E),
          x_extent(x_extent), k(1.)
    {
      // The spatial grid is already cut at sqrt(x_extent) * k because the regulator dies there;
      // the summand's support in FREQUENCY is the same ball, so the exact Matsubara sum can be cut
      // at the same radius. See QuadratureIntegrator_fT::set_frequency_cutoff.
      Base::set_frequency_cutoff(std::sqrt(this->x_extent) * this->k);
    }

    void set_x_extent(ctype x_extent)
    {
      this->x_extent = x_extent;
      Base::set_grid_extents({0, -1, 0}, {std::sqrt(x_extent) * k, 1, 2 * M_PI});
      // The spatial grid is already cut at sqrt(x_extent) * k because the regulator dies there;
      // the summand's support in FREQUENCY is the same ball, so the exact Matsubara sum can be cut
      // at the same radius. See QuadratureIntegrator_fT::set_frequency_cutoff.
      Base::set_frequency_cutoff(std::sqrt(this->x_extent) * this->k);
    }

    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0, -1, 0}, {std::sqrt(x_extent) * k, 1, 2 * M_PI});
      Base::set_k(k);
      // The spatial grid is already cut at sqrt(x_extent) * k because the regulator dies there;
      // the summand's support in FREQUENCY is the same ball, so the exact Matsubara sum can be cut
      // at the same radius. See QuadratureIntegrator_fT::set_frequency_cutoff.
      Base::set_frequency_cutoff(std::sqrt(this->x_extent) * this->k);
    }

    void set_typical_E(ctype typical_E) { Base::set_typical_E(typical_E); }

  private:
    ctype x_extent;
    ctype k;
  };
} // namespace DiFfRG
