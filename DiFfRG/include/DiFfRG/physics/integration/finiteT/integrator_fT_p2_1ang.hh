#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/integration/finiteT/quadrature_integrator_fT.hh>

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

      // Forward the kernel's Matsubara traits. The measure this adapter multiplies in is
      // independent of the frequency, so it changes neither the parity in q0 nor the support in
      // q0. Without this forwarding the traits are silently lost: QuadratureIntegrator_fT only
      // ever sees the adapter, never KERNEL, so `matsubara_even` never fired for any
      // Integrator_fT_p2* flow.
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

    Integrator_fT_p2_1ang(QuadratureProvider &quadrature_provider, const ConfigTree &config)
      requires provides_regulator<KERNEL>
        : Integrator_fT_p2_1ang(quadrature_provider, internal::make_int_grid<2, NT>(config, {"x_order", "cos1_order"}),
                                optimize_x_extent<typename KERNEL::Regulator>(config),
                                config.get_double("/physical/T", 1.0))
    {
      apply_matsubara_overrides(config);
    }

    Integrator_fT_p2_1ang(QuadratureProvider &quadrature_provider, const std::array<size_t, 2> grid_size,
                          ctype x_extent = 2., ctype T = 1, ctype typical_E = 1)
        : Base(quadrature_provider, grid_size, {0, 0.}, {std::sqrt(x_extent), 1.},
               {QuadratureType(QuadratureType::legendre), polar_quadrature()}, T, typical_E),
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
      Base::set_grid_extents({0, 0.}, {std::sqrt(x_extent) * k, 1.});
      // The spatial grid is already cut at sqrt(x_extent) * k because the regulator dies there;
      // the summand's support in FREQUENCY is the same ball, so the exact Matsubara sum can be cut
      // at the same radius. See QuadratureIntegrator_fT::set_frequency_cutoff.
      Base::set_frequency_cutoff(std::sqrt(this->x_extent) * this->k);
    }

    void set_k(ctype k)
    {
      this->k = k;
      Base::set_grid_extents({0, 0.}, {std::sqrt(x_extent) * k, 1.});
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
