#pragma once

// DiFfRG
#include <DiFfRG/common/types.hh>
#include <DiFfRG/physics/integration/map_scheduler.hh>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Fallback quadrature order for a momentum-space direction.
     *
     * Radial directions need far more points than angular ones, so the guess depends on the
     * name. Both are only guesses - make_int_grid() warns when either is taken.
     */
    inline size_t default_quadrature_order(const std::string &name)
    {
      return name.starts_with("x") || name.starts_with("q") ? 32 : 8;
    }

    template <int dim, typename NT = double>
    std::array<size_t, dim> make_int_grid(const ConfigTree &config, const std::array<std::string, dim> &names)
    {
      std::array<size_t, dim> int_grid;
      for (int i = 0; i < dim; ++i)
        int_grid[i] = config.get_uint_or_warn("/integration/" + names[i], default_quadrature_order(names[i]));
      if constexpr (get_type::is_autodiff<NT>) {
        const double factor = config.get_double("/integration/jacobian_quadrature_factor", 0.8);
        for (int i = 0; i < dim; ++i)
          int_grid[i] = static_cast<size_t>(factor * int_grid[i]);
      }
      return int_grid;
    }
  } // namespace internal

  template <typename KERNEL>
  concept provides_regulator = requires { typename KERNEL::Regulator; };

  template <typename NT, typename KERNEL, typename ctype, int dim, typename... ARGS>
  NT multidim_kernel_call(const ARGS &...args)
  {
    if constexpr (dim == 0)
      return KERNEL::kernel(args...);
    else {
      const ctype darg{};
      return multidim_kernel_call<NT, KERNEL, ctype, dim - 1, ARGS...>(darg, args...);
    }
  }

  template <typename NT, typename KERNEL, typename ctype, int dim, typename... ARGS>
  concept provides_kernel =
      requires(const ARGS &...args) { multidim_kernel_call<NT, KERNEL, ctype, dim, ARGS...>(args...); };

  template <typename NT, typename KERNEL, typename... ARGS>
  concept provides_constant = requires(const ARGS &...args) {
    { KERNEL::constant(args...) } -> std::convertible_to<NT>;
  };

  template <typename NT, typename KERNEL, typename ctype, int dim, typename... ARGS>
  concept is_valid_kernel =
      (provides_kernel<NT, KERNEL, ctype, dim, ARGS...> && provides_constant<NT, KERNEL, ARGS...>);

  template <typename NT, typename KERNEL, typename ctype, int dim, typename... ARGS>
  consteval void check_kernel_requirements()
  {
    static_assert(provides_kernel<NT, KERNEL, ctype, dim, ARGS...>,
                  "Kernel must provide a static 'kernel(...)' method callable with the integration arguments.");
    static_assert(provides_constant<NT, KERNEL, ARGS...>,
                  "Kernel must provide a static 'constant(...)' method returning the numeric type.");
  }

  /**
   * @brief Common base of every integrator, carrying the identity MapScheduler needs.
   *
   * There is deliberately no MPI-related setter here. Distribution is decided automatically by
   * MapScheduler from the size of each map() call, so the feature has no user-facing API at all:
   * an application scales across ranks without a single line of change.
   */
  class AbstractIntegrator
  {
  public:
    // Construction order is part of the program, so these ids come out the same on every rank
    // without any communication -- which is what lets MapScheduler put them in a plan that all
    // ranks must agree on.
    //
    // The copy constructor is deliberately left implicit, so that an id travels with the object.
    // Integrators are captured by value into device lambdas (KOKKOS_CLASS_LAMBDA), and a
    // user-defined copy constructor is host-only, which makes that capture ill-formed. Copies
    // sharing an id is harmless: only the host-side original ever schedules.
    AbstractIntegrator() : m_integrator_id(next_integrator_id()) {}

    /// Stable, rank-independent identity of this integrator.
    KOKKOS_FORCEINLINE_FUNCTION size_t integrator_id() const { return m_integrator_id; }

  protected:
    size_t m_integrator_id;

  private:
    static size_t next_integrator_id();
  };
} // namespace DiFfRG