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

  /// A kernel that names its regulator as `using Regulator = ...;`. Integrator constructors taking a ConfigTree need
  /// it to choose the radial cutoff (see optimize_x_extent()).
  template <typename KERNEL>
  concept provides_regulator = requires { typename KERNEL::Regulator; };

  /// Calls `KERNEL::kernel` with `dim` default-constructed integration variables followed by `args`. Used to test the
  /// kernel signature in provides_kernel.
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

  /**
   * @brief The integrand interface every integrator expects from its `KERNEL` class.
   *
   * A kernel is a class with two static member functions,
   * @code
   * struct MyKernel {
   *   static KOKKOS_FORCEINLINE_FUNCTION auto kernel(const double x1, ..., const double xdim, const auto &...args);
   *   static KOKKOS_FORCEINLINE_FUNCTION auto constant(const auto &...args);
   * };
   * @endcode
   * and an integrator computes
   * \f[
   *   I(\text{args}) = \texttt{constant}(\text{args}) + \int d\mu(x)\, \text{kernel}(x_1, \ldots, x_\text{dim},
   *   \text{args}) \,,
   * \f]
   * where \f$d\mu(x)\f$ is the measure of the integrator. `constant` is added once and is not integrated.
   * `kernel` is called with the `dim` integration variables first (of type `ctype`), then the extra arguments passed
   * to `get()`; `map()` additionally inserts the grid position between the two, see QuadratureIntegrator. Both must
   * return something convertible to `NT` and be callable on the device (`KOKKOS_FUNCTION` or similar).
   *
   * provides_kernel checks `kernel`, provides_constant checks `constant`, is_valid_kernel checks both.
   */
  template <typename NT, typename KERNEL, typename ctype, int dim, typename... ARGS>
  concept provides_kernel =
      requires(const ARGS &...args) { multidim_kernel_call<NT, KERNEL, ctype, dim, ARGS...>(args...); };

  /// `KERNEL::constant(args...)` exists and returns something convertible to `NT`. See provides_kernel.
  template <typename NT, typename KERNEL, typename... ARGS>
  concept provides_constant = requires(const ARGS &...args) {
    { KERNEL::constant(args...) } -> std::convertible_to<NT>;
  };

  /// `KERNEL` provides both `kernel` and `constant` for these arguments. See provides_kernel.
  template <typename NT, typename KERNEL, typename ctype, int dim, typename... ARGS>
  concept is_valid_kernel =
      (provides_kernel<NT, KERNEL, ctype, dim, ARGS...> && provides_constant<NT, KERNEL, ARGS...>);

  /// Fails compilation with a readable message if `KERNEL` does not satisfy is_valid_kernel.
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
   * There is no MPI-related setter: MapScheduler decides from the size of each map() call whether and how to spread
   * it over ranks, so an application runs on several ranks without changes.
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