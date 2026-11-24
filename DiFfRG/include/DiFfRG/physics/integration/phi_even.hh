#pragma once

// DiFfRG
#include <DiFfRG/common/quadrature/quadrature.hh>

// std
#include <cmath>
#include <cstddef>

/**
 * @file phi_even.hh
 * @brief Kernels even in the periodic angle \f$\phi\f$ (kernel_is_phi_even), and the folded \f$\phi\f$ rule for them.
 */

namespace DiFfRG
{
  /**
   * @brief Whether `K` declares `static constexpr bool phi_even = true`: its integrand is even in the periodic angle
   * \f$\phi\f$ of the integrator, \f$K(\ldots, \phi, \ldots) = K(\ldots, 2\pi - \phi, \ldots)\f$.
   *
   * The integrators with a periodic \f$\phi\f$ (Integrator_p2_4D_3ang, Integrator_fT_p2_4D_2ang) then fold their
   * trapezoidal rule onto \f$[0, \pi]\f$ (QuadratureType::trapezoidal_folded): with an even number \f$n\f$ of
   * \f$\phi\f$ nodes, the \f$n/2 + 1\f$ nodes \f$\phi_i = 2\pi i/n\f$, \f$i \le n/2\f$, carry the weights of themselves
   * and their mirror images \f$2\pi - \phi_i\f$. That is the same sum, term by term, at half the evaluations. An odd
   * \f$n\f$ has no node at \f$\pi\f$ and keeps the full rule. NumTracer proves the evenness and emits the trait
   * (MakeNTKernelDiFfRG); a wrong trait silently doubles the even part and drops the odd part of the integrand.
   */
  template <class K> inline constexpr bool kernel_is_phi_even = requires { requires K::phi_even; };

  namespace internal
  {
    /**
     * @brief The rule of a periodic angle on \f$[0, 2\pi)\f$ with `n` nodes: folded onto \f$[0, \pi]\f$ with
     * \f$n/2 + 1\f$ nodes if `even` and \f$n\f$ is even, the full periodic rule otherwise.
     */
    struct PhiRule {
      std::size_t nodes;
      double end;
      QuadratureType type;
      PhiRule(const bool even, const std::size_t n)
          : nodes(fold(even, n) ? n / 2 + 1 : n), end(fold(even, n) ? M_PI : 2. * M_PI),
            type(fold(even, n) ? QuadratureType::trapezoidal_folded : QuadratureType::trapezoidal)
      {
      }
      static constexpr bool fold(const bool even, const std::size_t n) { return even && n >= 2 && n % 2 == 0; }
    };
  } // namespace internal
} // namespace DiFfRG
