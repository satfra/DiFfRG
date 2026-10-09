#pragma once

#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>

#include <algorithm>
#include <cmath>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief van Albada (1982) slope limiter — second-order TVD and C¹ in u.
     *
     * Functional form (van Albada–van Leer–Roberts):
     *   phi(a, b) = ((a² + ε) · b + (b² + ε) · a) / (a² + b² + 2ε)   when a·b > 0,
     *              = 0                                                otherwise.
     *
     * Equivalent compact form used here:
     *   phi(a, b) = step(a·b) · (a · b² + b · a²) / (a² + b² + ε_floor)
     *
     * Key properties used by this project:
     *   - C¹ in (a, b) on the strict interior of {a·b > 0}: numerator and
     *     denominator are both polynomials, and the denominator is bounded
     *     away from zero by ε_floor.
     *   - TVD: limited slope vanishes when slopes disagree in sign.
     *   - Smooth-extremum: reduces to (a + b)/2 when a ≈ b > 0 (no clipping
     *     at smooth extrema, unlike MinMod).
     *
     * Diagnostic motivation: MinMod is C⁰ but NOT C¹ in u (sign-change kink
     * AND magnitude-tie kink). SUNDIALS IDA's Newton iteration needs at
     * least a locally Lipschitz Jacobian to converge; MinMod's kinks cause
     * the apparent Jacobian to flip by O(1) inside O(ε) perturbations of
     * the state, which stalls the implicit integrator. van Albada removes
     * the magnitude-tie kink entirely and leaves only the (much less
     * frequently crossed) sign-change boundary.
     *
     * Note: the sign-change boundary a·b = 0 is implemented here via a
     * branch (the ternary). If even that one kink is too sharp for IDA,
     * a softplus regularisation `phi · 0.5·(1 + tanh(a·b/δ))` is a drop-in
     * upgrade — but try the cleaner form first.
     */
    class VanAlbadaLimiter
    {
    public:
      template <typename NumberType> static NumberType slope_limit(const NumberType &a, const NumberType &b)
      {
        // Strict TVD: vanish on sign disagreement, including either argument zero.
        const auto ab = a * b;
        if (ab <= NumberType(0)) return NumberType(0);

        // ε_floor: defensive against the truly-zero-slope smooth-extremum case
        // where a² + b² could be machine zero. Set far below any physical
        // (du/h)² scale; ~1e-300 is safely below double's denormal range.
        const NumberType eps_floor = NumberType(1e-300);

        const auto a2 = a * a;
        const auto b2 = b * b;
        return (a * b2 + b * a2) / (a2 + b2 + eps_floor);
      }
    };
    static_assert(HasSlopeLimiter<VanAlbadaLimiter>);

  } // namespace def
} // namespace DiFfRG
