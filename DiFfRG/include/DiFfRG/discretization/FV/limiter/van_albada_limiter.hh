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
        // Strict TVD: vanish on sign disagreement, including either argument zero. Compared directly, as a * b
        // underflows for tiny slopes.
        const bool same_sign = (a > NumberType(0) && b > NumberType(0)) || (a < NumberType(0) && b < NumberType(0));
        if (!same_sign) return NumberType(0);

        // a b (a + b) / (a² + b²) is homogeneous of degree one, so it can be evaluated on a / s, b / s: with
        // s = max(|a|, |b|) the denominator is at least 1, and a² + b² cannot underflow.
        using std::abs;
        const NumberType s = abs(a) > abs(b) ? abs(a) : abs(b);
        const NumberType as = a / s, bs = b / s;
        return s * as * bs * (as + bs) / (as * as + bs * bs);
      }
    };
    static_assert(HasSlopeLimiter<VanAlbadaLimiter>);

  } // namespace def
} // namespace DiFfRG
