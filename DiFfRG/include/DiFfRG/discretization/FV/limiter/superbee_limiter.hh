#pragma once

#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>

#include <cmath>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Superbee slope limiter (Roe, 1986).
     *
     * The widest of the second-order TVD slope limiters:
     *   slope_limit(a, b) = maxmod( minmod(2a, b), minmod(a, 2b) )
     * where
     *   minmod(x, y) = sgn(x)+sgn(y) all-same-sign ? sign·min(|x|,|y|) : 0
     *   maxmod(x, y) = sgn(x)+sgn(y) all-same-sign ? sign·max(|x|,|y|) : 0
     *
     * Equivalent compact form (Sweby 1984): for r = du_1 / du_2,
     *   phi(r) = max(0, min(2r, 1), min(r, 2))
     * and the limited slope is phi(r)·du_2.
     *
     * Superbee returns the steepest slope still consistent with the TVD
     * region. Sharpens discontinuities much more aggressively than MinMod;
     * uses identity uniform-sign-of-the-two-arguments check the same way.
     */
    class SuperbeeLimiter
    {
    public:
      template <typename NumberType> static NumberType slope_limit(const NumberType &du_1, const NumberType &du_2)
      {
        using std::abs;
        using std::max;
        using std::min;

        // Branch-free same-sign indicator: 1 when du_1 and du_2 share strict sign, else 0.
        // Identity: |sgn(a)+sgn(b)| / 2.
        const int s1 = limiter_utils::sgn(du_1);
        const int s2 = limiter_utils::sgn(du_2);
        if (s1 == 0 || s2 == 0 || s1 != s2) return NumberType(0);

        // Both slopes strictly positive or strictly negative. Work in the
        // common-sign convention then re-apply the sign at the end.
        const auto a = abs(du_1);
        const auto b = abs(du_2);

        // minmod(2a, b) and minmod(a, 2b) in this regime collapse to
        // min(2a, b) and min(a, 2b); maxmod is then max() of those.
        const auto m1 = min(NumberType(2) * a, b);
        const auto m2 = min(a, NumberType(2) * b);
        const auto limited = max(m1, m2);

        return NumberType(s1) * limited;
      }
    };
    static_assert(HasSlopeLimiter<SuperbeeLimiter>);

  } // namespace def
} // namespace DiFfRG
