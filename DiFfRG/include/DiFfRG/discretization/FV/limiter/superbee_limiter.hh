#pragma once

#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>

#include <cmath>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Superbee limiter (Roe): for backward and forward slopes \f$a, b\f$ of the same sign,
     * \f$\mathrm{sgn}\,a\, \max\big(\min(2|a|, |b|), \min(|a|, 2|b|)\big)\f$, else 0.
     *
     * Equivalently \f$\varphi(r)\, b\f$ with \f$r = a/b\f$ and \f$\varphi(r) = \max(0, \min(2r, 1), \min(r, 2))\f$.
     * The steepest slope that keeps the reconstruction TVD: it sharpens fronts the most of the TVD limiters, and can
     * steepen smooth profiles into steps.
     */
    class SuperbeeLimiter
    {
    public:
      /// The Superbee slope; 0 unless @p du_1 and @p du_2 have the same strict sign.
      template <typename NumberType> static NumberType slope_limit(const NumberType &du_1, const NumberType &du_2)
      {
        using std::abs;
        using std::max;
        using std::min;

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
