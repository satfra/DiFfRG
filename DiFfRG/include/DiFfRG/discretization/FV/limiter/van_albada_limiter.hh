#pragma once

#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>

#include <algorithm>
#include <cmath>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief van Albada limiter: for backward and forward slopes \f$a, b\f$ of the same sign,
     * \f$ab(a + b)/(a^2 + b^2)\f$, else 0.
     *
     * TVD, and like MinMod zero at a local extremum. Inside the same-sign region it is a smooth (rational) function of
     * \f$a, b\f$, equal to \f$a\f$ for \f$a = b\f$; its only kink is where a slope changes sign. MinMod is also kinked
     * where \f$|a| = |b|\f$, so van Albada gives a smoother jacobian, which helps the Newton iteration of an implicit
     * timestepper.
     */
    class VanAlbadaLimiter
    {
    public:
      /// \f$ab(a + b)/(a^2 + b^2)\f$ if @p a and @p b have the same strict sign, else 0.
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
