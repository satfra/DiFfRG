#pragma once

#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Unlimited central-difference "limiter" — returns the arithmetic mean
     * of the two one-sided slopes.
     *
     * Diagnostic / comparison choice. NOT TVD: this can amplify oscillations at
     * discontinuities. Useful as the smooth-end bracket against MinModLimiter
     * to measure how much the TVD clipping contributes to time-stepper failures.
     */
    class CentralLimiter
    {
    public:
      template <typename NumberType> static NumberType slope_limit(const NumberType &du_1, const NumberType &du_2)
      {
        return NumberType(0.5) * (du_1 + du_2);
      }
    };
    static_assert(HasSlopeLimiter<CentralLimiter>);

  } // namespace def
} // namespace DiFfRG
