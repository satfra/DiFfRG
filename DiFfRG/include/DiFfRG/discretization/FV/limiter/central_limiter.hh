#pragma once

#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief No limiting: the central slope \f$\tfrac12(a + b)\f$ of the backward and forward slopes \f$a, b\f$.
     *
     * Not TVD: it creates oscillations at discontinuities and steep fronts. Useful as a reference, e.g. to see how much
     * the limiting of another limiter affects a flow.
     */
    class CentralLimiter
    {
    public:
      /// \f$\tfrac12(a + b)\f$.
      template <typename NumberType> static NumberType slope_limit(const NumberType &du_1, const NumberType &du_2)
      {
        return NumberType(0.5) * (du_1 + du_2);
      }
    };
    static_assert(HasSlopeLimiter<CentralLimiter>);

  } // namespace def
} // namespace DiFfRG
