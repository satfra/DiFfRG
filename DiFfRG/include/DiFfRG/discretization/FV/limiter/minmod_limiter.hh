#pragma once

// DiFfRG
#include <DiFfRG/discretization/FV/limiter/abstract_limiter.hh>

// standard library
#include <algorithm>
#include <cmath>
#include <cstddef>

// autodiff
#include <autodiff/forward/real/real.hpp>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief MinMod limiter, \f$\tfrac12(\mathrm{sgn}\,a + \mathrm{sgn}\,b)\min(|a|, |b|)\f$ for the backward and
     * forward slopes \f$a, b\f$.
     *
     * Same signs: the smaller slope in magnitude; opposite signs (a local extremum): zero. The most dissipative TVD
     * limiter, and the default of FV::KurganovTadmor::Assembler. It is not differentiable where \f$|a| = |b|\f$; there
     * its automatic-differentiation derivative is the mean of both branches.
     *
     * Usage:
     * @code
     * using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model,
     *                       def::TVDReconstructor<Discretization::dim, def::MinModLimiter, double>>;
     * @endcode
     */
    class MinModLimiter
    {
      template <typename NumberType>
      static NumberType abs_with_zero_tie_derivative(const NumberType &value, const NumberType & /*scale*/)
      {
        using std::abs;
        return abs(value);
      }

      // |a1 - a2| has no derivative where the two slopes tie, so there it contributes none and the limiter's
      // derivative is the mean of both branches. The tie is RELATIVE to the slopes: an absolute threshold
      // mistakes distinct slopes of a small-valued field for a tie. The value is kept exact either way, so the
      // AD limiter returns the same slope as the double one.
      template <size_t order, typename NumberType>
      static autodiff::Real<order, NumberType> abs_with_zero_tie_derivative(const autodiff::Real<order, NumberType> &value,
                                                                            const autodiff::Real<order, NumberType> &scale)
      {
        using std::abs;
        if (std::abs(value.val()) <= 1.0e-12 * std::abs(scale.val()))
          return autodiff::Real<order, NumberType>(std::abs(value.val()));
        return abs(value);
      }

    public:
      /// \f$\tfrac12(\mathrm{sgn}\,a + \mathrm{sgn}\,b)\min(|a|, |b|)\f$.
      template <typename NumberType> static NumberType slope_limit(const NumberType &du_1, const NumberType &du_2)
      {
        // Use the algebraic identity min(a,b) = 0.5*(a+b-|a-b|) instead of std::min so that
        // AD follows the active branch away from ties without introducing a std::min discontinuity.
        using std::abs;
        const auto a1 = abs(du_1), a2 = abs(du_2);
        const auto min_a = NumberType(0.5) * (a1 + a2 - abs_with_zero_tie_derivative(a1 - a2, a1 + a2));
        return NumberType(0.5) * (limiter_utils::sgn(du_1) + limiter_utils::sgn(du_2)) * min_a;
      }
    };
    static_assert(HasSlopeLimiter<MinModLimiter>);

  } // namespace def
} // namespace DiFfRG
