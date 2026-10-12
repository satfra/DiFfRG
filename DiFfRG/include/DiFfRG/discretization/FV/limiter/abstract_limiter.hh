#pragma once

// standard library
#include <concepts>

namespace DiFfRG
{
  namespace def
  {
    // clang-format off
    /**
     * @brief Concept of a slope limiter, the `Limiter` template parameter of def::TVDReconstructor.
     *
     * A limiter provides `static NumberType slope_limit(a, b)`: from the backward slope \f$a\f$ and the forward slope
     * \f$b\f$ of a cell along one direction (differences of neighbouring cell averages divided by the centre distance),
     * the slope used in the reconstruction.
     *
     * | Limiter          | slope_limit(a, b)                                                                  | TVD | Smooth in (a, b) |
     * |------------------|------------------------------------------------------------------------------------|-----|------------------|
     * | MinModLimiter    | \f$\tfrac12(\mathrm{sgn}\,a + \mathrm{sgn}\,b)\min(\lvert a\rvert, \lvert b\rvert)\f$                         | yes | no: kinks at \f$\lvert a\rvert = \lvert b\rvert\f$ and at sign changes |
     * | SuperbeeLimiter  | \f$\mathrm{sgn}\,a \max(\min(2\lvert a\rvert, \lvert b\rvert), \min(\lvert a\rvert, 2\lvert b\rvert))\f$ if \f$ab > 0\f$, else 0  | yes | no |
     * | VanAlbadaLimiter | \f$ab(a + b)/(a^2 + b^2)\f$ if \f$ab > 0\f$, else 0                                 | yes | inside \f$ab > 0\f$; kink at sign changes |
     * | CentralLimiter   | \f$\tfrac12(a + b)\f$                                                                | no  | yes |
     *
     * All but CentralLimiter return 0 at a local extremum (\f$ab \le 0\f$), so the reconstruction is first order
     * there. MinMod is the most dissipative, Superbee the least. A smooth limiter helps the Newton iteration of an
     * implicit timestepper.
     *
     * A new limiter only has to satisfy this concept; it must also work for autodiff types, since the jacobian
     * differentiates through it:
     * @code
     * class MyLimiter
     * {
     * public:
     *   template <typename NumberType>
     *   static NumberType slope_limit(const NumberType &du_1,
     *                                 const NumberType &du_2)
     *   {
     *     // …
     *   }
     * };
     * static_assert(HasSlopeLimiter<MyLimiter>);
     * @endcode
     */
    // clang-format on
    template <typename T>
    concept HasSlopeLimiter = requires(double a, double b) {
      { T::slope_limit(a, b) } -> std::convertible_to<double>;
    };

    namespace limiter_utils
    {
      /// The sign of @p val: -1, 0 or +1.
      template <typename T> int sgn(T val) { return (T{} < val) - (val < T{}); }
    } // namespace limiter_utils

  } // namespace def
} // namespace DiFfRG
