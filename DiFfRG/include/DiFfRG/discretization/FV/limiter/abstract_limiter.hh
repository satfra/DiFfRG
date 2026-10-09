#pragma once

// standard library
#include <concepts>

namespace DiFfRG
{
  namespace def
  {
    /**
     * @brief Concept that any slope-limiter mixin must satisfy.
     *
     * A limiter class must provide a static @c slope_limit method
     * accepting two slopes and returning a limited slope value.
     * The assembler uses this concept to produce clear compile
     * errors when a model does not provide a limiter.
     *
     * New limiters only need to satisfy this concept:
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
    template <typename T>
    concept HasSlopeLimiter = requires(double a, double b) {
      { T::slope_limit(a, b) } -> std::convertible_to<double>;
    };

    namespace limiter_utils
    {
      /**
       * @brief Signum helper: returns -1, 0 or +1.
       */
      template <typename T> int sgn(T val) { return (T{} < val) - (val < T{}); }
    } // namespace limiter_utils

  } // namespace def
} // namespace DiFfRG
