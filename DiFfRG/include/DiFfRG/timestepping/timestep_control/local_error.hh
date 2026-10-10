#pragma once

// standard library
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace DiFfRG
{
  namespace internal
  {
    /// Thrown by a step whose estimated local error is too large; the step controller retries it with a smaller step.
    struct StepRejected : public std::runtime_error {
      using std::runtime_error::runtime_error;
    };

    /**
     * @brief Weighted RMS norm of a local error estimate \f$e\f$ between the states \f$a\f$ and \f$b\f$,
     * \f[ \sqrt{\frac1N \sum_i \Big(\frac{e_i}{\text{atol} + \text{rtol} \max(|a_i|, |b_i|)}\Big)^2}\,. \f]
     * A step is accurate enough if this is at most 1.
     */
    template <typename VectorType>
    double weighted_rms(const VectorType &e, const VectorType &a, const VectorType &b, const double atol,
                        const double rtol)
    {
      double sum = 0.;
      for (unsigned int i = 0; i < e.size(); ++i) {
        const double w = atol + rtol * std::max(std::abs(a[i]), std::abs(b[i]));
        sum += (e[i] / w) * (e[i] / w);
      }
      return e.size() > 0 ? std::sqrt(sum / e.size()) : 0.;
    }
  } // namespace internal
} // namespace DiFfRG
