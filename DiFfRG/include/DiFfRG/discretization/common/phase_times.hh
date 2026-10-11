#pragma once

// standard library
#include <chrono>

namespace DiFfRG
{
  /**
   * @brief Wall time in seconds of the stages of the assemblers' residual or Jacobian calls, summed over all calls.
   *
   * The stages follow each other without gaps, so total() is the time of the calls themselves.
   */
  struct AssemblyPhaseTimes {
    /// Extractors (and, for a Jacobian, their derivatives); LDG: also the auxiliary levels and their Jacobians.
    double extract = 0.;
    /// The solution at the batch points (KT: the reconstruction and the traces).
    double gather = 0.;
    /// The model at all points: evaluate_batch and, for a jacobian, its AD.
    double evaluate = 0.;
    /// Contraction with the test functions and insertion into the global system (LDG: and the chain rule through
    /// the levels; KT: and the chain rule through the reconstruction).
    double scatter = 0.;
    /// Number of assembly calls.
    unsigned int calls = 0;

    /// Sum of all stages, in seconds.
    double total() const { return extract + gather + evaluate + scatter; }
  };

  namespace internal
  {
    /// Adds the consecutive stages of one assembly call to an AssemblyPhaseTimes.
    class PhaseTimer
    {
      using clock = std::chrono::steady_clock;

    public:
      explicit PhaseTimer(AssemblyPhaseTimes &times) : times(times), start(clock::now()), last(start) {}

      /// Add the time since the previous lap (or construction) to @p stage.
      void lap(double AssemblyPhaseTimes::*stage)
      {
        const auto now = clock::now();
        times.*stage += std::chrono::duration<double>(now - last).count();
        last = now;
      }
      /// Count the call and return its duration in seconds, up to the last lap.
      double finish()
      {
        ++times.calls;
        return std::chrono::duration<double>(last - start).count();
      }

    private:
      AssemblyPhaseTimes &times;
      const clock::time_point start;
      clock::time_point last;
    };
  } // namespace internal
} // namespace DiFfRG
