#pragma once

// standard library
#include <chrono>

namespace DiFfRG
{
  /**
   * @brief Wall time of the stages of the batched assemblers' residual or jacobian calls, summed over all calls.
   *
   * The stages follow each other without gaps, so total() is the time of the calls themselves.
   */
  struct AssemblyPhaseTimes {
    /// The extractors and, for a jacobian, their derivatives; under LDG also building the levels (which extract()
    /// does) and their jacobians.
    double extract = 0.;
    /// The solution at the batch points (KT: the reconstruction and the traces).
    double gather = 0.;
    /// The model at all points: evaluate_batch and, for a jacobian, its AD.
    double evaluate = 0.;
    /// Contraction with the test functions and insertion into the global system (LDG: and the chain rule through
    /// the levels; KT: and the chain rule through the reconstruction).
    double scatter = 0.;
    unsigned int calls = 0;

    double total() const { return extract + gather + evaluate + scatter; }
  };

  namespace internal
  {
    /**
     * @brief Books the consecutive stages of one assembly call into an AssemblyPhaseTimes.
     *
     * One clock read per stage boundary and no CPU time, so the timing stays negligible even for small calls on a
     * machine where reading the clock is a system call.
     */
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
