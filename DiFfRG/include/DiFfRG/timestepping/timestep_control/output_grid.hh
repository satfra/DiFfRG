#pragma once

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief The output times \f$t_\text{start} + n\, \Delta t_\text{out}\f$, \f$n \ge 1\f$, of a stepper that writes
     * interpolated frames, and of the final frame (finish()); the frame at \f$t_\text{start}\f$ is written separately.
     */
    class OutputGrid
    {
    public:
      OutputGrid(const double t_start, const double output_dt) : t_start(t_start), output_dt(output_dt) {}

      /// Call `write(t_out)` for every output time up to @p t_reached that was not written yet, in order.
      template <typename Write> void advance_to(const double t_reached, Write &&write)
      {
        // The slack absorbs the rounding of the accumulated step times.
        while (t_start + n * output_dt <= t_reached + 1e-10 * output_dt)
          write(t_start + n++ * output_dt);
      }

      /// After the last step: call `write(t_end)` unless the last output time already is @p t_end.
      template <typename Write> void finish(const double t_end, Write &&write)
      {
        if (t_end > t_start + (n - 1) * output_dt + 1e-10 * output_dt) write(t_end);
      }

    private:
      const double t_start, output_dt;
      unsigned int n = 1;
    };
  } // namespace internal
} // namespace DiFfRG
