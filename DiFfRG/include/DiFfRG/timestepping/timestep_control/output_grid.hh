#pragma once

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief The output times \f$t_\text{start} + n\, \Delta t_\text{out}\f$, \f$n \ge 1\f$, of a stepper that writes
     * interpolated frames; the frame at \f$t_\text{start}\f$ is written separately.
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

    private:
      const double t_start, output_dt;
      unsigned int n = 1;
    };
  } // namespace internal
} // namespace DiFfRG
