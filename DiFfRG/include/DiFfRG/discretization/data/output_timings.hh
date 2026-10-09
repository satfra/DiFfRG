#pragma once

// standard library
#include <chrono>
#include <cstddef>
#include <string>

namespace DiFfRG
{
  /**
   * @brief Wall-clock cost of producing one output frame, split by where the time went.
   *
   * Frames are separated by many timesteps, so the handful of clock reads per frame is free
   * compared to what they measure. The buckets are always collected; only the per-frame log
   * line is gated on verbosity.
   *
   * The buckets are *nested*, not disjoint. `contributor` is an umbrella over everything the
   * user callback does, and both `potential_solves` and `fe_attach` happen inside it. Only
   * `total` is the frame cost; summing the rest double-counts on purpose so that each bucket
   * answers "how much of the frame went here" independently.
   *
   *   total
   *     contributor          the whole user callback
   *       potential_solves     of which: internal::solve_potential
   *       fe_attach            of which: staging solution copies + backpressure
   *     build_patches        }
   *     data_filter          }  everything below runs in flush_frame,
   *     hdf5_write           }  after the contributor has returned
   *     hdf5_open_close      }
   *     fe_flush             }
   *     csv                  }
   */
  struct FrameTimings {
    double contributor = 0.;      ///< the whole user contributor callback (attach + readouts)
    double potential_solves = 0.; ///< of which: internal::solve_potential
    /** Of which: FEOutput::attach -- copying each solution vector into the frame's staging
     * buffer, plus any backpressure wait for a retired writer thread. */
    double fe_attach = 0.;
    double build_patches = 0.;   ///< DataOut::build_patches
    double data_filter = 0.;     ///< DataOutFilter pass: write_filtered_data + fill_node_data
    double hdf5_write = 0.;      ///< group/dataset/attribute creation and writes
    double hdf5_open_close = 0.; ///< H5Fopen + H5Fflush + H5Fclose
    /** Time the producer spent blocked because the writer queue was full -- i.e. the disk, not
     * the integrator, was the limit. Zero when the writes run inline. */
    double hdf5_queue_wait = 0.;
    /** Handing the frame to a writer thread and releasing the previous one's DataOut. With
     * VTK off, or `asynchronous` off, no thread is involved and this is the synchronous
     * release (and, when async is off, the VTU write itself). */
    double fe_flush = 0.;
    double csv = 0.;   ///< CsvOutput::flush
    double total = 0.; ///< the whole frame, from write_frame entry to flush_frame exit

    std::size_t potential_solve_count = 0;
    std::size_t hdf5_open_count = 0; ///< how many times the .h5 was opened for this frame

    /** Accumulate another sink's contribution into this frame. */
    FrameTimings &operator+=(const FrameTimings &other);
  };

  /** @brief Running statistics over all frames of one run. */
  class OutputTimings
  {
  public:
    void add(const FrameTimings &frame);

    /**
     * @brief Record what the writer thread did, which is not part of any frame's cost.
     *
     * Reported on its own line rather than folded into the buckets: once the writes are on a
     * background thread they no longer delay the integrator, and adding them to the frame
     * total would say the opposite of what happened.
     */
    void set_writer_totals(const FrameTimings &totals, bool asynchronous);

    /** Multi-line report: count, total, mean, max per bucket. Empty when no frame was recorded. */
    std::string format() const;

    std::size_t frames() const noexcept { return n_frames; }
    const FrameTimings &totals() const noexcept { return sum; }
    const FrameTimings &maxima() const noexcept { return max; }

  private:
    std::size_t n_frames = 0;
    FrameTimings sum;
    FrameTimings max;
    FrameTimings writer;
    bool writer_asynchronous = false;
  };

  /**
   * @brief Adds the wall time of its scope to a double, in seconds.
   *
   * Deliberately not deal.II's Timer (which is heavier and tied to MPI) nor the timesteppers'
   * CalcDtTimer (which is stateful and millisecond-granular).
   */
  class ScopedTimer
  {
  public:
    explicit ScopedTimer(double &sink) : sink(&sink), start(std::chrono::steady_clock::now()) {}
    /** A null sink measures nothing, for call paths that may run without a timing target. */
    explicit ScopedTimer(double *sink) : sink(sink), start(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() { stop(); }

    ScopedTimer(const ScopedTimer &) = delete;
    ScopedTimer &operator=(const ScopedTimer &) = delete;

    /** Charge the elapsed time now instead of at scope exit. Idempotent. */
    void stop()
    {
      if (sink == nullptr) return;
      *sink += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      sink = nullptr;
    }

  private:
    double *sink;
    std::chrono::steady_clock::time_point start;
  };
} // namespace DiFfRG
