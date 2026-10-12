#pragma once

// standard library
#include <chrono>
#include <cstddef>
#include <string>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Wall time in seconds of one output frame, by stage. Nested: potential_solves and fe_attach are part of
     * contributor; only total is the frame's cost.
     */
    struct FrameTimings {
      double contributor = 0.;      ///< the whole user contributor callback (attach + readouts)
      double potential_solves = 0.; ///< of which: internal::solve_potential
      /// of which: FEOutput::attach (copying the fields, waiting for a free buffer)
      double fe_attach = 0.;
      double build_patches = 0.;   ///< DataOut::build_patches
      double data_filter = 0.;     ///< DataOutFilter pass: write_filtered_data + fill_node_data
      double hdf5_write = 0.;      ///< group/dataset/attribute creation and writes
      double hdf5_open_close = 0.; ///< H5Fopen + H5Fflush + H5Fclose
      /// time blocked on a full HDF5 writer queue
      double hdf5_queue_wait = 0.;
      /// FEOutput::flush on the calling thread (with synchronous output: including the VTU write)
      double fe_flush = 0.;
      double csv = 0.;   ///< CsvOutput::flush
      double total = 0.; ///< the whole frame, from write_frame entry to flush_frame exit

      std::size_t potential_solve_count = 0;
      std::size_t hdf5_open_count = 0; ///< how many times the .h5 was opened for this frame

      /// Add another sink's contribution.
      FrameTimings &operator+=(const FrameTimings &other);
    };

    /// Totals and maxima of FrameTimings over all frames of a run.
    class OutputTimings
    {
    public:
      void add(const FrameTimings &frame);

      /// Record the writer thread's work, reported separately from the frame costs.
      void set_writer_totals(const FrameTimings &totals, bool asynchronous);

      /// Multi-line report: count, total, mean, max per stage. Empty if no frame was recorded.
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

    /// Adds the wall time of its scope, in seconds, to a double.
    class ScopedTimer
    {
    public:
      explicit ScopedTimer(double &sink) : sink(&sink), start(std::chrono::steady_clock::now()) {}
      /// A null @p sink measures nothing.
      explicit ScopedTimer(double *sink) : sink(sink), start(std::chrono::steady_clock::now()) {}
      ~ScopedTimer() { stop(); }

      ScopedTimer(const ScopedTimer &) = delete;
      ScopedTimer &operator=(const ScopedTimer &) = delete;

      /// Add the elapsed time now instead of at scope exit. Idempotent.
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
  } // namespace internal
} // namespace DiFfRG
