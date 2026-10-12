#pragma once

// DiFfRG
#include <DiFfRG/discretization/data/hdf5_frame.hh>
#include <DiFfRG/discretization/data/output_timings.hh>

// standard library
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Writes HDF5 frames on one background thread, shared by all HDF5Outputs of a run (HDF5 is not thread-safe).
     *
     * Each frame opens, writes and closes the file, so the file stays readable during the run. A full queue blocks the
     * producer; no frame is dropped.
     */
    class HDF5FrameWriter
    {
    public:
      /// @param queue_depth Maximum frames in flight; 0 writes every frame on the calling thread.
      explicit HDF5FrameWriter(unsigned int queue_depth);
      ~HDF5FrameWriter() noexcept;

      HDF5FrameWriter(const HDF5FrameWriter &) = delete;
      HDF5FrameWriter &operator=(const HDF5FrameWriter &) = delete;

      bool asynchronous() const noexcept { return depth > 0; }

      /// Queue a frame, blocking while the queue is full; rethrows an earlier writer error.
      void submit(HDF5Frame frame, FrameTimings *timings = nullptr);

      /// Wait until the queue is empty; rethrow the first writer error.
      void drain();

      /// Drain and stop the thread. Idempotent.
      void finish();

      /// Time spent by the writer thread so far.
      FrameTimings worker_totals() const;

    private:
      void loop();
      void rethrow_stored();

      const unsigned int depth;

      mutable std::mutex mutex;
      std::condition_variable producer_condition, consumer_condition;
      std::deque<HDF5Frame> queue;
      bool stopping = false;
      bool writing = false; ///< a frame has been popped but not finished
      bool finished = false;

      std::exception_ptr stored_exception;
      FrameTimings worker_timings;

      std::thread worker;
    };
  } // namespace internal
} // namespace DiFfRG
