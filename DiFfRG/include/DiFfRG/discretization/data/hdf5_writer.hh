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
  /**
   * @brief Writes staged HDF5 frames on a single background thread.
   *
   * One writer serves every HDF5Output of a run. That is a correctness requirement, not a
   * tuning choice: HDF5 here is built without thread safety, so all of its API calls must be
   * serialised, and confining them to one thread is the cheapest way to guarantee that.
   *
   * Each frame is still opened, written, flushed and closed on its own, so the file stays
   * readable while the run continues and a crash leaves a complete prefix of frames behind.
   * What the thread buys is that the integrator no longer waits for any of it.
   *
   * Backpressure blocks rather than drops: a producer that outruns the disk waits, and no
   * scientific data is ever discarded.
   */
  class HDF5FrameWriter
  {
  public:
    /**
     * @param queue_depth How many frames may be in flight. 0 runs every frame inline on the
     * calling thread and starts no thread at all -- the synchronous behaviour, kept because a
     * hard abort can otherwise lose the frames still queued.
     */
    explicit HDF5FrameWriter(unsigned int queue_depth);
    ~HDF5FrameWriter() noexcept;

    HDF5FrameWriter(const HDF5FrameWriter &) = delete;
    HDF5FrameWriter &operator=(const HDF5FrameWriter &) = delete;

    bool asynchronous() const noexcept { return depth > 0; }

    /**
     * @brief Hand a frame over, blocking while the queue is full.
     *
     * Rethrows a stored worker error first, so a failure surfaces on the thread that can act on
     * it. `timings` is charged the blocking wait only -- the write itself is deliberately not
     * charged to the frame, because it no longer sits on the frame's critical path.
     */
    void submit(HDF5Frame frame, FrameTimings *timings = nullptr);

    /** Wait for the queue to empty, then rethrow the first worker error, if any. */
    void drain();

    /** Drain and join. Idempotent; safe to call before destruction. */
    void finish();

    /** Work done by the writer thread so far, off the critical path. */
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
} // namespace DiFfRG
