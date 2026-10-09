#include <DiFfRG/discretization/data/hdf5_writer.hh>

#include <stdexcept>
#include <string>
#include <utility>

namespace DiFfRG
{
  HDF5FrameWriter::HDF5FrameWriter(const unsigned int queue_depth) : depth(queue_depth)
  {
    if (depth > 0) worker = std::thread([this]() { loop(); });
  }

  HDF5FrameWriter::~HDF5FrameWriter() noexcept
  {
    try {
      finish();
    } catch (...) {
      // A destructor must not terminate the process. Run boundaries call drain()/finish()
      // explicitly, which is where a worker error is meant to surface.
    }
  }

  void HDF5FrameWriter::rethrow_stored()
  {
    if (!stored_exception) return;
    auto to_throw = stored_exception;
    stored_exception = nullptr;
    std::rethrow_exception(to_throw);
  }

  void HDF5FrameWriter::submit(HDF5Frame frame, FrameTimings *timings)
  {
    if (frame.empty()) return;

    if (depth == 0) {
      // Synchronous: the write is on the critical path, so charge it to the frame.
      frame.run(timings);
      return;
    }

    std::unique_lock<std::mutex> lock(mutex);
    rethrow_stored();

    if (queue.size() >= depth) {
      // Waiting here means the disk, not the integrator, is the limit. Worth measuring.
      ScopedTimer wait_timer(timings != nullptr ? &timings->hdf5_queue_wait : nullptr);
      producer_condition.wait(lock, [this]() { return queue.size() < depth || stored_exception != nullptr; });
      wait_timer.stop();
      rethrow_stored();
    }

    queue.push_back(std::move(frame));
    lock.unlock();
    consumer_condition.notify_one();
  }

  void HDF5FrameWriter::loop()
  {
    for (;;) {
      HDF5Frame frame;
      {
        std::unique_lock<std::mutex> lock(mutex);
        consumer_condition.wait(lock, [this]() { return stopping || !queue.empty(); });
        if (queue.empty()) {
          if (stopping) return;
          continue;
        }
        frame = std::move(queue.front());
        queue.pop_front();
        writing = true;
      }
      // Notify here rather than after the write: the producer only cares that a queue slot is
      // free, and making it wait for the write to finish would defeat the whole arrangement.
      producer_condition.notify_one();

      std::exception_ptr failure;
      FrameTimings frame_timings;
      try {
        frame.run(&frame_timings);
      } catch (const std::exception &error) {
        // Without the frame's time the caller sees a failure with no way to tell which frame
        // produced it -- the write is several frames behind the main thread by then.
        failure = std::make_exception_ptr(std::runtime_error(
            "HDF5 frame at t = " + std::to_string(frame.time) + " failed to write: " + error.what()));
      } catch (...) {
        failure = std::current_exception();
      }

      {
        std::lock_guard<std::mutex> lock(mutex);
        worker_timings += frame_timings;
        if (failure && !stored_exception) stored_exception = failure;
        writing = false;
      }
      // Both a waiting producer and a waiting drain() need to hear about this.
      producer_condition.notify_all();
    }
  }

  void HDF5FrameWriter::drain()
  {
    if (depth == 0) return;

    std::unique_lock<std::mutex> lock(mutex);
    producer_condition.wait(lock, [this]() { return queue.empty() && !writing; });
    rethrow_stored();
  }

  void HDF5FrameWriter::finish()
  {
    if (depth == 0 || finished) {
      std::lock_guard<std::mutex> lock(mutex);
      rethrow_stored();
      return;
    }

    std::exception_ptr drain_error;
    try {
      drain();
    } catch (...) {
      drain_error = std::current_exception();
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      finished = true;
    }
    consumer_condition.notify_all();
    if (worker.joinable()) worker.join();

    if (drain_error) std::rethrow_exception(drain_error);

    std::lock_guard<std::mutex> lock(mutex);
    rethrow_stored();
  }

  FrameTimings HDF5FrameWriter::worker_totals() const
  {
    std::lock_guard<std::mutex> lock(mutex);
    return worker_timings;
  }
} // namespace DiFfRG
