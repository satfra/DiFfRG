#pragma once

// DiFfRG
#include <DiFfRG/discretization/data/output_timings.hh>

#include <hdf5lib/hdf5.hh>

// standard library
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief The one place HDF5 identifiers live while a frame is being written.
   *
   * Opens the file, hands out cached top-level group handles for the duration of the frame, and
   * flushes and closes on destruction -- including when an action throws, so a failed frame
   * still leaves a closed, readable file behind.
   *
   * Caching the group handles matters: a frame writes one dataset per scalar, and reopening
   * "scalars" for each of them would turn a handful of writes into a handful of group lookups.
   *
   * The context is deliberately not copyable and not shareable. HDF5 in this build is compiled
   * without thread safety (`HDF5_ENABLE_THREADSAFE=OFF`, and the system HDF5 in use likewise
   * leaves `H5_HAVE_THREADSAFE` undefined), so every HDF5 call in the process must be
   * serialised. Confining the identifiers to one short-lived object owned by whoever executes
   * the frame is what makes that a structural property rather than a review rule.
   */
  class HDF5FrameContext
  {
  public:
    /** Opens `path` read-write. Charges the open to `timings` if one is given. */
    explicit HDF5FrameContext(const std::string &path, FrameTimings *timings = nullptr);
    ~HDF5FrameContext() noexcept;

    HDF5FrameContext(const HDF5FrameContext &) = delete;
    HDF5FrameContext &operator=(const HDF5FrameContext &) = delete;

    /** A cached handle to a top-level group, opened on first use. */
    DiFfRG::hdf5::Group &group(const std::string &name);
    DiFfRG::hdf5::Group &root() { return root_group; }

  private:
    DiFfRG::hdf5::File h5_file;
    DiFfRG::hdf5::Group root_group;
    std::map<std::string, DiFfRG::hdf5::Group> open_groups;
    FrameTimings *timings;
  };

  /**
   * One write, deferred. Closures capture their payload **by value**: callers hand us pointers
   * into live simulation state (`Examples/YangMills/Full/model.hh` passes
   * `&variables.data()[idxv("ZA")]`) which the next timestep overwrites.
   */
  using HDF5Action = std::function<void(HDF5FrameContext &)>;

  /**
   * @brief Everything one output frame will write to one file, and nothing else.
   *
   * Holds no HDF5 identifiers, so it can be moved around -- and, once the writer thread exists,
   * handed to it -- without any HDF5 state crossing a thread boundary.
   */
  struct HDF5Frame {
    std::string path;
    double time = 0.;
    std::vector<HDF5Action> actions;

    bool empty() const noexcept { return actions.empty(); }

    /** Open the file, run every action in order, then flush and close. */
    void run(FrameTimings *timings = nullptr) const;
  };
} // namespace DiFfRG
