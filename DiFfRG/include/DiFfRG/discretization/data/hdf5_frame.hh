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
  namespace internal
  {
    /// The open file and cached top-level groups while one frame is written; closes the file on destruction.
    class HDF5FrameContext
    {
    public:
      /// Open @p path read-write.
      explicit HDF5FrameContext(const std::string &path, FrameTimings *timings = nullptr);
      ~HDF5FrameContext() noexcept;

      HDF5FrameContext(const HDF5FrameContext &) = delete;
      HDF5FrameContext &operator=(const HDF5FrameContext &) = delete;

      /// The top-level group @p name, opened on first use.
      DiFfRG::hdf5::Group &group(const std::string &name);
      DiFfRG::hdf5::Group &root() { return root_group; }

    private:
      DiFfRG::hdf5::File h5_file;
      DiFfRG::hdf5::Group root_group;
      std::map<std::string, DiFfRG::hdf5::Group> open_groups;
      FrameTimings *timings;
    };

    /// One deferred write; captures its data by value.
    using HDF5Action = std::function<void(HDF5FrameContext &)>;

    /// All writes of one output frame to one file. Holds no HDF5 handles, so it can be passed to the writer thread.
    struct HDF5Frame {
      std::string path;
      double time = 0.;
      std::vector<HDF5Action> actions;

      bool empty() const noexcept { return actions.empty(); }

      /// Open the file, run every action in order, close.
      void run(FrameTimings *timings = nullptr) const;
    };
  } // namespace internal
} // namespace DiFfRG
