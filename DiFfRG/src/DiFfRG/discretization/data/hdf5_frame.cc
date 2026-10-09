#include <DiFfRG/discretization/data/hdf5_frame.hh>

namespace DiFfRG
{
  HDF5FrameContext::HDF5FrameContext(const std::string &path, FrameTimings *timings) : timings(timings)
  {
    if (timings != nullptr) ++timings->hdf5_open_count;
    ScopedTimer open_timer(timings != nullptr ? &timings->hdf5_open_close : nullptr);
    h5_file = DiFfRG::hdf5::File::open(path, DiFfRG::hdf5::Access::ReadWrite);
    root_group = h5_file.root();
  }

  HDF5FrameContext::~HDF5FrameContext() noexcept
  {
    try {
      ScopedTimer close_timer(timings != nullptr ? &timings->hdf5_open_close : nullptr);
      // Release every cached child handle first. HDF5 reference-counts the file through them,
      // so leaving one alive keeps the file open at the OS level after H5Fclose -- holding the
      // lock a reader needs and leaving on-disk metadata unflushed.
      open_groups.clear();
      root_group = DiFfRG::hdf5::Group{};
      h5_file.flush();
      h5_file.close();
    } catch (...) {
      // A destructor must not terminate the process. The frame's caller already sees any
      // failure raised by the actions themselves.
    }
  }

  DiFfRG::hdf5::Group &HDF5FrameContext::group(const std::string &name)
  {
    const auto found = open_groups.find(name);
    if (found != open_groups.end()) return found->second;
    return open_groups.emplace(name, root_group.open_group(name)).first->second;
  }

  void HDF5Frame::run(FrameTimings *timings) const
  {
    if (actions.empty()) return;

    HDF5FrameContext context(path, timings);
    ScopedTimer write_timer(timings != nullptr ? &timings->hdf5_write : nullptr);
    for (const auto &action : actions)
      action(context);
  }
} // namespace DiFfRG
