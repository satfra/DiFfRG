// DiFfRG
#include <DiFfRG/discretization/data/hdf5_input.hh>

namespace DiFfRG
{
  HDF5Input::HDF5Input(const std::string file_name)
      : file_name(has_suffix(file_name, ".h5") ? file_name : file_name + ".h5")
  {
    path = this->file_name;

    if (!std::filesystem::exists(path))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not exist.");

    // Read-only: HDF5 takes a shared lock for it, so every MPI rank can read the same file at once.
    // ReadWrite takes an exclusive lock, and all ranks but one then fail in H5Fopen (EAGAIN).
    h5_file = DiFfRG::hdf5::File::open(path.string(), DiFfRG::hdf5::Access::ReadOnly);

    auto root = h5_file.root();
    if (!root.has_group("scalars"))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not contain a 'scalars' group.");
    if (!root.has_group("maps"))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not contain a 'maps' group.");
    if (!root.has_group("coordinates"))
      throw std::runtime_error("HDF5Input: The file '" + this->file_name + "' does not contain a 'coordinates' group.");

    scalars = root.open_group("scalars");
    maps = root.open_group("maps");
    coords = root.open_group("coordinates");
  }

  DiFfRG::hdf5::File &HDF5Input::get_file() { return h5_file; }
} // namespace DiFfRG
