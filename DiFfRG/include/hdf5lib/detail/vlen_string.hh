#pragma once

#include <hdf5lib/datatype.hh>

#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

namespace DiFfRG::hdf5::detail
{
  /// Pack a contiguous span of std::string into the C array-of-cstr form HDF5
  /// expects for variable-length string writes.
  inline std::vector<const char *> pack_strings(const std::string *src, std::size_t n)
  {
    std::vector<const char *> out(n);
    for (std::size_t i = 0; i < n; ++i)
      out[i] = src[i].c_str();
    return out;
  }

  /// Read variable-length strings into a span. The HDF5 library allocates a
  /// fresh C string per element; we copy into std::strings and reclaim.
  /// `dset` is the dataset id; we always pass H5S_ALL for both memory and
  /// file selection but build an explicit dataspace describing the
  /// memory buffer for H5Treclaim (which rejects H5S_ALL).
  inline void read_vlen_strings(hid_t dset, hid_t /*mem_space*/, hid_t /*file_space*/, hid_t mem_type,
                                std::string *dst, std::size_t n)
  {
    std::vector<char *> raw(n, nullptr);
    throw_if_negative(H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data()),
                      "H5Dread (vlen string) failed");
    for (std::size_t i = 0; i < n; ++i)
      dst[i] = raw[i] != nullptr ? std::string(raw[i]) : std::string();

    // Reclaim the per-element allocations. H5Dvlen_reclaim was renamed in
    // HDF5 1.12; we use H5Treclaim, which is the supported spelling in 2.x.
    // H5Treclaim requires an explicit (non-H5S_ALL) dataspace.
    hid_t reclaim_space = H5Dget_space(dset);
    if (reclaim_space >= 0) {
      H5Treclaim(mem_type, reclaim_space, H5P_DEFAULT, raw.data());
      H5Sclose(reclaim_space);
    }
  }
} // namespace DiFfRG::hdf5::detail
