#pragma once

#include <hdf5lib/group.hh>
#include <hdf5lib/handle.hh>

#include <string>

namespace DiFfRG::hdf5
{
  enum class Access { ReadOnly, ReadWrite, Truncate };

  class File
  {
  public:
    File() = default;

    static File open(const std::string &path, Access mode)
    {
      hid_t f = H5I_INVALID_HID;
      switch (mode) {
      case Access::ReadOnly:
        f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        break;
      case Access::ReadWrite:
        f = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        break;
      case Access::Truncate:
        f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        break;
      }
      throw_if_negative(f, mode == Access::Truncate ? "H5Fcreate failed" : "H5Fopen failed");
      File out;
      out.h_ = Handle::take(f);
      return out;
    }

    bool is_open() const noexcept { return h_.valid(); }
    hid_t id() const noexcept { return h_.get(); }

    void close() noexcept { h_.reset(); }

    void flush()
    {
      if (h_.valid()) throw_if_negative(H5Fflush(h_.get(), H5F_SCOPE_GLOBAL), "H5Fflush failed");
    }

    Group root()
    {
      hid_t g = H5Gopen2(h_.get(), "/", H5P_DEFAULT);
      throw_if_negative(g, "H5Gopen2(\"/\") failed");
      return Group::take(g);
    }

  private:
    Handle h_;
  };
} // namespace DiFfRG::hdf5
