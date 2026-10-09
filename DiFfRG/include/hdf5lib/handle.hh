#pragma once

#include <hdf5lib/detail/hdf5_capi.hh>

#include <stdexcept>
#include <string>
#include <utility>

namespace DiFfRG::hdf5
{
  inline void throw_if_negative(hid_t id, const char *what)
  {
    if (id < 0) throw std::runtime_error(std::string("hdf5lib: ") + what);
  }

  inline void throw_if_negative(herr_t err, const char *what)
  {
    if (err < 0) throw std::runtime_error(std::string("hdf5lib: ") + what);
  }

  /// RAII wrapper around an HDF5 hid_t. Uses H5I ref counting so it can hold
  /// any HDF5 identifier (file, group, dataset, datatype, dataspace, attribute,
  /// property list).
  class Handle
  {
  public:
    Handle() = default;

    explicit Handle(hid_t id) : id_(id)
    {
      if (id_ >= 0) H5Iinc_ref(id_);
    }

    /// Take ownership of an id without bumping its refcount. Used when
    /// constructing from `H5Fcreate`/`H5Dcreate2` etc. that already gave us
    /// a fresh reference.
    static Handle take(hid_t id)
    {
      Handle h;
      h.id_ = id;
      return h;
    }

    Handle(const Handle &o) : id_(o.id_)
    {
      if (id_ >= 0) H5Iinc_ref(id_);
    }

    Handle(Handle &&o) noexcept : id_(o.id_) { o.id_ = H5I_INVALID_HID; }

    Handle &operator=(const Handle &o)
    {
      if (this == &o) return *this;
      reset();
      id_ = o.id_;
      if (id_ >= 0) H5Iinc_ref(id_);
      return *this;
    }

    Handle &operator=(Handle &&o) noexcept
    {
      if (this == &o) return *this;
      reset();
      id_ = o.id_;
      o.id_ = H5I_INVALID_HID;
      return *this;
    }

    ~Handle() { reset(); }

    void reset() noexcept
    {
      if (id_ >= 0) {
        H5Idec_ref(id_);
        id_ = H5I_INVALID_HID;
      }
    }

    hid_t get() const noexcept { return id_; }
    bool valid() const noexcept { return id_ >= 0; }

    hid_t release() noexcept
    {
      hid_t t = id_;
      id_ = H5I_INVALID_HID;
      return t;
    }

  private:
    hid_t id_ = H5I_INVALID_HID;
  };
} // namespace DiFfRG::hdf5
