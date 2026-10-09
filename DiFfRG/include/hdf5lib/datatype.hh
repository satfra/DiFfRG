#pragma once

#include <hdf5lib/handle.hh>

#include <cstddef>
#include <cstdint>
#include <string>

namespace DiFfRG::hdf5
{
  class Datatype
  {
  public:
    Datatype() = default;
    explicit Datatype(hid_t id) : h_(id) {}
    static Datatype take(hid_t id) { return Datatype(Handle::take(id)); }

    hid_t id() const noexcept { return h_.get(); }
    bool valid() const noexcept { return h_.valid(); }

    bool operator==(const Datatype &other) const
    {
      if (!valid() || !other.valid()) return valid() == other.valid();
      htri_t r = H5Tequal(h_.get(), other.h_.get());
      throw_if_negative(static_cast<herr_t>(r < 0 ? -1 : 0), "H5Tequal failed");
      return r > 0;
    }

    bool operator!=(const Datatype &other) const { return !(*this == other); }

    /// Build a compound datatype with the given total size (in bytes).
    static Datatype compound(std::size_t size)
    {
      hid_t id = H5Tcreate(H5T_COMPOUND, size);
      throw_if_negative(id, "H5Tcreate(H5T_COMPOUND) failed");
      return Datatype::take(id);
    }

    /// Insert a member into a compound datatype.
    void insert(const std::string &name, std::size_t offset, const Datatype &member)
    {
      throw_if_negative(H5Tinsert(h_.get(), name.c_str(), offset, member.id()), "H5Tinsert failed");
    }

  private:
    explicit Datatype(Handle h) : h_(std::move(h)) {}
    Handle h_;
  };

  /// Primary template — must be specialised for each supported C++ type.
  /// Specialisations live in this header (builtins) or where the user type
  /// is defined (compound types).
  template <class T> struct TypeTrait;

  /// Convenience factory — `type_of<T>()` returns the HDF5 datatype for `T`.
  template <class T> Datatype type_of() { return TypeTrait<T>::get(); }

#define DIFFRG_HDF5LIB_PREDEFINED_TYPE(CXX_TYPE, H5_PREDEF)                                                            \
  template <> struct TypeTrait<CXX_TYPE> {                                                                             \
    static Datatype get() { return Datatype(H5_PREDEF); }                                                              \
  }

  // Native C/C++ scalar types. Using NATIVE_* means HDF5 will translate to
  // file format on write; storage is portable across platforms.
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(char, H5T_NATIVE_CHAR);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(signed char, H5T_NATIVE_SCHAR);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(unsigned char, H5T_NATIVE_UCHAR);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(short, H5T_NATIVE_SHORT);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(unsigned short, H5T_NATIVE_USHORT);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(int, H5T_NATIVE_INT);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(unsigned int, H5T_NATIVE_UINT);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(long, H5T_NATIVE_LONG);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(unsigned long, H5T_NATIVE_ULONG);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(long long, H5T_NATIVE_LLONG);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(unsigned long long, H5T_NATIVE_ULLONG);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(float, H5T_NATIVE_FLOAT);
  DIFFRG_HDF5LIB_PREDEFINED_TYPE(double, H5T_NATIVE_DOUBLE);

#undef DIFFRG_HDF5LIB_PREDEFINED_TYPE

  /// Variable-length UTF-8 string.
  template <> struct TypeTrait<std::string> {
    static Datatype get()
    {
      hid_t t = H5Tcopy(H5T_C_S1);
      throw_if_negative(t, "H5Tcopy(H5T_C_S1) failed");
      throw_if_negative(H5Tset_size(t, H5T_VARIABLE), "H5Tset_size(H5T_VARIABLE) failed");
      throw_if_negative(H5Tset_cset(t, H5T_CSET_UTF8), "H5Tset_cset(UTF8) failed");
      return Datatype::take(t);
    }
  };
} // namespace DiFfRG::hdf5
