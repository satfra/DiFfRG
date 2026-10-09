#pragma once

#include <hdf5lib/dataspace.hh>
#include <hdf5lib/datatype.hh>
#include <hdf5lib/detail/vlen_string.hh>
#include <hdf5lib/handle.hh>

#include <string>
#include <type_traits>
#include <vector>

namespace DiFfRG::hdf5
{
  class Dataset
  {
  public:
    Dataset() = default;
    explicit Dataset(hid_t id) : h_(id) {}
    static Dataset take(hid_t id) { return Dataset(Handle::take(id)); }

    hid_t id() const noexcept { return h_.get(); }
    bool valid() const noexcept { return h_.valid(); }

    Datatype datatype() const
    {
      hid_t t = H5Dget_type(h_.get());
      throw_if_negative(t, "H5Dget_type failed");
      return Datatype::take(t);
    }

    Dataspace dataspace() const
    {
      hid_t s = H5Dget_space(h_.get());
      throw_if_negative(s, "H5Dget_space failed");
      return Dataspace::take(s);
    }

    void resize(const Dims &dims)
    {
      throw_if_negative(H5Dset_extent(h_.get(), dims.data()), "H5Dset_extent failed");
    }

    // ----------------- write -----------------

    /// Write a single value into the dataset's full extent (must currently be
    /// shaped as a scalar OR a single-element simple space).
    template <class T>
      requires(!std::is_same_v<std::decay_t<T>, std::string>)
    void write(const T &value)
    {
      auto type = type_of<std::decay_t<T>>();
      throw_if_negative(H5Dwrite(h_.get(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &value),
                        "H5Dwrite (scalar) failed");
    }

    /// Write a single value into the element selected by `[offset, offset+1)`.
    /// This is the "append to chunked unlimited" pattern. Caller is
    /// responsible for resizing the dataset first.
    template <class T>
      requires(!std::is_same_v<std::decay_t<T>, std::string>)
    void write_at(hsize_t offset, const T &value)
    {
      auto file_space = dataspace();
      file_space.select_hyperslab({offset}, {1});
      hsize_t one = 1;
      hid_t mem_space = H5Screate_simple(1, &one, nullptr);
      throw_if_negative(mem_space, "H5Screate_simple (write_at mem) failed");
      auto type = type_of<std::decay_t<T>>();
      herr_t e = H5Dwrite(h_.get(), type.id(), mem_space, file_space.id(), H5P_DEFAULT, &value);
      H5Sclose(mem_space);
      throw_if_negative(e, "H5Dwrite (write_at) failed");
    }

    /// Variable-length string write into a 1-element selection.
    void write_at(hsize_t offset, const std::string &value)
    {
      auto file_space = dataspace();
      file_space.select_hyperslab({offset}, {1});
      hsize_t one = 1;
      hid_t mem_space = H5Screate_simple(1, &one, nullptr);
      throw_if_negative(mem_space, "H5Screate_simple (write_at mem, string) failed");
      auto type = type_of<std::string>();
      const char *cstr = value.c_str();
      herr_t e = H5Dwrite(h_.get(), type.id(), mem_space, file_space.id(), H5P_DEFAULT, &cstr);
      H5Sclose(mem_space);
      throw_if_negative(e, "H5Dwrite (write_at, string) failed");
    }

    /// Whole-dataset write of a single string.
    void write(const std::string &value)
    {
      auto type = type_of<std::string>();
      const char *cstr = value.c_str();
      throw_if_negative(H5Dwrite(h_.get(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &cstr),
                        "H5Dwrite (string scalar) failed");
    }

    /// Vector write — element count must match the dataset's element count.
    template <class T>
      requires(!std::is_same_v<std::decay_t<T>, std::string>)
    void write(const std::vector<T> &v)
    {
      write(v.data(), v.size());
    }

    /// Vector of strings.
    void write(const std::vector<std::string> &v)
    {
      auto type = type_of<std::string>();
      auto packed = detail::pack_strings(v.data(), v.size());
      throw_if_negative(H5Dwrite(h_.get(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, packed.data()),
                        "H5Dwrite (vector<string>) failed");
    }

    /// Raw-pointer write — covers the (former) ArrayAdapter path. The caller
    /// promises that the buffer holds `n` `T` elements that match the
    /// dataset's selected element count.
    template <class T>
      requires(!std::is_same_v<std::decay_t<T>, std::string>)
    void write(const T *data, std::size_t /*n*/)
    {
      auto type = type_of<std::decay_t<T>>();
      throw_if_negative(H5Dwrite(h_.get(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
                        "H5Dwrite (raw pointer) failed");
    }

    // ----------------- read -----------------

    template <class T>
      requires(!std::is_same_v<std::decay_t<T>, std::string>)
    void read(T &value)
    {
      auto type = type_of<std::decay_t<T>>();
      throw_if_negative(H5Dread(h_.get(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &value),
                        "H5Dread (scalar) failed");
    }

    void read(std::string &value)
    {
      auto type = type_of<std::string>();
      detail::read_vlen_strings(h_.get(), H5S_ALL, H5S_ALL, type.id(), &value, 1);
    }

    template <class T>
      requires(!std::is_same_v<std::decay_t<T>, std::string>)
    void read(std::vector<T> &v)
    {
      read(v.data(), v.size());
    }

    void read(std::vector<std::string> &v)
    {
      auto type = type_of<std::string>();
      detail::read_vlen_strings(h_.get(), H5S_ALL, H5S_ALL, type.id(), v.data(), v.size());
    }

    template <class T>
      requires(!std::is_same_v<std::decay_t<T>, std::string>)
    void read(T *data, std::size_t /*n*/)
    {
      auto type = type_of<std::decay_t<T>>();
      throw_if_negative(H5Dread(h_.get(), type.id(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data),
                        "H5Dread (raw pointer) failed");
    }

  private:
    explicit Dataset(Handle h) : h_(std::move(h)) {}
    Handle h_;
  };
} // namespace DiFfRG::hdf5
