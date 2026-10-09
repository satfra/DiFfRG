#pragma once

#include <hdf5lib/dataset.hh>
#include <hdf5lib/dataspace.hh>
#include <hdf5lib/datatype.hh>
#include <hdf5lib/detail/vlen_string.hh>
#include <hdf5lib/handle.hh>

#include <string>
#include <type_traits>
#include <vector>

namespace DiFfRG::hdf5
{
  class Group
  {
  public:
    Group() = default;
    explicit Group(hid_t id) : h_(id) {}
    static Group take(hid_t id) { return Group(Handle::take(id)); }

    hid_t id() const noexcept { return h_.get(); }
    bool valid() const noexcept { return h_.valid(); }

    // ----------------- subgroups -----------------

    Group create_group(const std::string &name)
    {
      hid_t g = H5Gcreate2(h_.get(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      throw_if_negative(g, "H5Gcreate2 failed");
      return Group::take(g);
    }

    Group open_group(const std::string &name)
    {
      hid_t g = H5Gopen2(h_.get(), name.c_str(), H5P_DEFAULT);
      throw_if_negative(g, "H5Gopen2 failed");
      return Group::take(g);
    }

    bool has_group(const std::string &name) const { return child_kind(name) == H5O_TYPE_GROUP; }
    bool has_dataset(const std::string &name) const { return child_kind(name) == H5O_TYPE_DATASET; }

    // ----------------- datasets -----------------

    Dataset create_dataset(const std::string &name, const Datatype &type, const Dataspace &space)
    {
      hid_t d = H5Dcreate2(h_.get(), name.c_str(), type.id(), space.id(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      throw_if_negative(d, "H5Dcreate2 failed");
      return Dataset::take(d);
    }

    /// Create a chunked dataset. Pass an unlimited Dataspace for an
    /// appendable dataset.
    Dataset create_chunked_dataset(const std::string &name, const Datatype &type, const Dataspace &space,
                                   const Dims &chunk)
    {
      hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
      throw_if_negative(dcpl, "H5Pcreate(H5P_DATASET_CREATE) failed");
      herr_t e = H5Pset_chunk(dcpl, static_cast<int>(chunk.size()), chunk.data());
      if (e < 0) {
        H5Pclose(dcpl);
        throw std::runtime_error("hdf5lib: H5Pset_chunk failed");
      }
      hid_t d = H5Dcreate2(h_.get(), name.c_str(), type.id(), space.id(), H5P_DEFAULT, dcpl, H5P_DEFAULT);
      H5Pclose(dcpl);
      throw_if_negative(d, "H5Dcreate2 (chunked) failed");
      return Dataset::take(d);
    }

    Dataset open_dataset(const std::string &name)
    {
      hid_t d = H5Dopen2(h_.get(), name.c_str(), H5P_DEFAULT);
      throw_if_negative(d, "H5Dopen2 failed");
      return Dataset::take(d);
    }

    // ----------------- soft link -----------------

    void create_soft_link(const std::string &name, const std::string &target_path)
    {
      throw_if_negative(
          H5Lcreate_soft(target_path.c_str(), h_.get(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT),
          "H5Lcreate_soft failed");
    }

    // ----------------- attributes -----------------

    template <class T>
      requires(!std::is_same_v<std::decay_t<T>, std::string>)
    void write_attribute(const std::string &name, const T &value)
    {
      auto type = type_of<std::decay_t<T>>();
      auto space = Dataspace::scalar();
      hid_t a = H5Acreate2(h_.get(), name.c_str(), type.id(), space.id(), H5P_DEFAULT, H5P_DEFAULT);
      throw_if_negative(a, "H5Acreate2 failed");
      herr_t e = H5Awrite(a, type.id(), &value);
      H5Aclose(a);
      throw_if_negative(e, "H5Awrite failed");
    }

    void write_attribute(const std::string &name, const std::string &value)
    {
      auto type = type_of<std::string>();
      auto space = Dataspace::scalar();
      hid_t a = H5Acreate2(h_.get(), name.c_str(), type.id(), space.id(), H5P_DEFAULT, H5P_DEFAULT);
      throw_if_negative(a, "H5Acreate2 (string) failed");
      const char *cstr = value.c_str();
      herr_t e = H5Awrite(a, type.id(), &cstr);
      H5Aclose(a);
      throw_if_negative(e, "H5Awrite (string) failed");
    }

    void write_attribute(const std::string &name, const char *value)
    {
      write_attribute(name, std::string(value));
    }

    bool has_attribute(const std::string &name) const
    {
      htri_t e = H5Aexists(h_.get(), name.c_str());
      throw_if_negative(e, "H5Aexists failed");
      return e > 0;
    }

    void delete_attribute(const std::string &name)
    {
      throw_if_negative(H5Adelete(h_.get(), name.c_str()), "H5Adelete failed");
    }

    /// Write an attribute, replacing any existing one of that name.
    ///
    /// Deliberately separate from write_attribute, which fails on a name that is already taken --
    /// for most attributes a second write is a bug, and that throw is the only thing that catches
    /// it. Use this only where a value is meant to change over the file's life.
    template <class T> void overwrite_attribute(const std::string &name, const T &value)
    {
      if (has_attribute(name)) delete_attribute(name);
      write_attribute(name, value);
    }

    template <class T>
      requires(!std::is_same_v<std::decay_t<T>, std::string>)
    T read_attribute(const std::string &name) const
    {
      hid_t a = H5Aopen(h_.get(), name.c_str(), H5P_DEFAULT);
      throw_if_negative(a, "H5Aopen failed");
      auto type = type_of<std::decay_t<T>>();
      T value{};
      herr_t e = H5Aread(a, type.id(), &value);
      H5Aclose(a);
      throw_if_negative(e, "H5Aread failed");
      return value;
    }

    template <class T>
      requires(std::is_same_v<std::decay_t<T>, std::string>)
    std::string read_attribute(const std::string &name) const
    {
      hid_t a = H5Aopen(h_.get(), name.c_str(), H5P_DEFAULT);
      throw_if_negative(a, "H5Aopen (string) failed");
      auto type = type_of<std::string>();
      char *raw = nullptr;
      herr_t e = H5Aread(a, type.id(), &raw);
      if (e < 0) {
        H5Aclose(a);
        throw std::runtime_error("hdf5lib: H5Aread (string) failed");
      }
      std::string out = raw != nullptr ? std::string(raw) : std::string();
      // Reclaim the per-element allocation for the variable-length string.
      hid_t scalar = H5Screate(H5S_SCALAR);
      H5Treclaim(type.id(), scalar, H5P_DEFAULT, &raw);
      H5Sclose(scalar);
      H5Aclose(a);
      return out;
    }

    // ----------------- iteration -----------------

    /// Names of all immediate children, in link-name order.
    std::vector<std::string> child_names() const
    {
      std::vector<std::string> out;
      auto cb = [](hid_t /*loc*/, const char *name, const H5L_info2_t * /*info*/, void *op_data) -> herr_t {
        auto &vec = *static_cast<std::vector<std::string> *>(op_data);
        vec.emplace_back(name);
        return 0;
      };
      hsize_t idx = 0;
      throw_if_negative(H5Literate2(h_.get(), H5_INDEX_NAME, H5_ITER_NATIVE, &idx, cb, &out),
                        "H5Literate2 failed");
      return out;
    }

    bool child_is_group(const std::string &name) const { return child_kind(name) == H5O_TYPE_GROUP; }
    bool child_is_dataset(const std::string &name) const { return child_kind(name) == H5O_TYPE_DATASET; }

  private:
    explicit Group(Handle h) : h_(std::move(h)) {}
    Handle h_;

    /// H5O_TYPE_UNKNOWN if the child does not exist (or is not an object).
    H5O_type_t child_kind(const std::string &name) const
    {
      if (H5Lexists(h_.get(), name.c_str(), H5P_DEFAULT) <= 0) return H5O_TYPE_UNKNOWN;
      H5O_info2_t info;
      if (H5Oget_info_by_name3(h_.get(), name.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT) < 0)
        return H5O_TYPE_UNKNOWN;
      return info.type;
    }
  };
} // namespace DiFfRG::hdf5
