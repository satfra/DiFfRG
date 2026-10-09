#pragma once

#include <hdf5lib/handle.hh>

#include <vector>

namespace DiFfRG::hdf5
{
  using Dims = std::vector<hsize_t>;

  class Dataspace
  {
  public:
    Dataspace() = default;
    explicit Dataspace(hid_t id) : h_(id) {}
    static Dataspace take(hid_t id) { return Dataspace(Handle::take(id)); }

    hid_t id() const noexcept { return h_.get(); }
    bool valid() const noexcept { return h_.valid(); }

    /// Number of elements (product of all extents). For a scalar space this is 1.
    hssize_t size() const
    {
      hssize_t n = H5Sget_simple_extent_npoints(h_.get());
      throw_if_negative(static_cast<herr_t>(n < 0 ? -1 : 0), "H5Sget_simple_extent_npoints failed");
      return n;
    }

    /// Current extents.
    Dims extents() const
    {
      const int rank = H5Sget_simple_extent_ndims(h_.get());
      throw_if_negative(static_cast<herr_t>(rank < 0 ? -1 : 0), "H5Sget_simple_extent_ndims failed");
      Dims out(static_cast<std::size_t>(rank));
      throw_if_negative(H5Sget_simple_extent_dims(h_.get(), out.data(), nullptr),
                        "H5Sget_simple_extent_dims failed");
      return out;
    }

    /// Restrict the selection to a hyperslab `[offset, offset+count)` along each dim.
    /// Stride and block default to 1 (the only pattern DiFfRG uses).
    void select_hyperslab(const Dims &offset, const Dims &count)
    {
      throw_if_negative(
          H5Sselect_hyperslab(h_.get(), H5S_SELECT_SET, offset.data(), nullptr, count.data(), nullptr),
          "H5Sselect_hyperslab failed");
    }

    static Dataspace scalar()
    {
      hid_t s = H5Screate(H5S_SCALAR);
      throw_if_negative(s, "H5Screate(H5S_SCALAR) failed");
      return Dataspace::take(s);
    }

    static Dataspace simple(const Dims &dims)
    {
      hid_t s = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
      throw_if_negative(s, "H5Screate_simple failed");
      return Dataspace::take(s);
    }

    /// Simple dataspace with the given current extents and unlimited maximum
    /// extents along every axis.
    static Dataspace simple_unlimited(const Dims &dims)
    {
      Dims maxdims(dims.size(), H5S_UNLIMITED);
      hid_t s = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), maxdims.data());
      throw_if_negative(s, "H5Screate_simple (unlimited) failed");
      return Dataspace::take(s);
    }

  private:
    explicit Dataspace(Handle h) : h_(std::move(h)) {}
    Handle h_;
  };
} // namespace DiFfRG::hdf5
