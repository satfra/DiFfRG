#pragma once

// standard library
#include <memory>

// external libraries
#include <autodiff/forward/real.hpp>

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/physics/interpolation/common.hh>

namespace DiFfRG
{
  /**
   * @brief A linear interpolator for 3D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class LinearInterpolator3D
  {
    static_assert(Coordinates::dim == 3, "LinearInterpolator3D requires 3D coordinates");

  public:
    /**
     * @brief Construct a LinearInterpolator3D object from a vector of data and a coordinate system.
     *
     * @param data vector containing the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const std::vector<NT> &data, const Coordinates &coordinates);
    /**
     * @brief Construct a LinearInterpolator3D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param size size of the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const NT *data, const Coordinates &coordinates);
    /**
     * @brief Construct a LinearInterpolator3D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator3D(const Coordinates &coordinates);
    /**
     * @brief Construct a copy of a LinearInterpolator3D object.
     *
     * @param other the object to copy
     */
    LinearInterpolator3D(const LinearInterpolator3D &other);

    template <typename NT2> void update(const NT2 *data)
    {
      if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

      for (uint i = 0; i < size; ++i)
        m_data[i] = NT(data[i]);

      update();
    }
    void update();

    NT *data() const;

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return NT the interpolated value
     */
    __device__ __host__ NT operator()(const typename Coordinates::ctype x, const typename Coordinates::ctype y,
                                      const typename Coordinates::ctype z) const
    {
#ifndef __CUDA_ARCH__
      using std::ceil;
      using std::floor;
      using std::max;
      using std::min;
#endif

      auto [idx_x, idx_y, idx_z] = coordinates.backward(x, y, z);
      idx_x = max(static_cast<decltype(idx_x)>(0), min(idx_x, static_cast<decltype(idx_x)>(shape[0] - 1)));
      idx_y = max(static_cast<decltype(idx_y)>(0), min(idx_y, static_cast<decltype(idx_y)>(shape[1] - 1)));
      idx_z = max(static_cast<decltype(idx_z)>(0), min(idx_z, static_cast<decltype(idx_z)>(shape[2] - 1)));

#ifndef __CUDA_ARCH__
      const auto *d_ptr = m_data.get();
#else
      const auto *d_ptr = device_data_ptr;
#endif

      uint x1 = min(ceil(idx_x + static_cast<decltype(idx_x)>(1e-16)), static_cast<decltype(idx_x)>(shape[0] - 1));
      const auto x0 = x1 - 1;
      uint y1 = min(ceil(idx_y + static_cast<decltype(idx_y)>(1e-16)), static_cast<decltype(idx_y)>(shape[1] - 1));
      const auto y0 = y1 - 1;
      uint z1 = min(ceil(idx_z + static_cast<decltype(idx_z)>(1e-16)), static_cast<decltype(idx_z)>(shape[2] - 1));
      const auto z0 = z1 - 1;

      const auto corner000 = d_ptr[x0 * shape[1] * shape[2] + y0 * shape[2] + z0];
      const auto corner001 = d_ptr[x0 * shape[1] * shape[2] + y0 * shape[2] + z1];
      const auto corner010 = d_ptr[x0 * shape[1] * shape[2] + y1 * shape[2] + z0];
      const auto corner011 = d_ptr[x0 * shape[1] * shape[2] + y1 * shape[2] + z1];
      const auto corner100 = d_ptr[x1 * shape[1] * shape[2] + y0 * shape[2] + z0];
      const auto corner101 = d_ptr[x1 * shape[1] * shape[2] + y0 * shape[2] + z1];
      const auto corner110 = d_ptr[x1 * shape[1] * shape[2] + y1 * shape[2] + z0];
      const auto corner111 = d_ptr[x1 * shape[1] * shape[2] + y1 * shape[2] + z1];

      return corner000 * (x1 - idx_x) * (y1 - idx_y) * (z1 - idx_z) +
             corner001 * (x1 - idx_x) * (y1 - idx_y) * (idx_z - z0) +
             corner010 * (x1 - idx_x) * (idx_y - y0) * (z1 - idx_z) +
             corner011 * (x1 - idx_x) * (idx_y - y0) * (idx_z - z0) +
             corner100 * (idx_x - x0) * (y1 - idx_y) * (z1 - idx_z) +
             corner101 * (idx_x - x0) * (y1 - idx_y) * (idx_z - z0) +
             corner110 * (idx_x - x0) * (idx_y - y0) * (z1 - idx_z) +
             corner111 * (idx_x - x0) * (idx_y - y0) * (idx_z - z0);
    }

    NT &operator[](const uint i);
    const NT &operator[](const uint i) const;

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    const uint size;
    const Coordinates coordinates;
    const std::array<uint, 3> shape;

    std::shared_ptr<NT[]> m_data;
    std::shared_ptr<thrust::device_vector<NT>> device_data;
    const NT *device_data_ptr;

    const bool owner;
  };
} // namespace DiFfRG
