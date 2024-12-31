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
   * @brief A linear interpolator for 2D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class LinearInterpolator2D
  {
    static_assert(Coordinates::dim == 2, "LinearInterpolator2D requires 2D coordinates");

  public:
    /**
     * @brief Construct a LinearInterpolator2D object from a vector of data and a coordinate system.
     *
     * @param data vector containing the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator2D(const std::vector<NT> &data, const Coordinates &coordinates);
    /**
     * @brief Construct a LinearInterpolator2D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param size size of the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator2D(const NT *data, const Coordinates &coordinates);
    /**
     * @brief Construct a LinearInterpolator2D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator2D(const Coordinates &coordinates);
    /**
     * @brief Construct a copy of a LinearInterpolator2D object.
     *
     * @param other the object to copy
     */
    LinearInterpolator2D(const LinearInterpolator2D &other);

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
    __device__ __host__ NT operator()(const typename Coordinates::ctype x, const typename Coordinates::ctype y) const
    {
#ifndef __CUDA_ARCH__
      using std::ceil;
      using std::floor;
      using std::max;
      using std::min;
#endif

      auto [idx_x, idx_y] = coordinates.backward(x, y);
      idx_x = max(static_cast<decltype(idx_x)>(0), min(idx_x, static_cast<decltype(idx_x)>(shape[0] - 1)));
      idx_y = max(static_cast<decltype(idx_y)>(0), min(idx_y, static_cast<decltype(idx_y)>(shape[1] - 1)));

#ifndef __CUDA_ARCH__
      const auto *d_ptr = m_data.get();
#else
      const auto *d_ptr = device_data_ptr;
#endif

      uint x1 = min(ceil(idx_x + static_cast<decltype(idx_x)>(1e-16)), static_cast<decltype(idx_x)>(shape[0] - 1));
      const auto x0 = x1 - 1;
      uint y1 = min(ceil(idx_y + static_cast<decltype(idx_y)>(1e-16)), static_cast<decltype(idx_y)>(shape[1] - 1));
      const auto y0 = y1 - 1;

      const auto corner00 = d_ptr[x0 * shape[1] + y0];
      const auto corner01 = d_ptr[x0 * shape[1] + y1];
      const auto corner10 = d_ptr[x1 * shape[1] + y0];
      const auto corner11 = d_ptr[x1 * shape[1] + y1];

      return corner00 * (x1 - idx_x) * (y1 - idx_y) + corner01 * (x1 - idx_x) * (idx_y - y0) +
             corner10 * (idx_x - x0) * (y1 - idx_y) + corner11 * (idx_x - x0) * (idx_y - y0);
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
    const std::array<uint, 2> shape;
    const Coordinates coordinates;

    std::shared_ptr<NT[]> m_data;
    std::shared_ptr<thrust::device_vector<NT>> device_data;
    const NT *device_data_ptr;

    const bool owner;
  };
} // namespace DiFfRG