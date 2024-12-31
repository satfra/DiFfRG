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
   * @brief A linear interpolator for 1D data, both on GPU and CPU
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class LinearInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "LinearInterpolator1D requires 1D coordinates");

  public:
    /**
     * @brief Construct a LinearInterpolator1D object from a vector of data and a coordinate system.
     *
     * @param data vector containing the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator1D(const std::vector<NT> &data, const Coordinates &coordinates);
    /**
     * @brief Construct a LinearInterpolator1D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param size size of the data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator1D(const NT *data, const Coordinates &coordinates);
    /**
     * @brief Construct a LinearInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param size size of the internal data
     * @param coordinates coordinate system of the data
     */
    LinearInterpolator1D(const Coordinates &coordinates);
    /**
     * @brief Construct a copy of a LinearInterpolator1D object.
     *
     * @param other the object to copy
     */
    LinearInterpolator1D(const LinearInterpolator1D &other);

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
    __device__ __host__ NT operator()(const typename Coordinates::ctype x) const
    {
#ifndef __CUDA_ARCH__
      using std::ceil;
      using std::floor;
      using std::max;
      using std::min;
#endif

      auto idx = coordinates.backward(x);
      idx = max(static_cast<decltype(idx)>(0), min(idx, static_cast<decltype(idx)>(size - 1)));
#ifndef __CUDA_ARCH__
      return m_data[uint(floor(idx))] * (1. - idx + floor(idx)) + m_data[uint(ceil(idx))] * (idx - floor(idx));
#else
      return device_data_ptr[uint(floor(idx))] * (1. - idx + floor(idx)) +
             device_data_ptr[uint(ceil(idx))] * (idx - floor(idx));
#endif
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

    std::shared_ptr<NT[]> m_data;
    std::shared_ptr<thrust::device_vector<NT>> device_data;
    const NT *device_data_ptr;

    const bool owner;
  };
} // namespace DiFfRG