#pragma once

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/physics/interpolation/common.hh>

// standard library
#include <memory>

// external libraries
#include <autodiff/forward/real.hpp>

namespace DiFfRG
{
  /**
   * @brief A linear interpolator for 1D data, using texture memory on the GPU and floating point arithmetic on the CPU.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class TexLinearInterpolator1D
  {
    static_assert(Coordinates::dim == 1, "TexLinearInterpolator1D requires 1D coordinates");

    static constexpr int max_device_count = 4;

  public:
    /**
     * @brief Construct a TexLinearInterpolator1D object from a vector of data and a coordinate system.
     *
     * @param data vector containing the data
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator1D(const std::vector<NT> &data, const Coordinates &coordinates);
    /**
     * @brief Construct a TexLinearInterpolator1D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator1D(const NT *data, const Coordinates &coordinates);
    /**
     * @brief Construct a TexLinearInterpolator1D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator1D(const Coordinates &coordinates);
    /**
     * @brief Construct a copy of a TexLinearInterpolator1D object.
     *
     * @param other the object to copy
     */
    TexLinearInterpolator1D(const TexLinearInterpolator1D &other);

    ~TexLinearInterpolator1D();

    template <typename NT2> void update(const NT2 *data)
    {
      if (!owner) throw std::runtime_error("Cannot update data of non-owner interpolator");

      if constexpr (std::is_same_v<NT2, autodiff::real>)
        for (uint i = 0; i < size; ++i) {
          m_data[i] = static_cast<float>(val(data[i]));
          m_data_AD[i] = static_cast<float>(derivative(data[i]));
        }
      else
        for (uint i = 0; i < size; ++i)
          m_data[i] = static_cast<float>(data[i]);

      update();
    }
    void update();

    float *data() const;
    float *data_AD() const;

    using ReturnType = typename internal::__TLITypes<NT>::ReturnType;

    /**
     * @brief Interpolate the data at a given point.
     *
     * @param x the point at which to interpolate
     * @return ReturnType the interpolated value
     */
    __device__ __host__ ReturnType operator()(const float x) const
    {
#ifdef __CUDA_ARCH__
      if constexpr (std::is_same_v<ReturnType, autodiff::real>)
        return std::array<double, 2>{tex1D<float>(texture[0], coordinates.backward(x) + 0.5f),
                                     tex1D<float>(texture_AD[0], coordinates.backward(x) + 0.5f)};
      else if constexpr (std::is_same_v<ReturnType, float>)
        return tex1D<float>(texture[0], coordinates.backward(x) + 0.5f);
#else
      float idx = coordinates.backward(x);
      idx = std::max(0.f, std::min(idx, static_cast<float>(size - 1)));
      if constexpr (std::is_same_v<ReturnType, autodiff::real>)
        return std::array<double, 2>{{m_data[uint(std::floor(idx))] * (1.f - idx + std::floor(idx)) +
                                          m_data[uint(std::ceil(idx))] * (idx - std::floor(idx)),
                                      m_data_AD[uint(std::floor(idx))] * (1.f - idx + std::floor(idx)) +
                                          m_data_AD[uint(std::ceil(idx))] * (idx - std::floor(idx))}};
      else if constexpr (std::is_same_v<ReturnType, float>)
        return m_data[uint(std::floor(idx))] * (1.f - idx + std::floor(idx)) +
               m_data[uint(std::ceil(idx))] * (idx - std::floor(idx));
#endif
    }

    ReturnType &operator[](const uint i);
    const ReturnType &operator[](const uint i) const;

    /**
     * @brief Get the coordinate system of the data.
     *
     * @return const Coordinates& the coordinate system
     */
    const Coordinates &get_coordinates() const { return coordinates; }

  private:
    const uint size;
    const Coordinates coordinates;

    std::shared_ptr<float[]> m_data;
    std::vector<cudaArray_t> device_array;
    cudaTextureObject_t texture[max_device_count];

    std::shared_ptr<float[]> m_data_AD;
    std::vector<cudaArray_t> device_array_AD;
    cudaTextureObject_t texture_AD[max_device_count];

    const bool owner;
    int n_devices;
  };
} // namespace DiFfRG