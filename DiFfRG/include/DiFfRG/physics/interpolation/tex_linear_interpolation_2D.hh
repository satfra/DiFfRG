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
   * @brief A linear interpolator for 2D data, using texture memory on the GPU and floating point arithmetic on the CPU.
   *
   * @tparam NT input data type
   * @tparam Coordinates coordinate system of the input data
   */
  template <typename NT, typename Coordinates> class TexLinearInterpolator2D
  {
    static_assert(Coordinates::dim == 2, "TexLinearInterpolator2D requires 2D coordinates");

  public:
    /**
     * @brief Construct a TexLinearInterpolator2D object from a vector of data and a coordinate system.
     *
     * @param data vector containing the data
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator2D(const std::vector<NT> &data, const Coordinates &coordinates);
    /**
     * @brief Construct a TexLinearInterpolator2D object from a pointer to data and a coordinate system.
     *
     * @param data pointer to the data
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator2D(const NT *data, const Coordinates &coordinates);
    /**
     * @brief Construct a TexLinearInterpolator2D with internal, zeroed data and a coordinate system.
     *
     * @param coordinates coordinate system of the data
     */
    TexLinearInterpolator2D(const Coordinates &coordinates);
    /**
     * @brief Construct a copy of a TexLinearInterpolator2D object.
     *
     * @param other the object to copy
     */
    TexLinearInterpolator2D(const TexLinearInterpolator2D &other);

    ~TexLinearInterpolator2D();

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
     * @param x x-coordinate of the point
     * @param y y-coordinate of the point
     * @return ReturnType the interpolated value
     */
    __device__ __host__ ReturnType operator()(const float x, const float y) const
    {
      auto [idx_x, idx_y] = coordinates.backward(x, y);
#ifdef __CUDA_ARCH__
      if constexpr (std::is_same_v<ReturnType, autodiff::real>)
        return std::array<double, 2>{tex2D<float>(texture, idx_y + 0.5f, idx_x + 0.5f),
                                     tex2D<float>(texture_AD, idx_y + 0.5f, idx_x + 0.5f)};
      else if constexpr (std::is_same_v<ReturnType, float>)
        return tex2D<float>(texture, idx_y + 0.5f, idx_x + 0.5f);
#else
      using std::ceil;
      using std::floor;
      using std::max;
      using std::min;

      idx_x = max(static_cast<decltype(idx_x)>(0), min(idx_x, static_cast<decltype(idx_x)>(shape[0] - 1)));
      idx_y = max(static_cast<decltype(idx_y)>(0), min(idx_y, static_cast<decltype(idx_y)>(shape[1] - 1)));

      uint x1 = min(ceil(idx_x + static_cast<decltype(idx_x)>(1e-16)), static_cast<decltype(idx_x)>(shape[0] - 1));
      const auto x0 = x1 - 1;
      uint y1 = min(ceil(idx_y + static_cast<decltype(idx_y)>(1e-16)), static_cast<decltype(idx_y)>(shape[1] - 1));
      const auto y0 = y1 - 1;

      const auto corner00 = m_data[x0 * shape[1] + y0];
      const auto corner01 = m_data[x0 * shape[1] + y1];
      const auto corner10 = m_data[x1 * shape[1] + y0];
      const auto corner11 = m_data[x1 * shape[1] + y1];

      const auto ret = corner00 * (x1 - idx_x) * (y1 - idx_y) + corner01 * (x1 - idx_x) * (idx_y - y0) +
                       corner10 * (idx_x - x0) * (y1 - idx_y) + corner11 * (idx_x - x0) * (idx_y - y0);

      if constexpr (std::is_same_v<ReturnType, autodiff::real>) {
        const auto corner00_AD = m_data_AD[x0 * shape[1] + y0];
        const auto corner01_AD = m_data_AD[x0 * shape[1] + y1];
        const auto corner10_AD = m_data_AD[x1 * shape[1] + y0];
        const auto corner11_AD = m_data_AD[x1 * shape[1] + y1];

        const auto ret_AD = corner00_AD * (x1 - idx_x) * (y1 - idx_y) + corner01_AD * (x1 - idx_x) * (idx_y - y0) +
                            corner10_AD * (idx_x - x0) * (y1 - idx_y) + corner11_AD * (idx_x - x0) * (idx_y - y0);

        return std::array<double, 2>{{ret, ret_AD}};
      } else if constexpr (std::is_same_v<ReturnType, float>)
        return ret;
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
    const std::array<uint, 2> shape;
    const Coordinates coordinates;

    std::shared_ptr<float[]> m_data;
    cudaArray_t device_array;
    cudaTextureObject_t texture;

    std::shared_ptr<float[]> m_data_AD;
    cudaArray_t device_array_AD;
    cudaTextureObject_t texture_AD;

    const bool owner;
  };
} // namespace DiFfRG