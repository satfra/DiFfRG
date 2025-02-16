#pragma once

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>

// standard library
#include <vector>

namespace DiFfRG
{
  /**
   * @brief A class that provides quadrature points and weights, in host and device memory.
   * The quadrature points and weights are computed using deal.II's QGauss class.
   * This avoids recomputing the quadrature points and weights for each integrator.
   */
  class QuadratureProvider
  {
  public:
    QuadratureProvider();
    ~QuadratureProvider();

    /**
     * @brief Get the quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double>
    const std::vector<NT> &get_points(const uint quadrature_size, const QuadratureType _t = QuadratureType::legendre)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_points_d(quadrature_size, _t);
      else if constexpr (std::is_same_v<NT, float>)
        return get_points_f(quadrature_size, _t);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_points");
    }
    const std::vector<double> &get_points_d(const uint quadrature_size, const QuadratureType _t);
    const std::vector<float> &get_points_f(const uint quadrature_size, const QuadratureType _t);

    /**
     * @brief Get the quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double>
    const std::vector<NT> &get_weights(const uint quadrature_size, const QuadratureType _t = QuadratureType::legendre)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_weights_d(quadrature_size, _t);
      else if constexpr (std::is_same_v<NT, float>)
        return get_weights_f(quadrature_size, _t);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_weights");
    }
    const std::vector<double> &get_weights_d(const uint quadrature_size, const QuadratureType _t);
    const std::vector<float> &get_weights_f(const uint quadrature_size, const QuadratureType _t);

#ifdef USE_CUDA
    /**
     * @brief Get the device-side quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double>
    const NT *get_device_points(const uint quadrature_size, const int device = 0,
                                const QuadratureType _t = QuadratureType::legendre)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_device_points_d(quadrature_size, device, _t);
      else if constexpr (std::is_same_v<NT, float>)
        return get_device_points_f(quadrature_size, device, _t);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_device_points");
    }
    const double *get_device_points_d(const uint quadrature_size, const int device, const QuadratureType _t);
    const float *get_device_points_f(const uint quadrature_size, const int device, const QuadratureType _t);

    /**
     * @brief Get the device-side quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double>
    const NT *get_device_weights(const uint quadrature_size, const int device = 0,
                                 const QuadratureType _t = QuadratureType::legendre)

    {
      if constexpr (std::is_same_v<NT, double>)
        return get_device_weights_d(quadrature_size, device, _t);
      else if constexpr (std::is_same_v<NT, float>)
        return get_device_weights_f(quadrature_size, device, _t);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_device_weights");
    }
    const double *get_device_weights_d(const uint quadrature_size, const int device, const QuadratureType _t);
    const float *get_device_weights_f(const uint quadrature_size, const int device, const QuadratureType _t);
#endif

  private:
    /**
     * @brief Compute the quadrature points and weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size
     */
    template <typename NT = double>
    void compute_quadrature(const uint quadrature_size, const QuadratureType _t = QuadratureType::legendre)
    {
      if constexpr (std::is_same_v<NT, double>)
        compute_quadrature_d(quadrature_size, _t);
      else if constexpr (std::is_same_v<NT, float>)
        compute_quadrature_f(quadrature_size, _t);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::compute_quadrature");
    }
    void compute_quadrature_d(const uint quadrature_size, const QuadratureType _t);
    void compute_quadrature_f(const uint quadrature_size, const QuadratureType _t);

    std::array<std::map<uint, Quadrature<double>>, static_cast<std::size_t>(QuadratureType::count)> quadrature_d;
    std::array<std::map<uint, Quadrature<float>>, static_cast<std::size_t>(QuadratureType::count)> quadrature_f;
  };
} // namespace DiFfRG