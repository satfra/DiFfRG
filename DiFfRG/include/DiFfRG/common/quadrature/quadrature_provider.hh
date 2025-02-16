#pragma once

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
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

    /**
     * @brief Get the quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double> const std::vector<NT> &get_matsubara_points(const NT T, const NT typical_E)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_matsubara_points_d(T, typical_E);
      else if constexpr (std::is_same_v<NT, float>)
        return get_matsubara_points_f(T, typical_E);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_matsubara_points");
    }
    const std::vector<double> &get_matsubara_points_d(const double T, const double typical_E);
    const std::vector<float> &get_matsubara_points_f(const float T, const float typical_E);

    /**
     * @brief Get the quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double> const std::vector<NT> &get_matsubara_weights(const NT T, const NT typical_E)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_matsubara_weights_d(T, typical_E);
      else if constexpr (std::is_same_v<NT, float>)
        return get_matsubara_weights_f(T, typical_E);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_matsubara_weights");
    }
    const std::vector<double> &get_matsubara_weights_d(const double T, const double typical_E);
    const std::vector<float> &get_matsubara_weights_f(const float T, const float typical_E);

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

    /**
     * @brief Get the device-side quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double>
    const NT *get_device_matsubara_points(const NT T, const NT typical_E, const int device = 0)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_device_matsubara_points_d(T, typical_E, device);
      else if constexpr (std::is_same_v<NT, float>)
        return get_device_matsubara_points_f(T, typical_E, device);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_device_matsubara_points");
    }
    const double *get_device_matsubara_points_d(const double T, const double typical_E, const int device);
    const float *get_device_matsubara_points_f(const float T, const float typical_E, const int device);

    /**
     * @brief Get the device-side quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double>
    const NT *get_device_matsubara_weights(const NT T, const NT typical_E, const int device = 0)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_device_matsubara_weights_d(T, typical_E, device);
      else if constexpr (std::is_same_v<NT, float>)
        return get_device_matsubara_weights_f(T, typical_E, device);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_device_matsubara_weights");
    }
    const double *get_device_matsubara_weights_d(const double T, const double typical_E, const int device);
    const float *get_device_matsubara_weights_f(const float T, const float typical_E, const int device);
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

    template <typename NT = double> void compute_matsubara_quadrature(const NT T, const NT typical_E)
    {
      if constexpr (std::is_same_v<NT, double>)
        compute_matsubara_quadrature_d(T, typical_E);
      else if constexpr (std::is_same_v<NT, float>)
        compute_matsubara_quadrature_f(T, typical_E);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::compute_matsubara_quadrature");
    }
    void compute_matsubara_quadrature_d(const double T, const double typical_E);
    void compute_matsubara_quadrature_f(const float T, const float typical_E);

    std::array<std::map<uint, Quadrature<double>>, static_cast<std::size_t>(QuadratureType::count)> quadrature_d;
    std::array<std::map<uint, Quadrature<float>>, static_cast<std::size_t>(QuadratureType::count)> quadrature_f;

    std::map<uint, MatsubaraQuadrature<double>> matsubara_quadrature_d;
    std::map<uint, MatsubaraQuadrature<float>> matsubara_quadrature_f;
  };
} // namespace DiFfRG