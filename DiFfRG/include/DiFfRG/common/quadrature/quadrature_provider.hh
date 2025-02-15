#pragma once

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>

// standard library
#include <map>
#include <string>
#include <vector>

// external libraries
#include <gsl/gsl_integration.h>

namespace DiFfRG
{
  struct Quadrature {
    Quadrature(const uint order, const std::string type = "legendre");
    ~Quadrature();

    const double *get_nodes() const;
    const double *get_weights() const;

    uint get_order() const;
    const std::string &get_type() const;

  private:
    const std::string type;
    const uint order;

    gsl_integration_fixed_workspace *w;
    const gsl_integration_fixed_type *T;

    double *nodes;
    double *weights;
  };

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
    template <typename NT = double> const std::vector<NT> &get_points(const uint quadrature_size)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_points_d(quadrature_size);
      else if constexpr (std::is_same_v<NT, float>)
        return get_points_f(quadrature_size);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_points");
    }
    const std::vector<double> &get_points_d(const uint quadrature_size);
    const std::vector<float> &get_points_f(const uint quadrature_size);

    /**
     * @brief Get the quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const std::vector<double>&
     */
    template <typename NT = double> const std::vector<NT> &get_weights(const uint quadrature_size)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_weights_d(quadrature_size);
      else if constexpr (std::is_same_v<NT, float>)
        return get_weights_f(quadrature_size);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_weights");
    }
    const std::vector<double> &get_weights_d(const uint quadrature_size);
    const std::vector<float> &get_weights_f(const uint quadrature_size);

#ifdef USE_CUDA
    /**
     * @brief Get the device-side quadrature points for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double> const NT *get_device_points(const uint quadrature_size, const int device = 0)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_device_points_d(quadrature_size, device);
      else if constexpr (std::is_same_v<NT, float>)
        return get_device_points_f(quadrature_size, device);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_device_points");
    }
    const double *get_device_points_d(const uint quadrature_size, const int device = 0);
    const float *get_device_points_f(const uint quadrature_size, const int device = 0);

    /**
     * @brief Get the device-side quadrature weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size Size of the quadrature.
     * @return const double*
     */
    template <typename NT = double> const NT *get_device_weights(const uint quadrature_size, const int device = 0)
    {
      if constexpr (std::is_same_v<NT, double>)
        return get_device_weights_d(quadrature_size, device);
      else if constexpr (std::is_same_v<NT, float>)
        return get_device_weights_f(quadrature_size, device);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::get_device_weights");
    }
    const double *get_device_weights_d(const uint quadrature_size, const int device = 0);
    const float *get_device_weights_f(const uint quadrature_size, const int device = 0);
#endif

  private:
    /**
     * @brief Compute the quadrature points and weights for a quadrature of size quadrature_size.
     *
     * @param quadrature_size
     */
    template <typename NT = double> void compute_quadrature(const uint quadrature_size)
    {
      if constexpr (std::is_same_v<NT, double>)
        compute_quadrature_d(quadrature_size);
      else if constexpr (std::is_same_v<NT, float>)
        compute_quadrature_f(quadrature_size);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::compute_quadrature");
    }
    void compute_quadrature_d(const uint quadrature_size);
    void compute_quadrature_f(const uint quadrature_size);

    std::map<uint, std::vector<double>> points_d;
    std::map<uint, std::vector<float>> points_f;
    std::map<uint, std::vector<double>> weights_d;
    std::map<uint, std::vector<float>> weights_f;

#ifdef USE_CUDA
    /**
     * @brief If necessary, compute the quadrature points and weights for a quadrature of size quadrature_size on the
     * host and then copy them to the device.
     *
     * @param quadrature_size
     */
    template <typename NT = double> void compute_device_quadrature(const uint quadrature_size, const int device = 0)
    {
      if constexpr (std::is_same_v<NT, double>)
        compute_device_quadrature_d(quadrature_size, device);
      else if constexpr (std::is_same_v<NT, float>)
        compute_device_quadrature_f(quadrature_size, device);
      static_assert(std::is_same_v<NT, double> || std::is_same_v<NT, float>,
                    "Unknown type requested of QuadratureProvider::compute_device_quadrature");
    }
    void compute_device_quadrature_d(const uint quadrature_size, const int device = 0);
    void compute_device_quadrature_f(const uint quadrature_size, const int device = 0);

    std::vector<std::map<uint, thrust::device_vector<double>>> device_points_d;
    std::vector<std::map<uint, thrust::device_vector<float>>> device_points_f;
    std::vector<std::map<uint, thrust::device_vector<double>>> device_weights_d;
    std::vector<std::map<uint, thrust::device_vector<float>>> device_weights_f;
#endif
  };
} // namespace DiFfRG