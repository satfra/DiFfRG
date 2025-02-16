// DiFfRG
#include <DiFfRG/common/quadrature/quadrature_provider.hh>

namespace DiFfRG
{
  QuadratureProvider::QuadratureProvider() {}
  QuadratureProvider::~QuadratureProvider() = default;

  const std::vector<double> &QuadratureProvider::get_points_d(const uint quadrature_size, const QuadratureType _t)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (quadrature_d[(size_t)_t].find(quadrature_size) == quadrature_d[(size_t)_t].end())
      compute_quadrature_d(quadrature_size, _t);
    return quadrature_d[(size_t)_t][quadrature_size].nodes();
  }
  const std::vector<double> &QuadratureProvider::get_weights_d(const uint quadrature_size, const QuadratureType _t)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (quadrature_d[(size_t)_t].find(quadrature_size) == quadrature_d[(size_t)_t].end())
      compute_quadrature_d(quadrature_size, _t);
    return quadrature_d[(size_t)_t][quadrature_size].weights();
  }

  const std::vector<float> &QuadratureProvider::get_points_f(const uint quadrature_size, const QuadratureType _t)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (quadrature_f[(size_t)_t].find(quadrature_size) == quadrature_f[(size_t)_t].end())
      compute_quadrature_f(quadrature_size, _t);
    return quadrature_f[(size_t)_t][quadrature_size].nodes();
  }
  const std::vector<float> &QuadratureProvider::get_weights_f(const uint quadrature_size, const QuadratureType _t)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (quadrature_f[(size_t)_t].find(quadrature_size) == quadrature_f[(size_t)_t].end())
      compute_quadrature_f(quadrature_size, _t);
    return quadrature_f[(size_t)_t][quadrature_size].weights();
  }

#ifdef __CUDACC__
  const double *QuadratureProvider::get_device_points_d(const uint quadrature_size, const int device,
                                                        const QuadratureType _t)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (quadrature_d[(size_t)_t].find(quadrature_size) == quadrature_d[(size_t)_t].end())
      compute_quadrature_d(quadrature_size, _t);
    return quadrature_d[(size_t)_t][quadrature_size].device_nodes();
  }
  const double *QuadratureProvider::get_device_weights_d(const uint quadrature_size, const int device,
                                                         const QuadratureType _t)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (quadrature_d[(size_t)_t].find(quadrature_size) == quadrature_d[(size_t)_t].end())
      compute_quadrature_d(quadrature_size, _t);
    return quadrature_d[(size_t)_t][quadrature_size].device_weights();
  }

  const float *QuadratureProvider::get_device_points_f(const uint quadrature_size, const int device,
                                                       const QuadratureType _t)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (quadrature_f[(size_t)_t].find(quadrature_size) == quadrature_f[(size_t)_t].end())
      compute_quadrature_f(quadrature_size, _t);
    return quadrature_f[(size_t)_t][quadrature_size].device_nodes();
  }
  const float *QuadratureProvider::get_device_weights_f(const uint quadrature_size, const int device,
                                                        const QuadratureType _t)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (quadrature_f[(size_t)_t].find(quadrature_size) == quadrature_f[(size_t)_t].end())
      compute_quadrature_f(quadrature_size, _t);
    return quadrature_f[(size_t)_t][quadrature_size].device_weights();
  }
#endif

  void QuadratureProvider::compute_quadrature_d(const uint quadrature_size, const QuadratureType _t)
  {
    auto [it, success] = quadrature_d[(size_t)_t].insert(std::make_pair(quadrature_size, Quadrature<double>()));
    if (!success) throw std::runtime_error("Failed to insert quadrature into map");
    it->second.reinit(quadrature_size, _t);
  }

  void QuadratureProvider::compute_quadrature_f(const uint quadrature_size, const QuadratureType _t)
  {
    auto [it, success] = quadrature_f[(size_t)_t].insert(std::make_pair(quadrature_size, Quadrature<float>()));
    if (!success) throw std::runtime_error("Failed to insert quadrature into map");
    it->second.reinit(quadrature_size, _t);
  }
} // namespace DiFfRG