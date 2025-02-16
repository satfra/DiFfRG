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

  const std::vector<double> &QuadratureProvider::get_matsubara_points_d(const double T, const double typical_E)
  {
    const auto size = MatsubaraQuadrature<double>::predict_size(T, typical_E);
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (matsubara_quadrature_d.find(size) == matsubara_quadrature_d.end()) compute_matsubara_quadrature_d(T, typical_E);
    return matsubara_quadrature_d[size].nodes();
  }
  const std::vector<double> &QuadratureProvider::get_matsubara_weights_d(const double T, const double typical_E)
  {
    const auto size = MatsubaraQuadrature<double>::predict_size(T, typical_E);
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (matsubara_quadrature_d.find(size) == matsubara_quadrature_d.end()) compute_matsubara_quadrature_d(T, typical_E);
    return matsubara_quadrature_d[size].weights();
  }

  const std::vector<float> &QuadratureProvider::get_matsubara_points_f(const float T, const float typical_E)
  {
    const auto size = MatsubaraQuadrature<float>::predict_size(T, typical_E);
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (matsubara_quadrature_f.find(size) == matsubara_quadrature_f.end()) compute_matsubara_quadrature_f(T, typical_E);
    return matsubara_quadrature_f[size].nodes();
  }
  const std::vector<float> &QuadratureProvider::get_matsubara_weights_f(const float T, const float typical_E)
  {
    const auto size = MatsubaraQuadrature<float>::predict_size(T, typical_E);
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (matsubara_quadrature_f.find(size) == matsubara_quadrature_f.end()) compute_matsubara_quadrature_f(T, typical_E);
    return matsubara_quadrature_f[size].weights();
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

  const double *QuadratureProvider::get_device_matsubara_points_d(const double T, const double typical_E,
                                                                  const int device)
  {
    const auto size = MatsubaraQuadrature<double>::predict_size(T, typical_E);
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (matsubara_quadrature_d.find(size) == matsubara_quadrature_d.end()) compute_matsubara_quadrature_d(T, typical_E);
    return matsubara_quadrature_d[size].device_nodes();
  }
  const double *QuadratureProvider::get_device_matsubara_weights_d(const double T, const double typical_E,
                                                                   const int device)
  {
    const auto size = MatsubaraQuadrature<double>::predict_size(T, typical_E);
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (matsubara_quadrature_d.find(size) == matsubara_quadrature_d.end()) compute_matsubara_quadrature_d(T, typical_E);
    return matsubara_quadrature_d[size].device_weights();
  }

  const float *QuadratureProvider::get_device_matsubara_points_f(const float T, const float typical_E, const int device)
  {
    const auto size = MatsubaraQuadrature<float>::predict_size(T, typical_E);
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (matsubara_quadrature_f.find(size) == matsubara_quadrature_f.end()) compute_matsubara_quadrature_f(T, typical_E);
    return matsubara_quadrature_f[size].device_nodes();
  }
  const float *QuadratureProvider::get_device_matsubara_weights_f(const float T, const float typical_E,
                                                                  const int device)
  {
    const auto size = MatsubaraQuadrature<float>::predict_size(T, typical_E);
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (matsubara_quadrature_f.find(size) == matsubara_quadrature_f.end()) compute_matsubara_quadrature_f(T, typical_E);
    return matsubara_quadrature_f[size].device_weights();
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

  void QuadratureProvider::compute_matsubara_quadrature_d(const double T, const double typical_E)
  {
    const auto size = MatsubaraQuadrature<double>::predict_size(T, typical_E);
    auto [it, success] = matsubara_quadrature_d.insert(std::make_pair(size, MatsubaraQuadrature<double>()));
    if (!success) throw std::runtime_error("Failed to insert quadrature into map");
    it->second.reinit(T, typical_E);
  }

  void QuadratureProvider::compute_matsubara_quadrature_f(const float T, const float typical_E)
  {
    const auto size = MatsubaraQuadrature<float>::predict_size(T, typical_E);
    auto [it, success] = matsubara_quadrature_f.insert(std::make_pair(size, MatsubaraQuadrature<float>()));
    if (!success) throw std::runtime_error("Failed to insert quadrature into map");
    it->second.reinit(T, typical_E);
  }
} // namespace DiFfRG