// DiFfRG
#include <DiFfRG/physics/integration/quadrature_provider.hh>

namespace DiFfRG
{
  Quadrature::Quadrature(const uint order, const std::string type) : order(order), type(type)
  {
    if (type == "legendre")
      T = gsl_integration_fixed_legendre;
    else
      throw std::runtime_error("Unknown quadrature type: " + type);
    w = gsl_integration_fixed_alloc(T, order, 0.0, 1.0, 0.0, 0.0);

    nodes = gsl_integration_fixed_nodes(w);
    weights = gsl_integration_fixed_weights(w);
  }

  Quadrature::~Quadrature() { gsl_integration_fixed_free(w); }

  const double *Quadrature::get_nodes() const { return nodes; }
  const double *Quadrature::get_weights() const { return weights; }

  uint Quadrature::get_order() const { return order; }
  const std::string &Quadrature::get_type() const { return type; }

  QuadratureProvider::QuadratureProvider()
  {
#ifdef __CUDACC__
    int n_cuda_devices = 0;
    cudaGetDeviceCount(&n_cuda_devices);
    device_points_d.resize(n_cuda_devices);
    device_weights_d.resize(n_cuda_devices);
    device_points_f.resize(n_cuda_devices);
    device_weights_f.resize(n_cuda_devices);
#endif
  }
  QuadratureProvider::~QuadratureProvider() = default;

  const std::vector<double> &QuadratureProvider::get_points_d(const uint quadrature_size)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (points_d.find(quadrature_size) == points_d.end()) compute_quadrature_d(quadrature_size);
    return points_d[quadrature_size];
  }
  const std::vector<float> &QuadratureProvider::get_weights_f(const uint quadrature_size)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (weights_f.find(quadrature_size) == weights_f.end()) compute_quadrature_f(quadrature_size);
    return weights_f[quadrature_size];
  }

  const std::vector<double> &QuadratureProvider::get_weights_d(const uint quadrature_size)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (weights_d.find(quadrature_size) == weights_d.end()) compute_quadrature_d(quadrature_size);
    return weights_d[quadrature_size];
  }
  const std::vector<float> &QuadratureProvider::get_points_f(const uint quadrature_size)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (points_f.find(quadrature_size) == points_f.end()) compute_quadrature_f(quadrature_size);
    return points_f[quadrature_size];
  }

#ifdef __CUDACC__
  const double *QuadratureProvider::get_device_points_d(const uint quadrature_size, const int device)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (device_points_d[device].find(quadrature_size) == device_points_d[device].end())
      compute_device_quadrature_d(quadrature_size, device);
    return thrust::raw_pointer_cast(device_points_d[device][quadrature_size].data());
  }
  const float *QuadratureProvider::get_device_points_f(const uint quadrature_size, const int device)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (device_points_f[device].find(quadrature_size) == device_points_f[device].end())
      compute_device_quadrature_f(quadrature_size, device);
    return thrust::raw_pointer_cast(device_points_f[device][quadrature_size].data());
  }

  const double *QuadratureProvider::get_device_weights_d(const uint quadrature_size, const int device)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (device_weights_d[device].find(quadrature_size) == device_weights_d[device].end())
      compute_device_quadrature_d(quadrature_size, device);
    return thrust::raw_pointer_cast(device_weights_d[device][quadrature_size].data());
  }
  const float *QuadratureProvider::get_device_weights_f(const uint quadrature_size, const int device)
  {
    // if the quadrature of size quadrature_size is not yet computed, compute it
    if (device_weights_f[device].find(quadrature_size) == device_weights_f[device].end())
      compute_device_quadrature_f(quadrature_size, device);
    return thrust::raw_pointer_cast(device_weights_f[device][quadrature_size].data());
  }
#endif

  void QuadratureProvider::compute_quadrature_d(const uint quadrature_size)
  {
    // compute the quadrature of size quadrature_size
    // and store it in points[quadrature_size] and weights[quadrature_size]

    const auto quadrature = Quadrature(quadrature_size, "legendre");
    const auto *nodes = quadrature.get_nodes();
    const auto *weights = quadrature.get_weights();

    // resize containers
    auto &m_points = this->points_d[quadrature_size];
    auto &m_weights = this->weights_d[quadrature_size];
    m_points.resize(quadrature_size);
    m_weights.resize(quadrature_size);

    // store the points and weights
    for (uint i = 0; i < quadrature_size; ++i) {
      m_points[i] = nodes[i];
      m_weights[i] = weights[i];
    }
  }
  void QuadratureProvider::compute_quadrature_f(const uint quadrature_size)
  {
    // compute the quadrature of size quadrature_size
    // and store it in points[quadrature_size] and weights[quadrature_size]
    const auto quadrature = Quadrature(quadrature_size, "legendre");
    const auto *nodes = quadrature.get_nodes();
    const auto *weights = quadrature.get_weights();

    // resize containers
    auto &m_points = this->points_f[quadrature_size];
    auto &m_weights = this->weights_f[quadrature_size];
    m_points.resize(quadrature_size);
    m_weights.resize(quadrature_size);

    // store the points and weights
    for (uint i = 0; i < quadrature_size; ++i) {
      m_points[i] = nodes[i];
      m_weights[i] = weights[i];
    }
  }

#ifdef __CUDACC__
  void QuadratureProvider::compute_device_quadrature_d(const uint quadrature_size, const int device)
  {
    if (points_d.find(quadrature_size) == points_d.end()) compute_quadrature_d(quadrature_size);

    cudaSetDevice(device);

    // copy the points and weights to the device
    const auto &m_points = this->points_d[quadrature_size];
    const auto &m_weights = this->weights_d[quadrature_size];
    auto &m_device_points = device_points_d[device][quadrature_size];
    auto &m_device_weights = device_weights_d[device][quadrature_size];
    m_device_points.resize(m_points.size());
    m_device_weights.resize(m_weights.size());
    thrust::copy(m_points.begin(), m_points.end(), m_device_points.begin());
    thrust::copy(m_weights.begin(), m_weights.end(), m_device_weights.begin());

    check_cuda();
    cudaDeviceSynchronize();

    cudaSetDevice(0);
  }
  void QuadratureProvider::compute_device_quadrature_f(const uint quadrature_size, const int device)
  {
    if (points_f.find(quadrature_size) == points_f.end()) compute_quadrature_f(quadrature_size);

    cudaSetDevice(device);

    // copy the points and weights to the device
    const auto &m_points = this->points_f[quadrature_size];
    const auto &m_weights = this->weights_f[quadrature_size];
    auto &m_device_points = device_points_f[device][quadrature_size];
    auto &m_device_weights = device_weights_f[device][quadrature_size];
    m_device_points.resize(m_points.size());
    m_device_weights.resize(m_weights.size());
    thrust::copy(m_points.begin(), m_points.end(), m_device_points.begin());
    thrust::copy(m_weights.begin(), m_weights.end(), m_device_weights.begin());

    check_cuda();
    cudaDeviceSynchronize();

    cudaSetDevice(0);
  }
#endif
} // namespace DiFfRG