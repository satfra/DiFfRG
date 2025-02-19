// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>

// Standard library
#include <cmath>

namespace DiFfRG
{
  template <typename NT> int MatsubaraQuadrature<NT>::predict_size(const NT T, const NT typical_E, const int step)
  {
    const NT relative_distance = abs(typical_E) / abs(T + 1e-16);
    if (is_close(T, NT(0)) || relative_distance > 1e+2) return -vacuum_quad_size;

    const NT E_max = (precision_factor + 100) * std::abs(typical_E);
    int size = 5 + int(std::sqrt(4 * E_max / (M_PI * M_PI * std::abs(T))));
    size = (int)std::ceil(size / (double)step) * step;
    return size;
  }

  template <typename NT>
  MatsubaraQuadrature<NT>::MatsubaraQuadrature(const NT T, const NT typical_E, const int step, const int min_size,
                                               const int max_size, const int vacuum_quad_size,
                                               const int precision_factor)
  {
    reinit(T, typical_E, step, min_size, max_size, vacuum_quad_size, precision_factor);
  }

  template <typename NT> MatsubaraQuadrature<NT>::MatsubaraQuadrature() : m_size(0), vacuum_quad_size(48) {}

  template <typename NT>
  void MatsubaraQuadrature<NT>::reinit(const NT T, const NT typical_E, const int step, const int min_size,
                                       const int max_size, const int vacuum_quad_size, const int precision_factor)
  {
    if (precision_factor < 0)
      this->precision_factor = 1;
    else
      this->precision_factor = precision_factor;

    if (vacuum_quad_size <= 6)
      this->vacuum_quad_size = 6;
    else
      this->vacuum_quad_size = vacuum_quad_size;

    this->T = T;
    this->typical_E = typical_E;

#ifdef __CUDACC__
    device_x.clear();
    device_w.clear();
#endif

    // Determine the number of nodes in the quadrature rule.
    m_size = predict_size(T, typical_E, step);
    if (m_size < 0) {
      m_size = abs(m_size);
      reinit_0();
      return;
    }
    m_size = std::max(min_size, std::min(max_size, m_size));

    // construct the recurrence relation for the quadrature rule from [1]
    std::vector<NT> a(m_size, 0.);
    std::vector<NT> b(m_size, 0.);

    for (int j = 0; j < m_size; ++j) {
      const double j1 = j + 1;
      a[j] = 2 * powr<2>(M_PI) / (4 * j + 1) / (4 * j + 5);
      b[j] = powr<4>(M_PI) / ((4 * j1 - 1) * (4 * j1 + 3)) / powr<2>(4 * j1 + 1);
    }
    a[0] = powr<2>(M_PI) / 15.;

    const NT mu0 = powr<2>(M_PI) / 6.;

    // compute the nodes and weights of the quadrature rule
    make_quadrature(a, b, mu0, x, w);

    // normalize the weights and scale the nodes
    for (int i = 0; i < m_size; ++i) {
      w[i] = T * w[i] / x[i];
      x[i] = 2. * M_PI * T / std::sqrt(x[i]);
    }
  }

  template <typename NT> const std::vector<NT> &MatsubaraQuadrature<NT>::nodes() const { return x; }
  template <typename NT> const std::vector<NT> &MatsubaraQuadrature<NT>::weights() const { return w; }

  template <typename NT> int MatsubaraQuadrature<NT>::size() const { return m_size; }
  template <typename NT> NT MatsubaraQuadrature<NT>::get_T() const { return T; }
  template <typename NT> NT MatsubaraQuadrature<NT>::get_typical_E() const { return typical_E; }

#ifdef USE_CUDA
  template <typename NT> const NT *MatsubaraQuadrature<NT>::device_nodes()
  {
    move_device_data();
#ifdef __CUDACC__
    return thrust::raw_pointer_cast(device_x.data());
#else
    return device_x.data().get();
#endif
  }

  template <typename NT> const NT *MatsubaraQuadrature<NT>::device_weights()
  {
    move_device_data();
#ifdef __CUDACC__
    return thrust::raw_pointer_cast(device_w.data());
#else
    return device_w.data().get();
#endif
  }
#endif

  template <typename NT> void MatsubaraQuadrature<NT>::move_device_data()
  {
#ifdef __CUDACC__
    if (device_x.size() == 0) {
      device_x.resize(m_size);
      device_w.resize(m_size);
      thrust::copy(x.begin(), x.end(), device_x.begin());
      thrust::copy(w.begin(), w.end(), device_w.begin());
    }
#endif
  }

  template <typename NT> void MatsubaraQuadrature<NT>::reinit_0()
  {
    this->T = 0;
    m_size = abs(m_size);
    if (m_size % 2 != 0) m_size++;
    if (is_close(typical_E, NT(0))) this->typical_E = 1.;

    // obtain a gauss-legendre quadrature rule for the interval [0, 1]
    Quadrature<NT> quad(m_size / 2, QuadratureType::legendre);

    // resize the nodes and weights
    x.resize(m_size);
    w.resize(m_size);

    // strategy: divide into two parts, one with linear and one with logarithmic scaling
    // the dividing point is somewhat above the typical energy scale
    const long double div = 2 * abs(typical_E);

    // the nodes with a linear scale
    for (int i = 0; i < m_size / 2; ++i) {
      x[i] = quad.nodes()[i] * div;
      w[i] = quad.weights()[i] * div / NT(2 * M_PI);
    }

    // the nodes with a logarithmic scale
    using std::log, std::exp, std::abs, std::min;
    const long double extent = 1e6 * abs(typical_E);
    const long double log_start = log(div);
    const long double log_ext = log(extent / div);
    for (int i = 0; i < m_size / 2; ++i) {
      x[i + m_size / 2] = exp(log_start + log_ext * quad.nodes()[i]);
      w[i + m_size / 2] = (quad.weights()[i] * log_ext * x[i + m_size / 2]) / NT(2 * M_PI);
    }
  }

  // explicit instantiation
  template class MatsubaraQuadrature<double>;
  template class MatsubaraQuadrature<float>;
} // namespace DiFfRG