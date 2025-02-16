// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/quadrature/matsubara.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>

namespace DiFfRG
{
  template <typename NT> size_t MatsubaraQuadrature<NT>::predict_size(const NT T, const NT typical_E, const size_t step)
  {
    const NT E_max = 10 * std::abs(typical_E);
    size_t size = 5 + int(std::sqrt(4 * E_max / (M_PI * M_PI * std::abs(T))));
    size = (size_t)std::ceil(size / (double)step) * step;
    return size;
  }

  template <typename NT>
  MatsubaraQuadrature<NT>::MatsubaraQuadrature(const NT T, const NT typical_E, const size_t step, const size_t min_size,
                                               const size_t max_size)
  {
    reinit(T, typical_E, step, min_size, max_size);
  }

  template <typename NT> MatsubaraQuadrature<NT>::MatsubaraQuadrature() {}

  template <typename NT>
  void MatsubaraQuadrature<NT>::reinit(const NT T, const NT typical_E, const size_t step, const size_t min_size,
                                       const size_t max_size)
  {
    this->T = T;
    this->typical_E = typical_E;

#ifdef USE_CUDA
    device_x.clear();
    device_w.clear();
#endif

    // Determine the number of nodes in the quadrature rule.
    m_size = predict_size(T, typical_E, step);
    m_size = std::max(min_size, std::min(max_size, m_size));

    // construct the recurrence relation for the quadrature rule from [1]
    std::vector<NT> a(m_size, 0.);
    std::vector<NT> b(m_size, 0.);

    for (size_t j = 0; j < m_size; ++j) {
      const double j1 = j + 1;
      a[j] = 2 * powr<2>(M_PI) / (4 * j + 1) / (4 * j + 5);
      b[j] = powr<4>(M_PI) / ((4 * j1 - 1) * (4 * j1 + 3)) / powr<2>(4 * j1 + 1);
    }
    a[0] = powr<2>(M_PI) / 15.;

    const NT mu0 = powr<2>(M_PI) / 6.;

    // compute the nodes and weights of the quadrature rule
    make_quadrature(a, b, mu0, x, w);

    // normalize the weights and scale the nodes
    for (size_t i = 0; i < m_size; ++i) {
      w[i] = T * w[i] / x[i];
      x[i] = 2. * M_PI * T / std::sqrt(x[i]);
    }
  }

  template <typename NT> const std::vector<NT> &MatsubaraQuadrature<NT>::nodes() const { return x; }
  template <typename NT> const std::vector<NT> &MatsubaraQuadrature<NT>::weights() const { return w; }

  template <typename NT> size_t MatsubaraQuadrature<NT>::size() const { return m_size; }
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

  // explicit instantiation
  template class MatsubaraQuadrature<double>;
  template class MatsubaraQuadrature<float>;
} // namespace DiFfRG