// DiFfRG
#include <DiFfRG/common/quadrature/quadrature.hh>

// external libraries
#include <gsl/gsl_integration.h>

namespace DiFfRG
{
  template <typename NT> Quadrature<NT>::Quadrature(const size_t order, const QuadratureType _t) { reinit(order, _t); }
  template <typename NT> Quadrature<NT>::Quadrature() {}

  template <typename NT> void Quadrature<NT>::reinit(const size_t order, const QuadratureType _t)
  {
    this->order = order;
    this->_t = _t;

#ifdef USE_CUDA
    m_device_nodes.clear();
    m_device_weights.clear();
#endif

    gsl_integration_fixed_workspace *w;
    const gsl_integration_fixed_type *T;

    // create gsl workspace and set quadrature type
    switch (_t) {
    case QuadratureType::legendre:
      T = gsl_integration_fixed_legendre;
      break;
    case QuadratureType::chebyshev:
      T = gsl_integration_fixed_chebyshev;
      break;
    case QuadratureType::hermite:
      T = gsl_integration_fixed_hermite;
      break;
    case QuadratureType::jacobi:
      T = gsl_integration_fixed_jacobi;
      break;
    case QuadratureType::laguerre:
      T = gsl_integration_fixed_laguerre;
      break;
    default:
      throw std::runtime_error("Unknown quadrature type");
    }
    w = gsl_integration_fixed_alloc(T, order, 0.0, 1.0, 0.0, 0.0);

    // get nodes and weights
    const double *gsl_nodes = gsl_integration_fixed_nodes(w);
    const double *gsl_weights = gsl_integration_fixed_weights(w);

    // copy nodes and weights to std::vector
    m_nodes.resize(order);
    m_weights.resize(order);
    for (size_t i = 0; i < order; ++i) {
      m_nodes[i] = gsl_nodes[i];
      m_weights[i] = gsl_weights[i];
    }

    // free gsl workspace
    gsl_integration_fixed_free(w);
  }

  template <typename NT> const std::vector<NT> &Quadrature<NT>::nodes() const { return m_nodes; }
  template <typename NT> const std::vector<NT> &Quadrature<NT>::weights() const { return m_weights; }

  template <typename NT> size_t Quadrature<NT>::size() const { return order; }
  template <typename NT> QuadratureType Quadrature<NT>::get_type() const { return _t; }

#ifdef USE_CUDA
  template <typename NT> const NT *Quadrature<NT>::device_nodes()
  {
    move_device_data();
    return thrust::raw_pointer_cast(m_device_nodes.data());
  }
  template <typename NT> const NT *Quadrature<NT>::device_weights()
  {
    move_device_data();
    return thrust::raw_pointer_cast(m_device_weights.data());
  }

  template <typename NT> void Quadrature<NT>::move_device_data()
  {
    if (m_device_nodes.size() == 0) {
      m_device_nodes.resize(order);
      m_device_weights.resize(order);
      thrust::copy(m_nodes.begin(), m_nodes.end(), m_device_nodes.begin());
      thrust::copy(m_weights.begin(), m_weights.end(), m_device_weights.begin());
    }
  }
#endif

  // explicit instantiation
  template class Quadrature<double>;
  template class Quadrature<float>;
} // namespace DiFfRG