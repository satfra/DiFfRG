// DiFfRG
#include <DiFfRG/common/quadrature/quadrature.hh>

// external libraries
#include <gsl/gsl_integration.h>

namespace DiFfRG
{
  QuadratureType::QuadratureType(const QuadratureKind kind) : kind(kind)
  {
    if (kind == QuadratureKind::legendre) {
      a = 0.;
      b = 1.;
    } else if (kind == QuadratureKind::chebyshev) {
      a = -1.;
      b = 1.;
    } else if (kind == QuadratureKind::chebyshev2) {
      a = -1.;
      b = 1.;
    } else if (kind == QuadratureKind::hermite) {
      a = 0.;
      b = 1.;
    } else if (kind == QuadratureKind::jacobi) {
      a = 0.;
      b = 1.;
    } else if (kind == QuadratureKind::laguerre) {
      a = 0.;
      b = 1.;
    } else if (kind == QuadratureKind::trapezoidal) {
      a = 0.;
      b = 1.;
    }
  }

  bool operator<(const QuadratureType &x, const QuadratureType &y)
  {
    if (x.kind != y.kind) return static_cast<int>(x.kind) < static_cast<int>(y.kind);
    if (!is_close(x.a, y.a)) return x.a < y.a;
    if (!is_close(x.b, y.b)) return x.b < y.b;
    if (!is_close(x.alpha, y.alpha)) return x.alpha < y.alpha;
    if (!is_close(x.beta, y.beta)) return x.beta < y.beta;
    return false;
  }

  template <typename NT> Quadrature<NT>::Quadrature(const size_t order, const QuadratureType _t) { reinit(order, _t); }
  template <typename NT> Quadrature<NT>::Quadrature() {}

  template <typename NT> void Quadrature<NT>::reinit(const size_t order, const QuadratureType _t)
  {
    this->order = order;
    this->_t = _t;

    // The periodic trapezoidal rule is not a Gauss rule and is not provided by GSL: it uses equispaced
    // nodes on [a,b) with equal weights and, for smooth periodic integrands (endpoints a and b identified,
    // e.g. an azimuthal angle phi in [0,2pi)), converges spectrally. Fill nodes/weights directly.
    if (_t.kind == QuadratureType::trapezoidal) {
      const std::string name = "trapezoidal";
      device_nodes = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
          "device_nodes_" + name + "_" + std::to_string(order), order);
      device_weights = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
          "device_weights_" + name + "_" + std::to_string(order), order);

      auto host_mirror_nodes = Kokkos::create_mirror_view(device_nodes);
      auto host_mirror_weights = Kokkos::create_mirror_view(device_weights);

      // x_i = a + (b-a) i/order, weight = (b-a)/order. The node at b is omitted (identified with a).
      const double h = (_t.b - _t.a) / static_cast<double>(order);
      for (size_t i = 0; i < order; ++i) {
        host_mirror_nodes(i) = _t.a + h * static_cast<double>(i);
        host_mirror_weights(i) = h;
      }

      Kokkos::deep_copy(device_nodes, host_mirror_nodes);
      Kokkos::deep_copy(device_weights, host_mirror_weights);

      host_nodes = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
          "host_nodes_" + name + "_" + std::to_string(order), order);
      host_weights = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
          "host_weights_" + name + "_" + std::to_string(order), order);
      Kokkos::deep_copy(host_nodes, host_mirror_nodes);
      Kokkos::deep_copy(host_weights, host_mirror_weights);
      return;
    }

    gsl_integration_fixed_workspace *w;
    const gsl_integration_fixed_type *T;

    // create gsl workspace and set quadrature type
    std::string name;
    switch (_t.kind) {
    case QuadratureType::legendre:
      T = gsl_integration_fixed_legendre;
      name = "legendre";
      break;
    case QuadratureType::chebyshev:
      T = gsl_integration_fixed_chebyshev;
      name = "chebyshev";
      break;
    case QuadratureType::chebyshev2:
      T = gsl_integration_fixed_chebyshev2;
      name = "chebyshev2";
      break;
    case QuadratureType::hermite:
      T = gsl_integration_fixed_hermite;
      name = "hermite";
      break;
    case QuadratureType::jacobi:
      T = gsl_integration_fixed_jacobi;
      name = "jacobi";
      break;
    case QuadratureType::laguerre:
      T = gsl_integration_fixed_laguerre;
      name = "laguerre";
      break;
    default:
      throw std::runtime_error("Unknown quadrature type");
    }
    w = gsl_integration_fixed_alloc(T, order, _t.a, _t.b, _t.alpha, _t.beta);

    // get nodes and weights
    const double *gsl_nodes = gsl_integration_fixed_nodes(w);
    const double *gsl_weights = gsl_integration_fixed_weights(w);

    device_nodes = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_nodes_" + name + "_" + std::to_string(order), order);
    device_weights = Kokkos::View<NT *, Kokkos::DefaultExecutionSpace::memory_space>(
        "device_weights_" + name + "_" + std::to_string(order), order);

    // create a host mirror view
    auto host_mirror_nodes = Kokkos::create_mirror_view(device_nodes);
    auto host_mirror_weights = Kokkos::create_mirror_view(device_weights);

    // copy data from gsl to host mirror view
    for (size_t i = 0; i < order; ++i) {
      host_mirror_nodes(i) = gsl_nodes[i];
      host_mirror_weights(i) = gsl_weights[i];
    }

    // copy data from host mirror view to device view
    Kokkos::deep_copy(device_nodes, host_mirror_nodes);
    Kokkos::deep_copy(device_weights, host_mirror_weights);

    // copy data from host mirror view to host view
    host_nodes = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_nodes_" + name + "_" + std::to_string(order), order);
    host_weights = Kokkos::View<NT *, Kokkos::DefaultHostExecutionSpace::memory_space>(
        "host_weights_" + name + "_" + std::to_string(order), order);
    Kokkos::deep_copy(host_nodes, host_mirror_nodes);
    Kokkos::deep_copy(host_weights, host_mirror_weights);

    // free gsl workspace
    gsl_integration_fixed_free(w);
  }

  template <typename NT> size_t Quadrature<NT>::size() const { return order; }
  template <typename NT> QuadratureType Quadrature<NT>::get_type() const { return _t; }

  // explicit instantiation
  template class Quadrature<double>;
  template class Quadrature<float>;
} // namespace DiFfRG