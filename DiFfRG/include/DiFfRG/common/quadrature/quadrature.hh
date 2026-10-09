#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/quadrature/diagonalization.hh>

// standard library
#include <vector>

namespace DiFfRG
{
  /**
   * @brief Obtain the quadrature rule from a given three-term recurrence relation.
   *
   * For a reference, see "Numerical Recipes in C++" by Press et al., third edition, chapter 4.6.2.
   *
   * @tparam T numeric type
   * @param a Diagonal elements of the tridiagonal Jacobi matrix
   * @param b Squares of the off-diagonal elements of the tridiagonal Jacobi matrix
   * @param mu0 The weight function at the left endpoint of the interval
   * @param x Quadrature points (output)
   * @param w Quadrature weights (output)
   */
  template <typename T>
  void make_quadrature(std::vector<T> &a, std::vector<T> &b, const T mu0, std::vector<T> &x, std::vector<T> &w)
  {
    const size_t n = a.size();
    if (b.size() != n) throw std::runtime_error("The size of b must be a.size().");
    x.resize(n);
    w.resize(n);

    w[0] = 1.;
    for (size_t i = 1; i < n; ++i) {
      b[i - 1] = std::sqrt(b[i - 1]);
      w[i] = 0.0;
    }
    diagonalize_tridiagonal_symmetric_matrix(a, b, w);
    for (size_t i = 0; i < n; i++) {
      x[i] = a[i];
      w[i] = mu0 * powr<2>(w[i]);
    }
  }

  struct QuadratureType {
    enum QuadratureKind { legendre, chebyshev, chebyshev2, laguerre, hermite, jacobi, trapezoidal, count };
    QuadratureKind kind;
    double a = 0., b = 1.;
    double alpha = 1., beta = 2.;

    QuadratureType(const QuadratureKind kind);
    QuadratureType() = default;
  };
  bool operator<(const QuadratureType &x, const QuadratureType &y);

  template <typename NT> class Quadrature
  {
  public:
    Quadrature();
    Quadrature(const size_t order, const QuadratureType _t);

    void reinit(const size_t order, const QuadratureType _t);

    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> nodes() const
    {
      if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultExecutionSpace::memory_space>) {
        return device_nodes;
      } else if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultHostExecutionSpace::memory_space>) {
        return host_nodes;
      } else {
        throw std::runtime_error("Invalid memory space");
      }
    }

    template <typename MemorySpace> Kokkos::View<const NT *, MemorySpace> weights() const
    {
      if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultExecutionSpace::memory_space>) {
        return device_weights;
      } else if constexpr (std::is_same_v<MemorySpace, Kokkos::DefaultHostExecutionSpace::memory_space>) {
        return host_weights;
      } else {
        throw std::runtime_error("Invalid memory space");
      }
    }

    size_t size() const;

    QuadratureType get_type() const;

  private:
    QuadratureType _t;
    unsigned order;

    Kokkos::View<NT *, GPU_memory> device_nodes;
    Kokkos::View<NT *, GPU_memory> device_weights;

    Kokkos::View<NT *, CPU_memory> host_nodes;
    Kokkos::View<NT *, CPU_memory> host_weights;
  };
} // namespace DiFfRG