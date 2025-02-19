#pragma once

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/quadrature/diagonalization.hh>
#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>

// standard library
#include <string>
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

  enum class QuadratureType { legendre, chebyshev, laguerre, hermite, jacobi, count };

  template <typename NT> class Quadrature
  {
  public:
    Quadrature();
    Quadrature(const size_t order, const QuadratureType _t);

    void reinit(const size_t order, const QuadratureType _t);

    const std::vector<NT> &nodes() const;
    const std::vector<NT> &weights() const;

    const NT *device_nodes();
    const NT *device_weights();

    size_t size() const;
    QuadratureType get_type() const;

  private:
    QuadratureType _t;
    uint order;

    std::vector<NT> m_nodes;
    std::vector<NT> m_weights;
#ifdef __CUDACC__
    thrust::device_vector<NT> m_device_nodes;
    thrust::device_vector<NT> m_device_weights;

    void move_device_data();
#endif
  };
} // namespace DiFfRG