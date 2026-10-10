#pragma once

// DiFfRG
#include <DiFfRG/common/kokkos.hh>
#include <DiFfRG/common/quadrature/diagonalization.hh>

// standard library
#include <vector>

/**
 * @file quadrature.hh
 * @brief One-dimensional quadrature rules \f$\int dx\, W(x) f(x) \approx \sum_i w_i f(x_i)\f$, with the nodes and
 * weights stored on host and device.
 */

namespace DiFfRG
{
  namespace internal
  {
    /**
     * @brief Gauss rule for a weight function \f$W\f$ from its three-term recurrence (Golub-Welsch algorithm).
     *
     * The orthogonal polynomials of \f$W\f$ obey \f$p_{j+1}(x) = (x - a_j) p_j(x) - b_j p_{j-1}(x)\f$. The nodes are
     * the eigenvalues of the symmetric tridiagonal matrix with diagonal \f$a_j\f$ and off-diagonal \f$\sqrt{b_j}\f$;
     * the weights are \f$w_i = \mu_0 v_{0,i}^2\f$ with \f$v_{0,i}\f$ the first component of the normalized \f$i\f$-th
     * eigenvector. See Press et al., Numerical Recipes, 3rd ed., ch. 4.6.2.
     *
     * @param a the \f$a_j\f$, \f$j = 0 \dots n-1\f$; overwritten.
     * @param b the \f$b_j\f$, same length as `a`; overwritten. The last entry is not used.
     * @param mu0 the zeroth moment \f$\mu_0 = \int dx\, W(x)\f$.
     * @param x the \f$n\f$ nodes (output).
     * @param w the \f$n\f$ weights (output); they include \f$W\f$, i.e. \f$\sum_i w_i f(x_i) \approx \int W f\f$.
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
  } // namespace internal

  // clang-format off
  /**
   * @brief Which quadrature rule to build: the kind and its parameters \f$a, b, \alpha, \beta\f$.
   *
   * The rules come from GSL (`gsl_integration_fixed`), except `trapezoidal`. The weights include the weight
   * function, \f$\sum_i w_i f(x_i) \approx \int dx\, W(x) f(x)\f$:
   *
   * | kind | \f$W(x)\f$ | interval | defaults |
   * |------|-------------|----------|----------|
   * | `legendre` | \f$1\f$ | \f$[a, b]\f$ | \f$[0, 1]\f$ |
   * | `chebyshev` | \f$1/\sqrt{(b - x)(x - a)}\f$ | \f$[a, b]\f$ | \f$[-1, 1]\f$ |
   * | `chebyshev2` | \f$\sqrt{(b - x)(x - a)}\f$ | \f$[a, b]\f$ | \f$[-1, 1]\f$ |
   * | `jacobi` | \f$(b - x)^\alpha (x - a)^\beta\f$ | \f$[a, b]\f$ | \f$[0, 1]\f$, \f$\alpha = 1\f$, \f$\beta = 2\f$ |
   * | `laguerre` | \f$(x - a)^\alpha e^{-b (x - a)}\f$ | \f$[a, \infty)\f$ | \f$a = 0\f$, \f$b = 1\f$, \f$\alpha = 1\f$ |
   * | `hermite` | \f$\lvert x - a \rvert^\alpha e^{-b (x - a)^2}\f$ | \f$(-\infty, \infty)\f$ | \f$a = 0\f$, \f$b = 1\f$, \f$\alpha = 1\f$ |
   * | `trapezoidal` | \f$1\f$ | \f$[a, b)\f$, periodic | \f$[0, 1)\f$ |
   *
   * Note the defaults \f$\alpha = 1\f$ for `laguerre` and `hermite`: set `alpha = 0` for the standard weights
   * \f$e^{-x}\f$ and \f$e^{-x^2}\f$. `trapezoidal` uses \f$n\f$ equidistant nodes \f$x_i = a + i h\f$, \f$h = (b-a)/n\f$,
   * with weights \f$h\f$; for smooth periodic integrands (e.g. an azimuthal angle on \f$[0, 2\pi)\f$) it converges
   * exponentially.
   */
  // clang-format on
  struct QuadratureType {
    /// The available rules, see QuadratureType.
    enum QuadratureKind { legendre, chebyshev, chebyshev2, laguerre, hermite, jacobi, trapezoidal, count };
    QuadratureKind kind = legendre;
    /// Interval or shift and scale parameters, see the table in QuadratureType.
    double a = 0., b = 1.;
    /// Exponents of the weight function, see the table in QuadratureType.
    double alpha = 1., beta = 2.;

    /// The rule `kind` with its default parameters.
    QuadratureType(const QuadratureKind kind);
    /// Gauss-Legendre on \f$[0, 1]\f$.
    QuadratureType() = default;
  };
  /// Ordering, so that QuadratureType can be a map key; parameters are compared to machine precision.
  bool operator<(const QuadratureType &x, const QuadratureType &y);

  /**
   * @brief An \f$n\f$-point quadrature rule, see QuadratureType, with nodes and weights on host and device.
   *
   * @code
   * Quadrature<double> quad(32, QuadratureType::legendre); // 32 points on [0, 1]
   * const auto x = quad.nodes<CPU_memory>();
   * const auto w = quad.weights<CPU_memory>();
   * double integral = 0.;
   * for (size_t i = 0; i < quad.size(); ++i) integral += w[i] * f(x[i]);
   * @endcode
   *
   * Usually obtained from a QuadratureProvider, which caches rules.
   *
   * @tparam NT the type of nodes and weights (computed in double precision).
   */
  template <typename NT> class Quadrature
  {
  public:
    /// An empty rule; call reinit() before use.
    Quadrature();
    /// The rule of type `_t` with `order` points.
    Quadrature(const size_t order, const QuadratureType _t);

    /// Rebuild as the rule of type `_t` with `order` points.
    void reinit(const size_t order, const QuadratureType _t);

    /// The nodes \f$x_i\f$ in `MemorySpace` (the device memory space or the host memory space).
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

    /// The weights \f$w_i\f$ in `MemorySpace`.
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

    /// Number of points \f$n\f$.
    size_t size() const;

    /// The rule's type and parameters.
    QuadratureType get_type() const;

  private:
    QuadratureType _t;
    unsigned order = 0;

    Kokkos::View<NT *, GPU_memory> device_nodes;
    Kokkos::View<NT *, GPU_memory> device_weights;

    Kokkos::View<NT *, CPU_memory> host_nodes;
    Kokkos::View<NT *, CPU_memory> host_weights;
  };
} // namespace DiFfRG