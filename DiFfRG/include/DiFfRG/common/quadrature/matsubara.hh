#pragma once

// DiFfRG
#include <DiFfRG/common/math.hh>
#include <DiFfRG/common/quadrature/quadrature.hh>

// C++ standard library
#include <cmath>
#include <vector>

namespace DiFfRG
{
  /**
   * @brief A quadrature rule for (bosonic) Matsubara frequencies, based on the method of Monien [1]. This class
   * provides nodes and weights for the summation
   * \f[
   * T \sum_{n=1}^{\infty} f(2\pi n T) \approx T \sum_{i=1}^{N-1} w_i f(x_i)
   * \f]
   *
   * [1] H. Monien, "Gaussian quadrature for sums: a rapidly convergent summation scheme", Math. Comp. 79, 857 (2010).
   * doi:10.1090/S0025-5718-09-02289-3
   *
   * @tparam NT numeric type to be used for all calculations
   */
  template <typename NT> class MatsubaraQuadrature
  {
  public:
    /**
     * @brief Create a new quadrature rule for Matsubara frequencies.
     *
     * @param T The temperature.
     * @param typical_E A typical energy scale, which determines the number of nodes in the quadrature rule.
     * @param step The step size of considered node sizes (e.g. step=2 implies only even numbers of nodes).
     */
    MatsubaraQuadrature(const NT T, const NT typical_E = 1., const size_t step = 1, const size_t min_size = 0,
                        const size_t max_size = powr<10>(2))
        : T(T), typical_E(typical_E)
    {
      // Determine the number of nodes in the quadrature rule.
      const NT E_max = 10 * std::abs(typical_E);
      m_size = 5 + int(std::sqrt(4 * E_max / (M_PI * M_PI * std::abs(T))));
      m_size = (size_t)std::ceil(m_size / (double)step) * step;
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

    /**
     * @brief Get the nodes of the quadrature rule.
     *
     * @return The nodes of the quadrature rule.
     */
    const std::vector<NT> &nodes() const { return x; }

    /**
     * @brief Get the weights of the quadrature rule.
     *
     * @return The weights of the quadrature rule.
     */
    const std::vector<NT> &weights() const { return w; }

    /**
     * @brief Get the size of the quadrature rule.
     */
    size_t size() const { return m_size; }

    /**
     * @brief Get the temperature of the quadrature rule.
     */
    NT get_T() const { return T; }

    /**
     * @brief Get the typical energy scale of the quadrature rule.
     */
    NT get_typical_E() const { return typical_E; }

    template <typename F> NT sum(const F &f) const
    {
      NT sum = T * f(static_cast<NT>(0));
      for (size_t i = 0; i < m_size; ++i)
        sum += w[i] * (f(x[i]) + f(-x[i]));
      return sum;
    }

  private:
    const double T, typical_E;
    /**
     * @brief Nodes of the quadrature rule.
     */
    std::vector<NT> x;

    /**
     * @brief Weights of the quadrature rule.
     */
    std::vector<NT> w;

    /**
     * @brief The number of nodes in the quadrature rule.
     */
    size_t m_size;
  };

} // namespace DiFfRG