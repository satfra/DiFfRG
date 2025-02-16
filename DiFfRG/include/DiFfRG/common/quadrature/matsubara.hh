#pragma once

// DiFfRG
#include <DiFfRG/common/cuda_prefix.hh>
#include <DiFfRG/common/math.hh>

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
    static size_t predict_size(const NT T, const NT typical_E = 1., const size_t step = 1);

    /**
     * @brief Create a new quadrature rule for Matsubara frequencies.
     *
     * @param T The temperature.
     * @param typical_E A typical energy scale, which determines the number of nodes in the quadrature rule.
     * @param step The step size of considered node sizes (e.g. step=2 implies only even numbers of nodes).
     */
    MatsubaraQuadrature(const NT T, const NT typical_E = 1., const size_t step = 1, const size_t min_size = 0,
                        const size_t max_size = powr<10>(2));

    MatsubaraQuadrature();

    void reinit(const NT T, const NT typical_E = 1., const size_t step = 1, const size_t min_size = 0,
                const size_t max_size = powr<10>(2));

    /**
     * @brief Get the nodes of the quadrature rule.
     *
     * @return The nodes of the quadrature rule.
     */
    const std::vector<NT> &nodes() const;

    /**
     * @brief Get the weights of the quadrature rule.
     *
     * @return The weights of the quadrature rule.
     */
    const std::vector<NT> &weights() const;

    /**
     * @brief Get the size of the quadrature rule.
     */
    size_t size() const;

    /**
     * @brief Get the temperature of the quadrature rule.
     */
    NT get_T() const;

    /**
     * @brief Get the typical energy scale of the quadrature rule.
     */
    NT get_typical_E() const;

    /**
     * @brief Compute a matsubara sum of a given function.
     *
     * @tparam F The function type.
     * @param f The function to be summed. Must have signature NT f(NT x).
     * @return NT The Matsubara sum of the function.
     */
    template <typename F> NT sum(const F &f) const
    {
      NT sum = T * f(static_cast<NT>(0));
      for (size_t i = 0; i < m_size; ++i)
        sum += w[i] * (f(x[i]) + f(-x[i]));
      return sum;
    }

#ifdef USE_CUDA
    const NT *device_nodes();

    const NT *device_weights();
#endif

  private:
    double T, typical_E;
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

#ifdef USE_CUDA
    thrust::device_vector<NT> device_x;
    thrust::device_vector<NT> device_w;

    void move_device_data();
#endif
  };
} // namespace DiFfRG