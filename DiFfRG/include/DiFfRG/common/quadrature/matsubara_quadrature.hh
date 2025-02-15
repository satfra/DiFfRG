#pragma once

// DiFfRG
#include <DiFfRG/common/quadrature.hh>

namespace DiFfRG
{
  /**
   * @brief A quadrature rule for (bosonic) Matsubara frequencies, based on the method of Monien [1].
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
    MatsubaraQuadrature(const NT T, const NT typical_E = 1., const NT step = 1) {}

  private:
    /**
     * @brief Nodes of the quadrature rule.
     */
    std::vector<NT> x;

    /**
     * @brief Weights of the quadrature rule.
     */
    std::vector<NT> w;
  };

} // namespace DiFfRG