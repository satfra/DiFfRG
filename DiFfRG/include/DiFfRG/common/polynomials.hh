#pragma once

// standard library
#include <cmath>
#include <vector>

// DiFfRG
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  /**
   * @brief A polynomial \f$P(x) = \sum_{i=0}^{n-1} c_i x^i\f$ with double coefficients.
   *
   * An empty coefficient list is the zero polynomial.
   */
  class Polynomial
  {
  public:
    /// From the coefficients \f$c_0, c_1, \dots\f$, lowest power first.
    Polynomial(const std::vector<double> &coefficients);

    /// Coefficient \f$c_i\f$ (unchecked index).
    const double &operator[](const uint &i) const;

    /// Coefficient \f$c_i\f$ (unchecked index).
    double &operator[](const uint &i);

    /// \f$P(x)\f$.
    double operator()(const double &x) const;

    /// \f$P \to s P\f$.
    void operator*=(const double &scalar);

    /// \f$P \to P Q\f$.
    void operator*=(const Polynomial &other);

    /**
     * @brief The antiderivative \f$C + \sum_i c_i x^{i+1}/(i+1)\f$.
     *
     * @param integration_constant the constant term \f$C\f$.
     */
    Polynomial anti_derivative(const double &integration_constant) const;

    /// The derivative \f$P'(x)\f$; the zero polynomial for a constant or empty \f$P\f$.
    Polynomial derivative() const;

    /// \f$\int_{x_\text{min}}^{x_\text{max}} P(x)\, dx\f$.
    double integral(const double &x_min, const double &x_max) const;

    /// The polynomial \f$Q(x) = P(x^2)\f$.
    Polynomial square_argument() const;

  private:
    std::vector<double> coefficients;
  };
} // namespace DiFfRG
