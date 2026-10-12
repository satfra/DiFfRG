// DiFfRG
#include <DiFfRG/common/polynomials.hh>

namespace DiFfRG
{
  Polynomial::Polynomial(const std::vector<double> &coefficients) : coefficients(coefficients) {}

  const double &Polynomial::operator[](const uint &i) const { return coefficients[i]; }
  double &Polynomial::operator[](const uint &i) { return coefficients[i]; }

  double Polynomial::operator()(const double &x) const
  {
    double result = 0.;
    for (auto c = coefficients.rbegin(); c != coefficients.rend(); ++c)
      result = result * x + *c;
    return result;
  }

  void Polynomial::operator*=(const double &scalar)
  {
    for (uint i = 0; i < coefficients.size(); ++i)
      coefficients[i] *= scalar;
  }

  void Polynomial::operator*=(const Polynomial &other)
  {
    if (coefficients.empty() || other.coefficients.empty()) {
      coefficients.clear();
      return;
    }
    std::vector<double> new_coefficients(coefficients.size() + other.coefficients.size() - 1);
    for (uint i = 0; i < coefficients.size(); ++i)
      for (uint j = 0; j < other.coefficients.size(); ++j)
        new_coefficients[i + j] += coefficients[i] * other.coefficients[j];
    coefficients = new_coefficients;
  }

  Polynomial Polynomial::anti_derivative(const double &integration_constant) const
  {
    std::vector<double> new_coefficients(coefficients.size() + 1);
    new_coefficients[0] = integration_constant;
    for (uint i = 0; i < coefficients.size(); ++i)
      new_coefficients[i + 1] = coefficients[i] / (i + 1);
    return Polynomial(new_coefficients);
  }

  Polynomial Polynomial::derivative() const
  {
    if (coefficients.empty()) return Polynomial({});
    std::vector<double> new_coefficients(coefficients.size() - 1);
    for (uint i = 1; i < coefficients.size(); ++i)
      new_coefficients[i - 1] = coefficients[i] * i;
    return Polynomial(new_coefficients);
  }

  double Polynomial::integral(const double &x_min, const double &x_max) const
  {
    Polynomial anti_derivative = this->anti_derivative(0.);
    return anti_derivative(x_max) - anti_derivative(x_min);
  }

  Polynomial Polynomial::square_argument() const
  {
    if (coefficients.empty()) return Polynomial({});
    std::vector<double> new_coefficients(2 * coefficients.size() - 1);
    for (uint i = 0; i < coefficients.size(); ++i)
      new_coefficients[2 * i] = coefficients[i];
    return new_coefficients;
  }
} // namespace DiFfRG