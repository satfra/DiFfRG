#pragma once

#include <DiFfRG/common/utils.hh>
#include <DiFfRG/physics/loop_integrals.hh>

namespace DiFfRG
{
  /**
   * @brief Find how far a momentum integral must extend for a given regulator: returns \f$x_\text{extent}\f$, the
   * upper limit of \f$q^2/k^2\f$.
   *
   * Every regulated loop carries \f$\partial_t R_k(q)\f$, which decays for \f$q^2 \gg k^2\f$. With \f$x = q^2/k^2\f$
   * and \f$k = 1\f$, this function integrates the typical loop integrand
   * \f[
   *   f(x) = \frac{\partial_t R(x)}{x + R(x)}
   * \f]
   * with LoopIntegrals::integrate() in `dim` dimensions up to \f$x_\text{extent}\f$, \f$2 x_\text{extent}\f$ and
   * \f$10 x_\text{extent}\f$, giving \f$I_1, I_2, I_3\f$. Starting from \f$x_\text{extent} = 1\f$, it grows
   * \f$x_\text{extent}\f$ by a factor 1.15 until both \f$|I_2 - I_1|/|I_1|\f$ and \f$|I_2 - I_3|/|I_2|\f$ are below
   * `/integration/x_extent_tolerance` (default \f$10^{-5}\f$).
   *
   * - A regulator that vanishes for all \f$x > 1\f$, such as LitimRegulator, gives \f$x_\text{extent} = 1\f$.
   * - The result is computed once per program run for each `Regulator` and `dim`, and then reused: later calls
   *   ignore their `config`.
   * - The integrators pass their own `dim`: the momentum dimension for vacuum integrators, the spacetime dimension
   *   at finite \f$T\f$ (so that a regulator in \f$q_0^2 + \vec q^{\,2}\f$ is covered too). A larger `dim` weights
   *   the tail more and gives a larger \f$x_\text{extent}\f$.
   * - Throws `std::runtime_error` if the tolerance cannot be reached.
   * - Prints its progress if `/output/verbosity` is above 1.
   *
   * @tparam Regulator provides static `RB(k2, q2)` and `RBdot(k2, q2)`, see regulators.hh
   * @tparam dim dimension of the test integral, \f$\geq 1\f$
   * @param config parameter file
   * @return \f$x_\text{extent}\f$, in units of \f$k^2\f$
   */
  template <typename Regulator, int dim> double optimize_x_extent(const ConfigTree &config)
  {
    static bool already_run = false;
    static double x_extent = 1.;

    if (already_run) return x_extent;

    const int verbosity = config.get_int("/output/verbosity", 0);
    const double x_extent_tolerance = config.get_double_or_warn("/integration/x_extent_tolerance", 1e-5);
    // Fixed rules, independent of /integration/x_order: they only locate the cutoff of a smooth regulator
    // function, and deal.II builds a QGauss of order n in O(n^2), which would be slow for large x_order.
    constexpr uint order = 256;
    auto optimize_x = [](double x) -> double { return 1. / (x + Regulator::RB(1., x)) * Regulator::RBdot(1., x); };

    // Integrate optimize_x over [0, x_extent], [0, 2 x_extent] and [0, 10 x_extent] (I1, I2, I3) and grow x_extent
    // until |I2 - I1| / |I1| and |I2 - I3| / |I2| are both below x_extent_tolerance.

    const dealii::QGauss<1> quadrature_1(1 * order);
    const dealii::QGauss<1> quadrature_2(2 * order);
    const dealii::QGauss<1> quadrature_3(10 * order);

    double eps1 = 1., eps2 = 1.;
    uint decrease_counter = 0;

    if (verbosity > 1) std::cout << "Optimizing x_extent" << std::endl;

    bool all_zero = true;
    for (uint i = 0; i < 10; i++) {
      const double x = 1. + 1e-3 + (std::pow(1.5, (double)i) - 1.);
      const double x_test = optimize_x(x);
      if (!is_close(x_test, 0.)) all_zero = false;
    }
    if (all_zero) {
      x_extent = 1.;
      if (verbosity > 1) std::cout << "x_extent is set to 1.\n" << std::endl;
      already_run = true;
      return x_extent;
    }

    while (eps1 > x_extent_tolerance || eps2 > x_extent_tolerance) {
      const double I1 = LoopIntegrals::integrate<double, dim>(optimize_x, quadrature_1, x_extent, 1.);
      const double I2 = LoopIntegrals::integrate<double, dim>(optimize_x, quadrature_2, 2. * x_extent, 1.);
      const double I3 = LoopIntegrals::integrate<double, dim>(optimize_x, quadrature_3, 10. * x_extent, 1.);

      if (std::abs((I2 - I1) / I1) > eps1 && std::abs((I2 - I3) / I2) > eps2)
        decrease_counter++;
      else
        decrease_counter = 0;
      if (decrease_counter > 2)
        throw std::runtime_error("Cannot reach requested precision for x_extent - increase x_extent_tolerance.");

      eps1 = std::abs((I2 - I1) / I1);
      eps2 = std::abs((I2 - I3) / I2);

      if (verbosity > 1)
        std::cout << "x_extent: " << x_extent << " I1: " << I1 << " I2: " << I2 << " I3: " << I3 << " eps1: " << eps1
                  << " eps2: " << eps2 << std::endl;

      if (eps1 > x_extent_tolerance || eps2 > x_extent_tolerance) x_extent *= 1.15;
    }
    if (verbosity > 1) std::cout << "Optimizing x_extent done.\n" << std::endl;

    already_run = true;
    return x_extent;
  }
} // namespace DiFfRG