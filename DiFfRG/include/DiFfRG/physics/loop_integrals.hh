#pragma once

// standard library
#include <cmath>

// external libraries
#include <deal.II/base/quadrature_lib.h>
#include <tbb/tbb.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>

namespace DiFfRG
{
  using namespace dealii;

  namespace LoopIntegrals
  {
    /**
     * @brief Performs the integral
     * \f[
     *    \int d\Omega_{d} \int_0^\infty dq f(q^2) q^{d-1}
     * \f]
     *
     * @tparam NT Number type used throughout.
     * @tparam d Dimension of the integral.
     * @tparam FUN Function type of the integrand.
     * @param fun Integrand, should be a callable with signature NT(double q^2).
     * @param x_quadrature Quadrature rule for the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param x_extent Extent of the integral over \f$q^2\f$, \f$q^2 = x * k^2\f$.
     * @param k Momentum scale as defined above.
     * @return NT Result of the integral.
     */
    template <typename NT, int d, typename FUN>
    NT integrate(const FUN &fun, const QGauss<1> &x_quadrature, const double x_extent, const double k)
    {
      const auto &x_q_p = x_quadrature.get_points();
      const auto &x_q_w = x_quadrature.get_weights();
      const int x_size = x_quadrature.size();

      static_assert(d > 0, "Dimension must be greater than zero");
      // Dimension of the spatial integral
      const double S_d = 2. * std::pow(M_PI, d / 2.) / std::tgammal(d / 2.);
      // Prefactor for the spatial integral
      const double prefactor = S_d                    // angular integral
                               * powr<-d>(2. * M_PI); // fourier factors

      // Summed serially on purpose: tbb::parallel_reduce would make the summation order depend on
      // work stealing, and the result of this integral is compared against a tolerance in
      // optimize_x_extent(). A last-bit difference there can flip the convergence test and change
      // x_extent - the upper limit of every subsequent momentum integral - by a factor of 1.15.
      // The grids here are a few thousand points and this runs a handful of times at startup, so
      // there is nothing to gain from parallelising it.
      NT result(0);
      const double q_max = std::sqrt(x_extent) * k;
      for (int x_it = 0; x_it < x_size; x_it++) {
        const double q = x_q_p[x_it][0] * q_max;
        const double q_weight = x_q_w[x_it] * q_max;
        const double q2 = powr<2>(q);

        result += q_weight * prefactor * std::pow(q, d - 1) // integral over q in d dimensions
                  * fun(q2);                                // integrand
      }
      return result;
    }
  } // namespace LoopIntegrals
} // namespace DiFfRG