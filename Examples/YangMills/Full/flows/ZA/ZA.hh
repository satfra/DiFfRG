#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"
#include "kernel.hh"

namespace DiFfRG
{

  class ZA_integrator
  {
  public:
    ZA_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &config);

    using Regulator = DiFfRG::PolynomialExpRegulator<>;

    Integrator_p2_1ang<4, double, ZA_kernel<Regulator>, DiFfRG::GPU_exec> integrator;

    DiFfRG::GPU_exec map(double *dest, const FocusedLogCoordinates1D<double> &coordinates, const double &k,
                         const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA3,
                         const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZAcbc,
                         const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA4SP,
                         const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA4tadpole,
                         const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZc,
                         const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &Zc,
                         const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZA,
                         const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA);

    template <typename IT, typename C, typename... T>
    DiFfRG::GPU_exec map(IT *dest, const C &coordinates, const device::tuple<T...> &args)
    {
      return device::apply([&](const auto... t) { return map(dest, coordinates, t...); }, args);
    }

    void get(double &dest, const double &p, const double &k,
             const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA3,
             const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZAcbc,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA4SP,
             const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA4tadpole,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZc,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &Zc,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZA,
             const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA);

    template <typename IT, typename... T> void get(IT &dest, const double &p, const device::tuple<T...> &args)
    {
      device::apply([&](const auto... t) { get(dest, p, t...); }, args);
    }

  private:
    DiFfRG::QuadratureProvider &quadrature_provider;
  };
} // namespace DiFfRG
using DiFfRG::ZA_integrator;