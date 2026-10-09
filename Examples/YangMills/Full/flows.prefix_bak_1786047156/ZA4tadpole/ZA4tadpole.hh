#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename> class ZA4tadpole_kernel;

  class ZA4tadpole_integrator
  {
  public:
    ZA4tadpole_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json);

    using Regulator = DiFfRG::PolynomialExpRegulator<>;

    Integrator_p2_4D_2ang<4, double, ZA4tadpole_kernel<Regulator>, DiFfRG::GPU_exec> integrator;

    DiFfRG::GPU_exec map(double *dest, const LogLinLinPeriodicCoordinates &coordinates, const double &k,
                         const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA3,
                         const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZAcbc,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4SP,
                         const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA4tadpole,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA);

    template <typename IT, typename C, typename... T> DiFfRG::GPU_exec map(IT *dest, const C &coordinates, const device::tuple<T...> &args)
    {
      return device::apply([&](const auto... t) { return map(dest, coordinates, t...); }, args);
    }

    void get(double &dest, const double &S0, const double &S1, const double &SPhi, const double &k,
             const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA3,
             const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZAcbc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4SP,
             const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA4tadpole,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
             const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA);

    template <typename IT, typename... T>
    void get(IT &dest, const double &S0, const double &S1, const double &SPhi, const device::tuple<T...> &args)
    {
      device::apply([&](const auto... t) { get(dest, S0, S1, SPhi, t...); }, args);
    }

  private:
    DiFfRG::QuadratureProvider &quadrature_provider;
  };
} // namespace DiFfRG
using DiFfRG::ZA4tadpole_integrator;