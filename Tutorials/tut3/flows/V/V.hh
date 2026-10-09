#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/interpolation.hh"
#include "DiFfRG/physics/physics.hh"

namespace DiFfRG
{
  template <typename> class V_kernel;

  class V_integrator
  {
  public:
    V_integrator(DiFfRG::QuadratureProvider &quadrature_provider, const DiFfRG::ConfigTree &json);

    using Regulator = DiFfRG::PolynomialExpRegulator<>;

    Integrator_p2<3, double, V_kernel<Regulator>, DiFfRG::TBB_exec> integrator;

    Integrator_p2<3, autodiff::real, V_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD;

    void get(double &dest, const double &k, const double &N, const double &T, const double &m2Pi, const double &m2Sigma);

    template <typename IT, typename... T> void get(IT &dest, const device::tuple<T...> &args)
    {
      device::apply([&](const auto... t) { get(dest, t...); }, args);
    }

    void get(autodiff::real &dest, const double &k, const double &N, const double &T, const autodiff::real &m2Pi,
             const autodiff::real &m2Sigma);

  private:
    DiFfRG::QuadratureProvider &quadrature_provider;
  };
} // namespace DiFfRG
using DiFfRG::V_integrator;