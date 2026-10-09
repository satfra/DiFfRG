#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/interpolation.hh"
#include "kernel.hh"

namespace DiFfRG {

  class V_pion_integrator
  {
    public:
    V_pion_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config)
    ;


    using Regulator = DiFfRG::PolynomialExpRegulator<>;

    Integrator_p2<3, double, V_pion_kernel<Regulator>, DiFfRG::TBB_exec> integrator;

    Integrator_p2<3, autodiff::real, V_pion_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD;

    Integrator_p2<3, autodiff::Real<2, double>, V_pion_kernel<Regulator>, DiFfRG::TBB_exec> integrator_AD2;

    void get(double& dest, const double& k, const double& N, const double& T, const double& m2Pi)
    ;

    template<typename IT, typename ...T>
    void get(IT& dest, const device::tuple<T...>& args)
    {
      device::apply([&](const auto...t){get(dest, t...);}, args);
    }

    void get(autodiff::real& dest, const double& k, const double& N, const double& T, const autodiff::real& m2Pi)
    ;

    void get(autodiff::Real<2, double>& dest, const double& k, const double& N, const double& T, const autodiff::Real<2, double>& m2Pi)
    ;
    private:
    DiFfRG::QuadratureProvider& quadrature_provider;
  };
}
using DiFfRG::V_pion_integrator;