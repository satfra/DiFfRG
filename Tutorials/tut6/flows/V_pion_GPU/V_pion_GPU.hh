#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/interpolation.hh"
#include "kernel.hh"

namespace DiFfRG {

  class V_pion_GPU_integrator
  {
    public:
    V_pion_GPU_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config)
    ;


    using Regulator = DiFfRG::PolynomialExpRegulator<>;

    Integrator_fT_p2<4, double, V_pion_GPU_kernel<Regulator>, DiFfRG::GPU_exec> integrator;

    Integrator_fT_p2<4, autodiff::real, V_pion_GPU_kernel<Regulator>, DiFfRG::GPU_exec> integrator_AD;

    Integrator_fT_p2<4, autodiff::Real<2, double>, V_pion_GPU_kernel<Regulator>, DiFfRG::GPU_exec> integrator_AD2;

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

    void map_points(const DiFfRG::PointSpan<double> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& m2Pi)
    ;

    void map_points(const DiFfRG::PointSpan<autodiff::real> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<autodiff::real>& m2Pi)
    ;

    void map_points(const DiFfRG::PointSpan<autodiff::Real<2, double>> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<autodiff::Real<2, double>>& m2Pi)
    ;
    private:
    DiFfRG::QuadratureProvider& quadrature_provider;
  };
}
using DiFfRG::V_pion_GPU_integrator;