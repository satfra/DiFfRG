#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/interpolation.hh"
#include "kernel.hh"

namespace DiFfRG {

  class VAuto_integrator
  {
    public:
    VAuto_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config)
    ;


    using Regulator = DiFfRG::PolynomialExpRegulator<>;

    Integrator_fT_p2<4, double, VAuto_kernel<Regulator>, DiFfRG::GPU_exec> integrator;

    Integrator_fT_p2<4, autodiff::real, VAuto_kernel<Regulator>, DiFfRG::GPU_exec> integrator_AD;

    Integrator_fT_p2<4, autodiff::Real<2, double>, VAuto_kernel<Regulator>, DiFfRG::GPU_exec> integrator_AD2;

    void get(double& dest, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZPi, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZSig)
    ;

    template<typename IT, typename ...T>
    void get(IT& dest, const device::tuple<T...>& args)
    {
      device::apply([&](const auto...t){get(dest, t...);}, args);
    }

    void get(autodiff::real& dest, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const autodiff::real& d1V, const autodiff::real& d2V, const autodiff::real& d3V, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZPi, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZSig)
    ;

    void get(autodiff::Real<2, double>& dest, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const autodiff::Real<2, double>& d1V, const autodiff::Real<2, double>& d2V, const autodiff::Real<2, double>& d3V, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZPi, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZSig)
    ;

    void map_points(const DiFfRG::PointSpan<double> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& hPhi, const DiFfRG::PointArg<double>& etaPi0, const DiFfRG::PointArg<double>& rhoPhi, const DiFfRG::PointArg<double>& d1V, const DiFfRG::PointArg<double>& d2V, const DiFfRG::PointArg<double>& d3V, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZPi, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZSig)
    ;

    void map_points(const DiFfRG::PointSpan<autodiff::real> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& hPhi, const DiFfRG::PointArg<double>& etaPi0, const DiFfRG::PointArg<double>& rhoPhi, const DiFfRG::PointArg<autodiff::real>& d1V, const DiFfRG::PointArg<autodiff::real>& d2V, const DiFfRG::PointArg<autodiff::real>& d3V, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZPi, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZSig)
    ;

    void map_points(const DiFfRG::PointSpan<autodiff::Real<2, double>> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& hPhi, const DiFfRG::PointArg<double>& etaPi0, const DiFfRG::PointArg<double>& rhoPhi, const DiFfRG::PointArg<autodiff::Real<2, double>>& d1V, const DiFfRG::PointArg<autodiff::Real<2, double>>& d2V, const DiFfRG::PointArg<autodiff::Real<2, double>>& d3V, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZPi, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZSig)
    ;
    private:
    DiFfRG::QuadratureProvider& quadrature_provider;
  };
}
using DiFfRG::VAuto_integrator;