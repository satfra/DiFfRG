#pragma once

#include "DiFfRG/physics/integration.hh"
#include "DiFfRG/physics/physics.hh"
#include "DiFfRG/physics/interpolation.hh"
#include "kernel.hh"

namespace DiFfRG {

  class Mesons_integrator
  {
    public:
    Mesons_integrator(DiFfRG::QuadratureProvider& quadrature_provider, const DiFfRG::ConfigTree& config)
    ;


    using Regulator = DiFfRG::PolynomialExpRegulator<>;

    Integrator_fT_p2_1ang<4, DiFfRG::MultiValue<double, 2>, Mesons_kernel<Regulator>, DiFfRG::GPU_exec> integrator;

    DiFfRG::GPU_exec map(const std::array<double*, 2>& dest, const BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>& coordinates, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZPi, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZSig)
    ;

    template<typename C, typename ...T>
    DiFfRG::GPU_exec map(const std::array<double*, 2>& dest, const C& coordinates, const device::tuple<T...>& args)
    {
      return device::apply([&](const auto...t){return map(dest, coordinates, t...);}, args);
    }

    void get(DiFfRG::MultiValue<double, 2>& dest, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZPi, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZSig)
    ;

    template<typename IT, typename ...T>
    void get(IT& dest, const double& p0, const double& p, const device::tuple<T...>& args)
    {
      device::apply([&](const auto...t){get(dest, p0, p, t...);}, args);
    }

    static constexpr std::array<const char*, 2> output_names{{"ZPi", "ZSig"}};
    private:
    DiFfRG::QuadratureProvider& quadrature_provider;
  };
}
using DiFfRG::Mesons_integrator;