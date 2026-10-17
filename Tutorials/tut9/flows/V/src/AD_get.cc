#include "../kernel.hh"

#include "../V.hh"

void V_integrator::get(autodiff::real& dest, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const autodiff::real& d1V, const autodiff::real& d2V, const autodiff::real& d3V, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZPi, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZSig)
{
  integrator_AD.get(dest,  k, T, hPhi, etaPi0, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
}
void V_integrator::get(autodiff::Real<2, double>& dest, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const autodiff::Real<2, double>& d1V, const autodiff::Real<2, double>& d2V, const autodiff::Real<2, double>& d3V, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZPi, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZSig)
{
  integrator_AD2.get(dest,  k, T, hPhi, etaPi0, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
}