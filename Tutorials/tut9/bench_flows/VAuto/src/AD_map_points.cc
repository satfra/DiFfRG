#include "../kernel.hh"

#include "../VAuto.hh"

void VAuto_integrator::map_points(const DiFfRG::PointSpan<autodiff::real> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& hPhi, const DiFfRG::PointArg<double>& etaPi0, const DiFfRG::PointArg<double>& rhoPhi, const DiFfRG::PointArg<autodiff::real>& d1V, const DiFfRG::PointArg<autodiff::real>& d2V, const DiFfRG::PointArg<autodiff::real>& d3V, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZPi, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZSig)
{
  integrator_AD.map_points(dest,  k, T, hPhi, etaPi0, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
}
void VAuto_integrator::map_points(const DiFfRG::PointSpan<autodiff::Real<2, double>> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& hPhi, const DiFfRG::PointArg<double>& etaPi0, const DiFfRG::PointArg<double>& rhoPhi, const DiFfRG::PointArg<autodiff::Real<2, double>>& d1V, const DiFfRG::PointArg<autodiff::Real<2, double>>& d2V, const DiFfRG::PointArg<autodiff::Real<2, double>>& d3V, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZPi, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZSig)
{
  integrator_AD2.map_points(dest,  k, T, hPhi, etaPi0, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
}