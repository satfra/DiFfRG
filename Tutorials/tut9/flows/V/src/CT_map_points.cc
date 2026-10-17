#include "../kernel.hh"

#include "../V.hh"

void V_integrator::map_points(const DiFfRG::PointSpan<double> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& hPhi, const DiFfRG::PointArg<double>& etaPi0, const DiFfRG::PointArg<double>& rhoPhi, const DiFfRG::PointArg<double>& d1V, const DiFfRG::PointArg<double>& d2V, const DiFfRG::PointArg<double>& d3V, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZPi, const DiFfRG::PointArg<SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>>& ZSig)
{
  integrator.map_points(dest,  k, T, hPhi, etaPi0, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
}