#include "../kernel.hh"

#include "../ZPi.hh"

void ZPi_integrator::get(double& dest, const double& p0, const double& p, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZPi, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZSig)
{
  integrator.get(dest, p0, p,  k, T, hPhi, etaPi0, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
}