#include "../kernel.hh"

#include "../MesonsPlain.hh"

DiFfRG::GPU_exec MesonsPlain_integrator::map(const std::array<double*, 2>& dest, const BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>& coordinates, const double& k, const double& T, const double& hPhi, const double& etaPi0, const double& rhoPhi, const double& d1V, const double& d2V, const double& d3V, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZPi, const SplineInterpolator1DStack<double, BosonicCoordinates1DFiniteT<int, double, LogarithmicCoordinates1D<double>>>& ZSig)
{
  #if !defined(NDEBUG) || defined(DIFFRG_CHECK_SHARED_GRIDS)
  DiFfRG::check_shared_grid("ZPi, ZSig", ZPi, ZSig);
  #endif
  return integrator.map(dest, coordinates, k, T, hPhi, etaPi0, rhoPhi, d1V, d2V, d3V, ZPi, ZSig);
}