#include "../kernel.hh"

#include "../Zc.hh"

DiFfRG::GPU_exec
Zc_integrator::map(double *dest, const LogarithmicCoordinates1D<double> &coordinates, const double &k,
                   const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA3,
                   const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZAcbc,
                   const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA4,
                   const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZc,
                   const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &Zc,
                   const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZA,
                   const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA)
{
  return integrator.map(dest, coordinates, k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);
}