#include "../kernel.hh"

#include "../ZA4.hh"

DiFfRG::GPU_exec
ZA4_integrator::map(double *dest, const LogarithmicCoordinates1D<double> &coordinates, const double &k,
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