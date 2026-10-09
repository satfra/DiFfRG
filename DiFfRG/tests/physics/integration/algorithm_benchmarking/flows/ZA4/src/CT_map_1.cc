#include "../kernel.hh"

#include "../ZA4.hh"

DiFfRG::GPU_exec
ZA4_integrator::map(double *dest, const LogarithmicCoordinates1D<double> &coordinates, const double &k,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &ZA3,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &ZAcbc,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &ZA4,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &dtZc,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &Zc,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &dtZA,
                    const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &ZA)
{
  return integrator.map(dest, coordinates, k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);
}