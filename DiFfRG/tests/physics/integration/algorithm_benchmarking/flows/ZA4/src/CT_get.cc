#include "../kernel.hh"

#include "../ZA4.hh"

void ZA4_integrator::get(double &dest, const double &p, const double &k,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &ZA3,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &ZAcbc,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &ZA4,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &dtZc,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &Zc,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &dtZA,
                         const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, MemorySpace> &ZA)
{
  integrator.get(dest, p, k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);
}