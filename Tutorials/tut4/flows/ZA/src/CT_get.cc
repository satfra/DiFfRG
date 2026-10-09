#include "../kernel.hh"

#include "../ZA.hh"

void ZA_integrator::get(double &dest, const double &p, const double &k,
                        const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA3,
                        const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZAcbc,
                        const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA4,
                        const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZc,
                        const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &Zc,
                        const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &dtZA,
                        const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>> &ZA)
{
  integrator.get(dest, p, k, ZA3, ZAcbc, ZA4, dtZc, Zc, dtZA, ZA);
}