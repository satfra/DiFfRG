#include "../kernel.hh"

#include "../ZA3.hh"

DiFfRG::GPU_exec
ZA3_integrator::map(double *dest, const FocusedLogLinLinPeriodicCoordinates &coordinates, const double &k,
                    const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA3,
                    const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZAcbc,
                    const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA4SP,
                    const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA4tadpole,
                    const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZc,
                    const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &Zc,
                    const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZA,
                    const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA)
{
  return integrator.map(dest, coordinates, k, ZA3, ZAcbc, ZA4SP, ZA4tadpole, dtZc, Zc, dtZA, ZA);
}