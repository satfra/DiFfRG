#include "../kernel.hh"

#include "../Zc.hh"

void Zc_integrator::get(double &dest, const double &p, const double &k,
                        const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA3,
                        const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZAcbc,
                        const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA4SP,
                        const LinearInterpolatorND<double, FocusedLogLinLinPeriodicCoordinates> &ZA4tadpole,
                        const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZc,
                        const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &Zc,
                        const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &dtZA,
                        const SplineInterpolator1D<double, FocusedLogCoordinates1D<double>> &ZA)
{
  integrator.get(dest, p, k, ZA3, ZAcbc, ZA4SP, ZA4tadpole, dtZc, Zc, dtZA, ZA);
}