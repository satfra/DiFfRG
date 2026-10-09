// clang-format off
#include "../kernel.hh"

#include "../ZA4tadpole.hh"

DiFfRG::GPU_exec ZA4tadpole_integrator::map(double *dest, const LogLinLinPeriodicCoordinates &coordinates, const double &k,
const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA3,
const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZAcbc,
const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA4SP,
const LinearInterpolatorND<double, LogLinLinPeriodicCoordinates, GPU_memory> &ZA4tadpole,
const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZc,
const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &Zc,
const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &dtZA,
const SplineInterpolator1D<double, LogarithmicCoordinates1D<double>, GPU_memory> &ZA)
{
return integrator.map(dest, coordinates, k, ZA3, ZAcbc, ZA4SP, ZA4tadpole, dtZc, Zc, dtZA, ZA);
// clang-format on
}