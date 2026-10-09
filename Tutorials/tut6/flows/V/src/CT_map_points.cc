#include "../kernel.hh"

#include "../V.hh"

void V_integrator::map_points(const DiFfRG::PointSpan<double> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& m2Pi, const DiFfRG::PointArg<double>& m2Sigma)
{
  integrator.map_points(dest,  k, N, T, m2Pi, m2Sigma);
}