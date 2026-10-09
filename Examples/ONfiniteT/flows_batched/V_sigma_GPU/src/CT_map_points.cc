#include "../kernel.hh"

#include "../V_sigma_GPU.hh"

void V_sigma_GPU_integrator::map_points(const DiFfRG::PointSpan<double> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& m2Sigma)
{
  integrator.map_points(dest,  k, N, T, m2Sigma);
}