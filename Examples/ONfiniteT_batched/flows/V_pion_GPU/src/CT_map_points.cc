#include "../kernel.hh"

#include "../V_pion_GPU.hh"

void V_pion_GPU_integrator::map_points(const DiFfRG::PointSpan<double> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<double>& m2Pi)
{
  integrator.map_points(dest,  k, N, T, m2Pi);
}