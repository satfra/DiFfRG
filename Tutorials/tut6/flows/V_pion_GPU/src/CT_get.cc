#include "../kernel.hh"

#include "../V_pion_GPU.hh"

void V_pion_GPU_integrator::get(double& dest, const double& k, const double& N, const double& T, const double& m2Pi)
{
  integrator.get(dest,  k, N, T, m2Pi);
}