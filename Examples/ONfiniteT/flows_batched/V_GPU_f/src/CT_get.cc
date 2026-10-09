#include "../kernel.hh"

#include "../V_GPU_f.hh"

void V_GPU_f_integrator::get(double& dest, const double& k, const double& N, const double& T, const double& m2Pi, const double& m2Sigma)
{
  integrator.get(dest,  k, N, T, m2Pi, m2Sigma);
}