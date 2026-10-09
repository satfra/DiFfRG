#include "../kernel.hh"

#include "../V_sigma.hh"

void V_sigma_integrator::get(double& dest, const double& k, const double& N, const double& T, const double& m2Sigma)
{
  integrator.get(dest,  k, N, T, m2Sigma);
}