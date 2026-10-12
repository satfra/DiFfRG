#include "../kernel.hh"

#include "../V_GPU.hh"

void V_GPU_integrator::get(autodiff::real& dest, const double& k, const double& N, const double& T, const autodiff::real& m2Pi, const autodiff::real& m2Sigma)
{
  integrator_AD.get(dest,  k, N, T, m2Pi, m2Sigma);
}
void V_GPU_integrator::get(autodiff::Real<2, double>& dest, const double& k, const double& N, const double& T, const autodiff::Real<2, double>& m2Pi, const autodiff::Real<2, double>& m2Sigma)
{
  integrator_AD2.get(dest,  k, N, T, m2Pi, m2Sigma);
}