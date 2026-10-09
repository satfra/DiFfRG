#include "../kernel.hh"

#include "../V_pion.hh"

void V_pion_integrator::get(autodiff::real& dest, const double& k, const double& N, const double& T, const autodiff::real& m2Pi)
{
  integrator_AD.get(dest,  k, N, T, m2Pi);
}
void V_pion_integrator::get(autodiff::Real<2, double>& dest, const double& k, const double& N, const double& T, const autodiff::Real<2, double>& m2Pi)
{
  integrator_AD2.get(dest,  k, N, T, m2Pi);
}