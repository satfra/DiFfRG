#include "../kernel.hh"

#include "../V.hh"

void V_integrator::get(double &dest, const double &k, const double &N, const double &T, const double &m2Pi, const double &m2Sigma)
{
  integrator.get(dest, k, N, T, m2Pi, m2Sigma);
}