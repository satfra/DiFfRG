#include "../kernel.hh"

#include "../etaQ.hh"

 void etaQ_integrator::get(autodiff::real& dest, const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const autodiff::real& etaQ, const autodiff::real& etaPhi, const autodiff::real& hPhi, const autodiff::real& d1V, const autodiff::real& d2V, const autodiff::real& d3V, const double& rhoPhi)
{
integrator_AD.get(dest,  k, p, p0f, Nf, Nc, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
}
