#include "../kernel.hh"

#include "../etaPhi.hh"

 void etaPhi_integrator::get(double& dest, const double& k, const double& p, const double& p0f, const double& Nf, const double& Nc, const double& T, const double& muq, const double& etaQ, const double& etaPhi, const double& hPhi, const double& d1V, const double& d2V, const double& d3V, const double& rhoPhi)
{
integrator.get(dest,  k, p, p0f, Nf, Nc, T, muq, etaQ, etaPhi, hPhi, d1V, d2V, d3V, rhoPhi);
}
