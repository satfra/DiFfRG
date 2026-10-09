#include "../kernel.hh"

#include "../V_pion.hh"

void V_pion_integrator::map_points(const DiFfRG::PointSpan<autodiff::real> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<autodiff::real>& m2Pi)
{
  integrator_AD.map_points(dest,  k, N, T, m2Pi);
}
void V_pion_integrator::map_points(const DiFfRG::PointSpan<autodiff::Real<2, double>> dest, const DiFfRG::PointArg<double>& k, const DiFfRG::PointArg<double>& N, const DiFfRG::PointArg<double>& T, const DiFfRG::PointArg<autodiff::Real<2, double>>& m2Pi)
{
  integrator_AD2.map_points(dest,  k, N, T, m2Pi);
}