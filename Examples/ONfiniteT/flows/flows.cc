#include "./flows.hh"

ONFiniteTFlows::ONFiniteTFlows(const DiFfRG::ConfigTree& config) : quadrature_provider(config), V(quadrature_provider, config), V_pion(quadrature_provider, config), V_sigma(quadrature_provider, config)
{
}
void ONFiniteTFlows::set_k(const double k)
{
  DiFfRG::all_set_k(V, k);DiFfRG::all_set_k(V_pion, k);DiFfRG::all_set_k(V_sigma, k);
}
void ONFiniteTFlows::set_T(const double T)
{
  DiFfRG::all_set_T(V, T);DiFfRG::all_set_T(V_pion, T);DiFfRG::all_set_T(V_sigma, T);
}
void ONFiniteTFlows::set_typical_E(const double E)
{
  DiFfRG::all_set_typical_E(V, E);DiFfRG::all_set_typical_E(V_pion, E);DiFfRG::all_set_typical_E(V_sigma, E);
}
void ONFiniteTFlows::set_x_extent(const double x_extent)
{
  DiFfRG::all_set_x_extent(V, x_extent);DiFfRG::all_set_x_extent(V_pion, x_extent);DiFfRG::all_set_x_extent(V_sigma, x_extent);
}