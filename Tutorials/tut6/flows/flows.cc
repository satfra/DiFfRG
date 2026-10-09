#include "./flows.hh"

ONFlows::ONFlows(const DiFfRG::ConfigTree& config) : quadrature_provider(config), V(quadrature_provider, config), V_GPU(quadrature_provider, config), V_pion(quadrature_provider, config), V_pion_GPU(quadrature_provider, config), V_sigma(quadrature_provider, config), V_sigma_GPU(quadrature_provider, config)
{
}
void ONFlows::set_k(const double k)
{
  DiFfRG::all_set_k(V, k);DiFfRG::all_set_k(V_GPU, k);DiFfRG::all_set_k(V_pion, k);DiFfRG::all_set_k(V_pion_GPU, k);DiFfRG::all_set_k(V_sigma, k);DiFfRG::all_set_k(V_sigma_GPU, k);
}
void ONFlows::set_T(const double T)
{
  DiFfRG::all_set_T(V, T);DiFfRG::all_set_T(V_GPU, T);DiFfRG::all_set_T(V_pion, T);DiFfRG::all_set_T(V_pion_GPU, T);DiFfRG::all_set_T(V_sigma, T);DiFfRG::all_set_T(V_sigma_GPU, T);
}
void ONFlows::set_typical_E(const double E)
{
  DiFfRG::all_set_typical_E(V, E);DiFfRG::all_set_typical_E(V_GPU, E);DiFfRG::all_set_typical_E(V_pion, E);DiFfRG::all_set_typical_E(V_pion_GPU, E);DiFfRG::all_set_typical_E(V_sigma, E);DiFfRG::all_set_typical_E(V_sigma_GPU, E);
}
void ONFlows::set_x_extent(const double x_extent)
{
  DiFfRG::all_set_x_extent(V, x_extent);DiFfRG::all_set_x_extent(V_GPU, x_extent);DiFfRG::all_set_x_extent(V_pion, x_extent);DiFfRG::all_set_x_extent(V_pion_GPU, x_extent);DiFfRG::all_set_x_extent(V_sigma, x_extent);DiFfRG::all_set_x_extent(V_sigma_GPU, x_extent);
}