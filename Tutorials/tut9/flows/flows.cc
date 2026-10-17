#include "./flows.hh"

QMFlows::QMFlows(const DiFfRG::ConfigTree& config) : quadrature_provider(config), Mesons(quadrature_provider, config), V(quadrature_provider, config)
{
}
void QMFlows::set_k(const double k)
{
  DiFfRG::all_set_k(Mesons, k);DiFfRG::all_set_k(V, k);
}
void QMFlows::set_T(const double T)
{
  DiFfRG::all_set_T(Mesons, T);DiFfRG::all_set_T(V, T);
}
void QMFlows::set_typical_E(const double E)
{
  DiFfRG::all_set_typical_E(Mesons, E);DiFfRG::all_set_typical_E(V, E);
}
void QMFlows::set_x_extent(const double x_extent)
{
  DiFfRG::all_set_x_extent(Mesons, x_extent);DiFfRG::all_set_x_extent(V, x_extent);
}