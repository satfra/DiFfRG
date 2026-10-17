#include "./flows.hh"

QMBenchFlows::QMBenchFlows(const DiFfRG::ConfigTree& config) : quadrature_provider(config), MesonsPlain(quadrature_provider, config), VAuto(quadrature_provider, config), ZPi(quadrature_provider, config), ZSig(quadrature_provider, config)
{
}
void QMBenchFlows::set_k(const double k)
{
  DiFfRG::all_set_k(MesonsPlain, k);DiFfRG::all_set_k(VAuto, k);DiFfRG::all_set_k(ZPi, k);DiFfRG::all_set_k(ZSig, k);
}
void QMBenchFlows::set_T(const double T)
{
  DiFfRG::all_set_T(MesonsPlain, T);DiFfRG::all_set_T(VAuto, T);DiFfRG::all_set_T(ZPi, T);DiFfRG::all_set_T(ZSig, T);
}
void QMBenchFlows::set_typical_E(const double E)
{
  DiFfRG::all_set_typical_E(MesonsPlain, E);DiFfRG::all_set_typical_E(VAuto, E);DiFfRG::all_set_typical_E(ZPi, E);DiFfRG::all_set_typical_E(ZSig, E);
}
void QMBenchFlows::set_x_extent(const double x_extent)
{
  DiFfRG::all_set_x_extent(MesonsPlain, x_extent);DiFfRG::all_set_x_extent(VAuto, x_extent);DiFfRG::all_set_x_extent(ZPi, x_extent);DiFfRG::all_set_x_extent(ZSig, x_extent);
}