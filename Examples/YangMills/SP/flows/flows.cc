#include "./flows.hh"

YangMillsFlows::YangMillsFlows(const DiFfRG::ConfigTree &config)
    : quadrature_provider(config), ZA(quadrature_provider, config), ZA3(quadrature_provider, config),
      ZA4(quadrature_provider, config), ZAcbc(quadrature_provider, config), Zc(quadrature_provider, config)
{
}
void YangMillsFlows::set_k(const double k)
{
  DiFfRG::all_set_k(ZA, k);
  DiFfRG::all_set_k(ZA3, k);
  DiFfRG::all_set_k(ZA4, k);
  DiFfRG::all_set_k(ZAcbc, k);
  DiFfRG::all_set_k(Zc, k);
}
void YangMillsFlows::set_T(const double T)
{
  DiFfRG::all_set_T(ZA, T);
  DiFfRG::all_set_T(ZA3, T);
  DiFfRG::all_set_T(ZA4, T);
  DiFfRG::all_set_T(ZAcbc, T);
  DiFfRG::all_set_T(Zc, T);
}
void YangMillsFlows::set_typical_E(const double E)
{
  DiFfRG::all_set_typical_E(ZA, E);
  DiFfRG::all_set_typical_E(ZA3, E);
  DiFfRG::all_set_typical_E(ZA4, E);
  DiFfRG::all_set_typical_E(ZAcbc, E);
  DiFfRG::all_set_typical_E(Zc, E);
}
void YangMillsFlows::set_x_extent(const double x_extent)
{
  DiFfRG::all_set_x_extent(ZA, x_extent);
  DiFfRG::all_set_x_extent(ZA3, x_extent);
  DiFfRG::all_set_x_extent(ZA4, x_extent);
  DiFfRG::all_set_x_extent(ZAcbc, x_extent);
  DiFfRG::all_set_x_extent(Zc, x_extent);
}