#include "./flows.hh"

YangMillsFlows::YangMillsFlows(const DiFfRG::ConfigTree &json)
    : quadrature_provider(json), ZA(quadrature_provider, json), ZA3(quadrature_provider, json), ZA4SP(quadrature_provider, json),
      ZA4tadpole(quadrature_provider, json), ZAcbc(quadrature_provider, json), Zc(quadrature_provider, json)
{
}
void YangMillsFlows::set_k(const double k)
{
  DiFfRG::all_set_k(ZA, k);
  DiFfRG::all_set_k(ZA3, k);
  DiFfRG::all_set_k(ZA4SP, k);
  DiFfRG::all_set_k(ZA4tadpole, k);
  DiFfRG::all_set_k(ZAcbc, k);
  DiFfRG::all_set_k(Zc, k);
}
void YangMillsFlows::set_T(const double T)
{
  DiFfRG::all_set_T(ZA, T);
  DiFfRG::all_set_T(ZA3, T);
  DiFfRG::all_set_T(ZA4SP, T);
  DiFfRG::all_set_T(ZA4tadpole, T);
  DiFfRG::all_set_T(ZAcbc, T);
  DiFfRG::all_set_T(Zc, T);
}
void YangMillsFlows::set_typical_E(const double E)
{
  DiFfRG::all_set_typical_E(ZA, E);
  DiFfRG::all_set_typical_E(ZA3, E);
  DiFfRG::all_set_typical_E(ZA4SP, E);
  DiFfRG::all_set_typical_E(ZA4tadpole, E);
  DiFfRG::all_set_typical_E(ZAcbc, E);
  DiFfRG::all_set_typical_E(Zc, E);
}
void YangMillsFlows::set_x_extent(const double x_extent)
{
  DiFfRG::all_set_x_extent(ZA, x_extent);
  DiFfRG::all_set_x_extent(ZA3, x_extent);
  DiFfRG::all_set_x_extent(ZA4SP, x_extent);
  DiFfRG::all_set_x_extent(ZA4tadpole, x_extent);
  DiFfRG::all_set_x_extent(ZAcbc, x_extent);
  DiFfRG::all_set_x_extent(Zc, x_extent);
}