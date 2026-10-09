#include "./flows.hh"

YangMillsFlows::YangMillsFlows(const DiFfRG::ConfigTree &json)
    : quadrature_provider(json), ZA4(quadrature_provider, json)
{
}
void YangMillsFlows::set_k(const double k) { DiFfRG::all_set_k(ZA4, k); }
void YangMillsFlows::set_T(const double T) { DiFfRG::all_set_T(ZA4, T); }
void YangMillsFlows::set_typical_E(const double E) { DiFfRG::all_set_typical_E(ZA4, E); }
void YangMillsFlows::set_x_extent(const double x_extent) { DiFfRG::all_set_x_extent(ZA4, x_extent); }