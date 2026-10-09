#include "./flows.hh"

  LPA_QM_flows::LPA_QM_flows(const DiFfRG::ConfigTree& json) : quadrature_provider(json), etaPhi(quadrature_provider, json), etaQ(quadrature_provider, json), hPhi0(quadrature_provider, json), V(quadrature_provider, json) 
{}
 void LPA_QM_flows::set_k(const double k)
{
DiFfRG::all_set_k(etaPhi, k);DiFfRG::all_set_k(etaQ, k);DiFfRG::all_set_k(hPhi0, k);DiFfRG::all_set_k(V, k);
}
 void LPA_QM_flows::set_T(const double T)
{
DiFfRG::all_set_T(etaPhi, T);DiFfRG::all_set_T(etaQ, T);DiFfRG::all_set_T(hPhi0, T);DiFfRG::all_set_T(V, T);
}
 void LPA_QM_flows::set_typical_E(const double E)
{
DiFfRG::all_set_typical_E(etaPhi, E);DiFfRG::all_set_typical_E(etaQ, E);DiFfRG::all_set_typical_E(hPhi0, E);DiFfRG::all_set_typical_E(V, E);
}
 void LPA_QM_flows::set_x_extent(const double x_extent)
{
DiFfRG::all_set_x_extent(etaPhi, x_extent);DiFfRG::all_set_x_extent(etaQ, x_extent);DiFfRG::all_set_x_extent(hPhi0, x_extent);DiFfRG::all_set_x_extent(V, x_extent);
}
