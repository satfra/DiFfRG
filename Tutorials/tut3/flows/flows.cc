#include "./flows.hh"

ONFiniteTFlows::ONFiniteTFlows(const DiFfRG::ConfigTree &json) : quadrature_provider(json), V(quadrature_provider, json) {}
void ONFiniteTFlows::set_k(const double k) { DiFfRG::all_set_k(V, k); }
void ONFiniteTFlows::set_T(const double T) { DiFfRG::all_set_T(V, T); }
void ONFiniteTFlows::set_typical_E(const double E) { DiFfRG::all_set_typical_E(V, E); }
void ONFiniteTFlows::set_x_extent(const double x_extent) { DiFfRG::all_set_x_extent(V, x_extent); }