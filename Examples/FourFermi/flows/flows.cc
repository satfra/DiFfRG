#include "flows.hh"

FourFermiFlowEquations::FourFermiFlowEquations(const ConfigTree &json)
    : FlowEquationsFiniteT(
          json, json.get_double("/physical/T"),
          [&](double q2) {
            return 1. / (q2 + __REGULATOR__::RB(powr<2>(k), q2)) * __REGULATOR__::RBdot(powr<2>(k), q2);
          },
          [&](double q0) {
            return 1. / (powr<2>(q0) + __REGULATOR__::RB(powr<2>(k), powr<2>(q0))) *
                   __REGULATOR__::RBdot(powr<2>(k), powr<2>(q0));
          },
          [&](double q0) { return 1. / powr<2>(powr<2>(q0) + powr<2>(k)); }),
      grid_size_int{{x_quadrature_order}}, grid_sizes_int_fT{{x_quadrature_order, x0_quadrature_order}},
      grid_sizes_angle_int{{x_quadrature_order, 2 * angle_quadrature_order, x0_quadrature_order}},
      grid_sizes_4D_int{{x_quadrature_order, angle_quadrature_order, angle_quadrature_order, x0_quadrature_order}},
      lambda1_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json),
      lambda2_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json),
      lambda3_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json),
      lambda4_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json),
      lambda5_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json),
      lambda6_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json),
      lambda7_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json),
      lambda8_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json),
      lambda9_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json),
      lambda10_integrator(quadrature_provider, grid_sizes_int_fT, x_extent, q0_extent, x0_summands, json)
{
}
