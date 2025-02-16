#include "flows.hh"

QuarkMesonFlowEquations::QuarkMesonFlowEquations(const JSONValue &json)
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

      grid_size_int{{x_quadrature_order}}, grid_sizes_angle_int{{x_quadrature_order, angle_quadrature_order}},
      grid_sizes_3D_int{{x_quadrature_order, angle_quadrature_order, angle_quadrature_order}},
      grid_sizes_4D_int{{x_quadrature_order, angle_quadrature_order, angle_quadrature_order, angle_quadrature_order}},

      grid_sizes_int_fT{{x_quadrature_order, x0_quadrature_order}},
      grid_sizes_angle_int_fT{{x_quadrature_order, angle_quadrature_order, x0_quadrature_order}},
      grid_sizes_4D_int_fT{{x_quadrature_order, angle_quadrature_order, angle_quadrature_order, x0_quadrature_order}},

      grid_sizes_2D_cartesian_int{{x_quadrature_order, x_quadrature_order}},
      grid_sizes_3D_cartesian_int{{x_quadrature_order, x_quadrature_order, x_quadrature_order}},

      quadrature_provider(json), V_integrator(quadrature_provider, grid_size_int, x_extent, json),
      etaQ_integrator(quadrature_provider, grid_sizes_angle_int, x_extent, json),
      etaPhi_integrator(quadrature_provider, grid_sizes_angle_int, x_extent, json),
      hPhi0_integrator(quadrature_provider, grid_sizes_angle_int, x_extent, json)
{
}
