#include "flows.hh"

YangMillsFlowEquations::YangMillsFlowEquations(const JSONValue &json)
    : FlowEquations(json,
                    [](double x) { return powr<-1>(x + __REGULATOR__::RB(1., x)) * __REGULATOR__::RBdot(1., x); }),
      grid_size_int{{x_quadrature_order}}, grid_sizes_angle_int{{x_quadrature_order, 2 * angle_quadrature_order}},
      grid_sizes_3D_int{{x_quadrature_order, angle_quadrature_order, angle_quadrature_order}},
      grid_sizes_4D_int{{x_quadrature_order, angle_quadrature_order, angle_quadrature_order, angle_quadrature_order}},
      grid_sizes_2D_cartesian_int{{x_quadrature_order, x_quadrature_order}},
      grid_sizes_3D_cartesian_int{{x_quadrature_order, x_quadrature_order, x_quadrature_order}},
      ZA_integrator(quadrature_provider, grid_sizes_angle_int, x_extent, json),
      m2A_integrator(quadrature_provider, grid_sizes_angle_int, x_extent, json),
      Zc_integrator(quadrature_provider, grid_sizes_angle_int, x_extent, json),
      ZA3_integrator(quadrature_provider, grid_sizes_3D_int, x_extent, json),
      ZAcbc_integrator(quadrature_provider, grid_sizes_3D_int, x_extent, json),
      ZA4tadpole_integrator(quadrature_provider, grid_sizes_3D_int, x_extent, json),
      ZA4SP_integrator(quadrature_provider, grid_sizes_4D_int, x_extent, json)
{
}
