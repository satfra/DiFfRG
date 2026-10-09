#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DiFfRG/common/config_tree.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/discretization/mesh/rectangular_mesh.hh"
#include "DiFfRG/model/model.hh"

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <vector>
#include <DiFfRG/common/init.hh>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;

  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;

  class QuadraticInitialConditionModel : public def::AbstractModel<QuadraticInitialConditionModel, Components>
  {
  public:
    template <typename Vector> void initial_condition(const Point<1> &pos, Vector &values) const
    {
      values[0] = pos[0] * pos[0];
    }
  };

  void ensure_logger()
  {
    try {
      auto log = spdlog::stdout_color_mt("log");
      log->set_pattern("log: [%v]");
    } catch (const spdlog::spdlog_ex &) {
    }
  }
} // namespace

TEST_CASE("FV FlowingVariables stores cell averages", "[FV][data]")
{
  DiFfRG::Init();
  ensure_logger();

  ConfigTree json =
      json::value({{"physical", {}},
                   {"discretization",
                    {{"fe_order", 0},
                     {"overintegration", 0},
                     {"output_subdivisions", 1},
                     {"EoM_abs_tol", 1e-10},
                     {"EoM_max_iter", 0},
                     {"grid", {{"x_grid", "0:1:2"}, {"y_grid", "0:1:1"}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
                   {"output", {{"verbosity", 0}, {"vtk", false}}}});

  using Mesh = RectangularMesh<1>;
  using Discretization = FV::Discretization<Components, Mesh>;

  QuadraticInitialConditionModel model;
  const Config::GridAxis x_axis(0.0, 1.0, 2.0);
  Mesh mesh(Config::ConfigurationMesh<1>(0u, std::vector<Config::GridAxis>{x_axis}));
  Discretization discretization(mesh, json);
  FV::FlowingVariables<Discretization> state(discretization);
  state.interpolate(model);

  const auto &support_points = discretization.get_support_points();
  const auto &values = state.spatial_data();
  REQUIRE(values.size() == 2);
  REQUIRE(support_points.size() == 2);

  for (unsigned int i = 0; i < values.size(); ++i) {
    const double center = support_points[i][0];
    if (center < 1.0) {
      CHECK(values[i] == Catch::Approx(1.0 / 3.0).margin(1.0e-14));
      CHECK(values[i] != Catch::Approx(center * center).margin(1.0e-3));
    } else {
      CHECK(values[i] == Catch::Approx(7.0 / 3.0).margin(1.0e-14));
      CHECK(values[i] != Catch::Approx(center * center).margin(1.0e-3));
    }
  }
}
