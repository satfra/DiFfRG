// Tests for Mesh Configuration

#include "DiFfRG/common/config_tree.hh"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "catch2/generators/catch_generators_adapters.hpp"
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/mesh/configuration_mesh.hh>
#include <catch2/catch_all.hpp>
#include <deal.II/base/point.h>

using namespace DiFfRG::Config;

// // Test case for default constructor
TEST_CASE("Mesh Configuration exists", "[MeshConfiguration]")
{
  SECTION("initializes correctly the mesh config")
  {

    // triangulation.refine_global(json.get_uint("/discretization/grid/refine"));
    DiFfRG::ConfigTree json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1,1:0.2:2"}, {"z_grid", "0:0.1:1"}}},
         }},
    });
    ConfigurationMesh<3> mesh_config(json);
    REQUIRE(size(mesh_config.grids[0]) == 1);
    REQUIRE(size(mesh_config.grids[1]) == 2);
    REQUIRE(size(mesh_config.grids[2]) == 1);
  }

  SECTION("refine parameter works correctly")
  {

    DiFfRG::ConfigTree json_with_refine = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1"}, {"refine", 1}}},
         }},
    });
    ConfigurationMesh<1> mesh_config_with_refine(json_with_refine);
    CHECK(mesh_config_with_refine.refine == 1);
    DiFfRG::ConfigTree json_without_refine = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1"}}},
         }},
    });
    ConfigurationMesh<1> mesh_config_without_refine(json_without_refine);
    CHECK(mesh_config_without_refine.refine == 0);
  }

  SECTION("works only with x_axis")
  {
    DiFfRG::ConfigTree json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1"}}},
         }},
    });
    ConfigurationMesh<1> mesh_config(json);
    REQUIRE(size(mesh_config.grids[0]) == 1);
  }
  SECTION("check range consistency")
  {
    DiFfRG::ConfigTree json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.1:1,1.1:0.2:2"}}},
         }},
    });
    CHECK_THROWS(ConfigurationMesh<1>(json));
  }
  SECTION("check a correct step size array")
  {
    DiFfRG::ConfigTree json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.5:1,1:0.6:2"}}},
         }},
    });
    ConfigurationMesh<1> mesh_config(json);
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[0]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[0] == std::vector<double>{0.5, 0.5, 0.5, 0.5});
  }
  SECTION("check a correct step size array in 2D")
  {
    DiFfRG::ConfigTree json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.5:1,1:0.6:2"}, {"y_grid", "0:1.0:2,2:1.0:4"}}},
         }},
    });
    ConfigurationMesh<2> mesh_config(json);
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[0]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[0] == std::vector<double>{0.5, 0.5, 0.5, 0.5});
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[1]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[1] == std::vector<double>(4, 1.0));
  }
  SECTION("check a correct step size array in 3D")
  {
    DiFfRG::ConfigTree json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid", {{"x_grid", "0:0.5:1,1:0.6:2"}, {"y_grid", "0:1.0:2,2:1.0:4"}, {"z_grid", "0:2.0:2,2:2.0:4"}}},
         }},
    });
    ConfigurationMesh<3> mesh_config(json);
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[0]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[0] == std::vector<double>{0.5, 0.5, 0.5, 0.5});
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[1]) == 4);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[1] == std::vector<double>(4, 1.0));
    REQUIRE(size(mesh_config.get_step_withs_for_triangulation()[2]) == 2);
    REQUIRE(mesh_config.get_step_withs_for_triangulation()[2] == std::vector<double>(2, 2.0));
  }

  SECTION("Get lower_left and upper_right of grid in 3D", "[MeshConfiguration]")
  {
    using dealii::Point;
    DiFfRG::ConfigTree json = DiFfRG::json::value({
        {"discretization",
         {
             {"grid",
              {{"x_grid", "0:0.5:1,1:0.6:2"}, {"y_grid", "0.1:1.0:2,2:1.0:4"}, {"z_grid", "0.2:0.2:2,2:2.0:4"}}},
         }},
    });
    ConfigurationMesh<3> mesh_config(json);
    CHECK(mesh_config.get_lower_left() == Point<3, double>(0.0, 0.1, 0.2));
    CHECK(mesh_config.get_upper_right() == Point<3, double>(2.0, 4.0, 4.0));

    DiFfRG::ConfigTree json2 = DiFfRG::json::value({
        {"discretization",
         {
             {"grid",
              {{"x_grid", "0.2:0.5:1,1:0.6:3"}, {"y_grid", "0.3:1.0:2,2:1.0:6"}, {"z_grid", "0.4:0.2:2,2:2.0:8"}}},
         }},
    });
    ConfigurationMesh<3> mesh_config2(json2);
    CHECK(mesh_config2.get_lower_left() == Point<3, double>(0.2, 0.3, 0.4));
    CHECK(mesh_config2.get_upper_right() == Point<3, double>(3.0, 6.0, 8.0));
  }
}

TEST_CASE("Mesh Configuration computes origin-centered triangulation data", "[MeshConfiguration][origin-centered]")
{
  using Catch::Matchers::WithinAbs;

  SECTION("keeps default triangulation data unchanged")
  {
    const std::vector<GridAxis> x_grid{GridAxis(0.0, 0.1, 5.0)};
    const ConfigurationMesh<1> mesh_config(0u, x_grid);
    const auto data = mesh_config.get_triangulation_data();

    CHECK(data.lower_left == mesh_config.get_lower_left());
    CHECK(data.upper_right == mesh_config.get_upper_right());
    CHECK(data.step_sizes == mesh_config.get_step_withs_for_triangulation());
  }

  SECTION("adjusts bounds and first subrange widths coherently")
  {
    const std::vector<GridAxis> x_grid{GridAxis(0.0, 0.1, 5.0)};
    const ConfigurationMesh<1> mesh_config(0u, x_grid);
    const auto data = mesh_config.get_triangulation_data(true);

    constexpr double expected_width = 5.0 / 50.5;
    REQUIRE(data.step_sizes[0].size() == 51);
    CHECK_THAT(data.lower_left[0], WithinAbs(-0.5 * expected_width, 1.0e-12));
    CHECK_THAT(data.upper_right[0], WithinAbs(5.0, 1.0e-12));
    for (const double width : data.step_sizes[0])
      CHECK_THAT(width, WithinAbs(expected_width, 1.0e-12));
  }

  SECTION("accounts for refinement and preserves later subranges")
  {
    const std::vector<GridAxis> x_grid{GridAxis(0.0, 0.25, 1.0), GridAxis(1.0, 0.5, 2.0)};
    const ConfigurationMesh<1> mesh_config(1u, x_grid);
    const auto data = mesh_config.get_triangulation_data(true);

    constexpr double expected_first_width = 1.0 / 4.75;
    REQUIRE(data.step_sizes[0].size() == 7);
    CHECK_THAT(data.lower_left[0], WithinAbs(-expected_first_width / 4.0, 1.0e-12));
    CHECK_THAT(data.upper_right[0], WithinAbs(2.0, 1.0e-12));
    for (std::size_t i = 0; i < 5; ++i)
      CHECK_THAT(data.step_sizes[0][i], WithinAbs(expected_first_width, 1.0e-12));
    for (std::size_t i = 5; i < 7; ++i)
      CHECK_THAT(data.step_sizes[0][i], WithinAbs(0.5, 1.0e-12));
  }

  SECTION("rejects axes not starting at zero")
  {
    const std::vector<GridAxis> x_grid{GridAxis(0.1, 0.1, 1.0)};
    const ConfigurationMesh<1> mesh_config(0u, x_grid);

    CHECK_THROWS_WITH(mesh_config.get_triangulation_data(true),
                      "Origin-centered rectangular meshes require every configured axis to start at zero.");
  }
}

TEST_CASE("GridAxis tests", "[MeshConfiguration]")
{
  SECTION("parses range_string \"0:0.1:1\" correctly")
  {
    GridAxis grid_axis("0:0.1:1");
    REQUIRE(grid_axis.min == 0.0);
    REQUIRE(grid_axis.step == 0.1);
    REQUIRE(grid_axis.max == 1.0);
  }

  SECTION("parses range_string \"0.1:0.1:2\" correctly")
  {
    GridAxis grid_axis("0.1:0.1:2");
    REQUIRE(grid_axis.min == 0.1);
    REQUIRE(grid_axis.step == 0.1);
    REQUIRE(grid_axis.max == 2.0);
  }

  SECTION("returns array of steps")
  {
    GridAxis grid_axis("0.0:0.5:2");
    REQUIRE(size(grid_axis.get_stepwiths()) == 4);
    REQUIRE(grid_axis.get_stepwiths() == std::vector<double>{0.5, 0.5, 0.5, 0.5});
  }

  SECTION("returns array of steps if step does not add up exactly to max")
  {
    GridAxis grid_axis("0.0:0.6:2");
    REQUIRE(size(grid_axis.get_stepwiths()) == 4);
    REQUIRE(grid_axis.get_stepwiths() == std::vector<double>{0.5, 0.5, 0.5, 0.5});
  }

  SECTION("returns array of steps if step does not add up exactly to max, with one step")
  {
    GridAxis grid_axis("0:2.0:2");
    REQUIRE(size(grid_axis.get_stepwiths()) == 1);
    REQUIRE(grid_axis.get_stepwiths() == std::vector<double>{2.0});
  }

  SECTION("chooses the smallest origin-centered cell count with a smaller step width")
  {
    const GridAxis grid_axis(0.0, 0.3, 5.0);
    const auto step_widths = grid_axis.get_stepwiths(true);

    constexpr double expected_width = 5.0 / 17.5;
    REQUIRE(step_widths.size() == 18);
    CHECK_THAT(step_widths.front(), Catch::Matchers::WithinAbs(expected_width, 1.0e-12));
    CHECK(step_widths.front() < grid_axis.step);
    CHECK_FALSE(5.0 / 16.5 < grid_axis.step);
  }

  SECTION("accounts for global refinement when choosing the smallest origin-centered cell count")
  {
    const GridAxis grid_axis(0.0, 0.3, 5.0);
    const auto step_widths = grid_axis.get_stepwiths(true, 1u);

    constexpr double expected_width = 5.0 / 16.75;
    REQUIRE(step_widths.size() == 17);
    CHECK_THAT(step_widths.front(), Catch::Matchers::WithinAbs(expected_width, 1.0e-12));
    CHECK(step_widths.front() < grid_axis.step);
  }
}

TEST_CASE("throws when range is invalid", "[MeshConfiguration]")
{
  CHECK_THROWS(GridAxis("0.1:0.01:0.0"));
  CHECK_THROWS(GridAxis("0.1:2.0:1.0"));
  CHECK_THROWS(GridAxis("0.0:0.0:1.0"));
  CHECK_THROWS(GridAxis("0.0:-0.1:1.0"));
}

TEST_CASE("parses default JSON grid configuration", "[MeshConfiguration]")
{
  SECTION("get_defaults returns valid parseable JSON")
  {
    std::string default_json_str = ConfigurationMesh<3>::get_defaults();
    DiFfRG::ConfigTree default_json = DiFfRG::json::parse(default_json_str);

    // Parse the defaults into a ConfigurationMesh
    ConfigurationMesh<3> mesh_config(default_json);

    // Verify it matches expected default configuration - all grids use the same default range
    REQUIRE(size(mesh_config.grids[0]) == 3);
    CHECK(mesh_config.grids[0][0].min == 0.0);
    CHECK(mesh_config.grids[0][0].max == 7e-3);
    CHECK(mesh_config.grids[0][1].min == 7e-3);
    CHECK(mesh_config.grids[0][1].max == 9e-3);
    CHECK(mesh_config.grids[0][2].min == 9e-3);
    CHECK(mesh_config.grids[0][2].max == 1.1e-2);

    REQUIRE(size(mesh_config.grids[1]) == 3);
    CHECK(mesh_config.grids[1][0].min == 0.0);
    CHECK(mesh_config.grids[1][0].max == 7e-3);

    REQUIRE(size(mesh_config.grids[2]) == 3);
    CHECK(mesh_config.grids[2][0].min == 0.0);
    CHECK(mesh_config.grids[2][0].max == 7e-3);

    CHECK(mesh_config.refine == 0);

    CHECK(mesh_config.get_lower_left() == dealii::Point<3, double>(0.0, 0.0, 0.0));
    CHECK(mesh_config.get_upper_right() == dealii::Point<3, double>(1.1e-2, 1.1e-2, 1.1e-2));
  }
}
