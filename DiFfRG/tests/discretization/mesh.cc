#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>

#include <algorithm>
#include <limits>
#include <vector>

//--------------------------------------------
// Test logic
//--------------------------------------------

namespace
{
  using Catch::Matchers::WithinAbs;
  using DiFfRG::RectangularMesh;
  using DiFfRG::RectangularMeshOptions;
  using DiFfRG::Config::ConfigurationMesh;
  using DiFfRG::Config::GridAxis;

  struct AxisGeometry {
    double lower = std::numeric_limits<double>::max();
    double upper = std::numeric_limits<double>::lowest();
    std::vector<double> widths;
    std::vector<double> centers;
    std::vector<double> vertices;
  };

  AxisGeometry get_axis_geometry(const dealii::Triangulation<1> &triangulation)
  {
    AxisGeometry geometry;
    for (const auto &cell : triangulation.active_cell_iterators()) {
      const double lower = std::min(cell->vertex(0)[0], cell->vertex(1)[0]);
      const double upper = std::max(cell->vertex(0)[0], cell->vertex(1)[0]);
      geometry.lower = std::min(geometry.lower, lower);
      geometry.upper = std::max(geometry.upper, upper);
      geometry.widths.push_back(upper - lower);
      geometry.centers.push_back(cell->center()[0]);
      geometry.vertices.push_back(lower);
      geometry.vertices.push_back(upper);
    }
    return geometry;
  }
} // namespace

TEST_CASE("RectangularMesh default construction keeps configured geometry", "[discretization][mesh]")
{
  const std::vector<GridAxis> x_grid{GridAxis(0.0, 0.1, 5.0)};
  const ConfigurationMesh<1> mesh_config(0u, x_grid);
  const RectangularMesh<1> mesh(mesh_config);
  const auto geometry = get_axis_geometry(mesh.get_triangulation());

  CHECK(mesh.get_triangulation().n_active_cells() == 50);
  CHECK_THAT(geometry.lower, WithinAbs(0.0, 1.0e-12));
  CHECK_THAT(geometry.upper, WithinAbs(5.0, 1.0e-12));
  for (const double width : geometry.widths)
    CHECK_THAT(width, WithinAbs(0.1, 1.0e-12));
}

TEST_CASE("Deprecated JSON RectangularMesh delegates to ConfigurationMesh", "[discretization][mesh][migration]")
{
  const DiFfRG::ConfigTree json = DiFfRG::json::parse(R"({
    "discretization": {
      "grid": {
        "x_grid": "0.0:0.25:1.0",
        "y_grid": "0.0:0.5:2.0",
        "z_grid": "0.0:1.0:1.0",
        "refine": 0
      }
    }
  })");

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  const RectangularMesh<2> legacy(json);
#pragma GCC diagnostic pop
  const RectangularMesh<2> current{ConfigurationMesh<2>(json)};

  CHECK(legacy.get_triangulation().n_active_cells() == current.get_triangulation().n_active_cells());
  auto legacy_cell = legacy.get_triangulation().begin_active();
  auto current_cell = current.get_triangulation().begin_active();
  for (; legacy_cell != legacy.get_triangulation().end(); ++legacy_cell, ++current_cell)
    CHECK(legacy_cell->center().distance(current_cell->center()) == Catch::Approx(0.0));
}

TEST_CASE("RectangularMesh can center its first cell on the origin", "[discretization][mesh][origin-centered]")
{
  const std::vector<GridAxis> x_grid{GridAxis(0.0, 0.1, 5.0)};
  const ConfigurationMesh<1> mesh_config(0u, x_grid);
  const RectangularMesh<1> mesh(mesh_config, RectangularMeshOptions{.origin_cell_centered = true});
  const auto geometry = get_axis_geometry(mesh.get_triangulation());

  constexpr double expected_width = 5.0 / 50.5;
  CHECK(mesh.get_triangulation().n_active_cells() == 51);
  CHECK_THAT(geometry.lower, WithinAbs(-0.5 * expected_width, 1.0e-12));
  CHECK_THAT(geometry.upper, WithinAbs(5.0, 1.0e-12));
  CHECK(std::ranges::any_of(geometry.centers, [](const double center) { return std::abs(center) <= 1.0e-12; }));
  for (const double width : geometry.widths)
    CHECK_THAT(width, WithinAbs(expected_width, 1.0e-12));
}

TEST_CASE("RectangularMesh centers the origin along every axis", "[discretization][mesh][origin-centered]")
{
  const std::vector<GridAxis> x_grid{GridAxis(0.0, 0.25, 1.0)};
  const std::vector<GridAxis> y_grid{GridAxis(0.0, 0.5, 2.0)};
  const ConfigurationMesh<2> mesh_config(0u, x_grid, y_grid);
  const RectangularMesh<2> mesh(mesh_config, RectangularMeshOptions{.origin_cell_centered = true});

  bool found_origin_cell = false;
  double upper_x = std::numeric_limits<double>::lowest();
  double upper_y = std::numeric_limits<double>::lowest();
  for (const auto &cell : mesh.get_triangulation().active_cell_iterators()) {
    found_origin_cell =
        found_origin_cell || (std::abs(cell->center()[0]) <= 1.0e-12 && std::abs(cell->center()[1]) <= 1.0e-12);
    for (const auto vertex_index : cell->vertex_indices()) {
      upper_x = std::max(upper_x, cell->vertex(vertex_index)[0]);
      upper_y = std::max(upper_y, cell->vertex(vertex_index)[1]);
    }
  }

  CHECK(mesh.get_triangulation().n_active_cells() == 25);
  CHECK(found_origin_cell);
  CHECK_THAT(upper_x, WithinAbs(1.0, 1.0e-12));
  CHECK_THAT(upper_y, WithinAbs(2.0, 1.0e-12));
}

TEST_CASE("RectangularMesh centers the two-dimensional JSON grid on the origin",
          "[discretization][mesh][origin-centered]")
{
  const DiFfRG::ConfigTree json = DiFfRG::json::parse(R"({
    "discretization": {
      "grid": {
        "x_grid": "0.0:1e-2:2e-1",
        "y_grid": "0.0:1e-2:2e-1",
        "z_grid": "1e-5:5e-3:2e-1",
        "refine": 0
      }
    }
  })");
  const ConfigurationMesh<2> mesh_config(json);
  const RectangularMesh<2> mesh(mesh_config, RectangularMeshOptions{.origin_cell_centered = true});

  constexpr double expected_width = 0.2 / 20.5;
  double lower_x = std::numeric_limits<double>::max();
  double lower_y = std::numeric_limits<double>::max();
  double upper_x = std::numeric_limits<double>::lowest();
  double upper_y = std::numeric_limits<double>::lowest();
  bool found_origin_cell = false;
  std::size_t cells_centered_on_x_axis = 0;
  std::size_t cells_centered_on_y_axis = 0;
  double largest_x_axis_cell_center = 0.0;
  double largest_y_axis_cell_center = 0.0;

  for (const auto &cell : mesh.get_triangulation().active_cell_iterators()) {
    double cell_lower_x = std::numeric_limits<double>::max();
    double cell_lower_y = std::numeric_limits<double>::max();
    double cell_upper_x = std::numeric_limits<double>::lowest();
    double cell_upper_y = std::numeric_limits<double>::lowest();
    for (const auto vertex_index : cell->vertex_indices()) {
      cell_lower_x = std::min(cell_lower_x, cell->vertex(vertex_index)[0]);
      cell_lower_y = std::min(cell_lower_y, cell->vertex(vertex_index)[1]);
      cell_upper_x = std::max(cell_upper_x, cell->vertex(vertex_index)[0]);
      cell_upper_y = std::max(cell_upper_y, cell->vertex(vertex_index)[1]);
    }

    lower_x = std::min(lower_x, cell_lower_x);
    lower_y = std::min(lower_y, cell_lower_y);
    upper_x = std::max(upper_x, cell_upper_x);
    upper_y = std::max(upper_y, cell_upper_y);
    found_origin_cell =
        found_origin_cell || (std::abs(cell->center()[0]) <= 1.0e-12 && std::abs(cell->center()[1]) <= 1.0e-12);
    if (std::abs(cell->center()[1]) <= 1.0e-12) {
      ++cells_centered_on_x_axis;
      largest_x_axis_cell_center = std::max(largest_x_axis_cell_center, cell->center()[0]);
    }
    if (std::abs(cell->center()[0]) <= 1.0e-12) {
      ++cells_centered_on_y_axis;
      largest_y_axis_cell_center = std::max(largest_y_axis_cell_center, cell->center()[1]);
    }

    CHECK_THAT(cell_upper_x - cell_lower_x, WithinAbs(expected_width, 1.0e-12));
    CHECK_THAT(cell_upper_y - cell_lower_y, WithinAbs(expected_width, 1.0e-12));
  }

  CHECK(mesh_config.get_triangulation_data(true).step_sizes.size() == 2);
  CHECK(mesh.get_triangulation().n_active_cells() == 21 * 21);
  CHECK(found_origin_cell);
  CHECK(cells_centered_on_x_axis == 21);
  CHECK(cells_centered_on_y_axis == 21);
  CHECK_THAT(largest_x_axis_cell_center, WithinAbs(20.0 * expected_width, 1.0e-12));
  CHECK_THAT(largest_y_axis_cell_center, WithinAbs(20.0 * expected_width, 1.0e-12));
  CHECK_THAT(lower_x, WithinAbs(-0.5 * expected_width, 1.0e-12));
  CHECK_THAT(lower_y, WithinAbs(-0.5 * expected_width, 1.0e-12));
  CHECK_THAT(upper_x, WithinAbs(0.2, 1.0e-12));
  CHECK_THAT(upper_y, WithinAbs(0.2, 1.0e-12));
}

TEST_CASE("RectangularMesh keeps a final refined cell centered on the origin",
          "[discretization][mesh][origin-centered]")
{
  const std::vector<GridAxis> x_grid{GridAxis(0.0, 0.25, 1.0)};
  const ConfigurationMesh<1> mesh_config(1u, x_grid);
  const RectangularMesh<1> mesh(mesh_config, RectangularMeshOptions{.origin_cell_centered = true});
  const auto geometry = get_axis_geometry(mesh.get_triangulation());

  constexpr double expected_coarse_width = 1.0 / 4.75;
  constexpr double expected_final_width = expected_coarse_width / 2.0;
  CHECK(mesh.get_triangulation().n_active_cells() == 10);
  CHECK_THAT(geometry.lower, WithinAbs(-0.5 * expected_final_width, 1.0e-12));
  CHECK_THAT(geometry.upper, WithinAbs(1.0, 1.0e-12));
  CHECK(std::ranges::any_of(geometry.centers, [](const double center) { return std::abs(center) <= 1.0e-12; }));
  for (const double width : geometry.widths)
    CHECK_THAT(width, WithinAbs(expected_final_width, 1.0e-12));
}

TEST_CASE("RectangularMesh only adjusts the first configured subrange", "[discretization][mesh][origin-centered]")
{
  const std::vector<GridAxis> x_grid{GridAxis(0.0, 0.25, 1.0), GridAxis(1.0, 0.5, 2.0)};
  const ConfigurationMesh<1> mesh_config(0u, x_grid);
  const RectangularMesh<1> mesh(mesh_config, RectangularMeshOptions{.origin_cell_centered = true});
  const auto geometry = get_axis_geometry(mesh.get_triangulation());

  constexpr double expected_first_width = 1.0 / 4.5;
  CHECK(mesh.get_triangulation().n_active_cells() == 7);
  CHECK_THAT(geometry.upper, WithinAbs(2.0, 1.0e-12));
  CHECK(std::ranges::any_of(geometry.vertices, [](const double vertex) { return std::abs(vertex - 1.0) <= 1.0e-12; }));
  CHECK(std::ranges::count_if(geometry.widths, [](const double width) {
          return std::abs(width - expected_first_width) <= 1.0e-12;
        }) == 5);
  CHECK(std::ranges::count_if(geometry.widths, [](const double width) { return std::abs(width - 0.5) <= 1.0e-12; }) ==
        2);
}

TEST_CASE("RectangularMesh rejects origin centering for axes not starting at zero",
          "[discretization][mesh][origin-centered]")
{
  const std::vector<GridAxis> x_grid{GridAxis(0.1, 0.1, 1.0)};
  const ConfigurationMesh<1> mesh_config(0u, x_grid);

  CHECK_THROWS_WITH(RectangularMesh<1>(mesh_config, RectangularMeshOptions{.origin_cell_centered = true}),
                    "Origin-centered rectangular meshes require every configured axis to start at zero.");
}
