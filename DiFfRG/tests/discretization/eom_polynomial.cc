#define CATCH_CONFIG_MAIN
#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <catch2/catch_all.hpp>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <iomanip>
#include <iostream>

namespace
{
  using namespace dealii;
  using namespace DiFfRG;

  template <int dim, typename Initial, typename EoM, typename Check>
  void run(const unsigned int subdivisions, const Initial &initial, const EoM &eom, const Config::EoMConfig &config,
           const Check &check, const EoMDataKind data_kind = EoMDataKind::finite_volume)
  {
    DiFfRG::Init();
    Triangulation<dim> mesh;
    GridGenerator::subdivided_hyper_cube(mesh, subdivisions, 0., 1.);
    FESystem<dim> fe(FE_DGQ<dim>(0), dim);
    DoFHandler<dim> dofs(mesh);
    dofs.distribute_dofs(fe);
    MappingQ1<dim> mapping;
    Vector<double> state(dofs.n_dofs());
    const QGauss<dim> quadrature(4);
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    for (const auto &cell : dofs.active_cell_iterators()) {
      std::array<double, dim> average{};
      for (unsigned int q = 0; q < quadrature.size(); ++q) {
        const auto p = mapping.transform_unit_to_real_cell(cell, quadrature.point(q));
        const auto value = initial(p);
        for (unsigned int c = 0; c < dim; ++c)
          average[c] += quadrature.weight(q) * value[c];
      }
      cell->get_dof_indices(indices);
      for (unsigned int i = 0; i < indices.size(); ++i)
        state[indices[i]] = average[fe.system_to_component_index(i).first];
    }
    auto cell = dofs.begin_active();
    const auto result = get_EoM_result<dim>(
        cell, state, dofs, mapping, eom, [](const auto &p, const auto &) { return p; }, config,
        std::optional<Point<dim>>{}, nullptr, nullptr, data_kind);
    check(result);
  }

  const auto raw_eom = [](const auto &, const auto &u) { return std::array<double, 2>{{u[0], u[1]}}; };
} // namespace

TEST_CASE("FV cubic patch recovers coupled root and Hessian from true cell averages", "[EoM][patch][moments]")
{
  Config::EoMConfig config;
  config.patch_order = 3;
  const bool finite_volume = GENERATE(true, false);
  run<2>(
      16,
      [](const auto &p) {
        const double x = p[0] - .3, y = p[1] - .4;
        return std::array<double, 2>{{2 * x + .4 * y + .3 * x * x + .2 * x * y + .1 * y * y * y,
                                      .4 * x + 3 * y + .1 * x * x * x + .3 * x * y + .2 * y * y}};
      },
      raw_eom, config,
      [=](const auto &result) {
        REQUIRE(result.hessian.has_value());
        if (!finite_volume) {
          CHECK_FALSE(result.local_fit);
          return;
        }
        REQUIRE(result.local_fit != nullptr);
        Tensor<2, 2> exact;
        exact[0][0] = 2.;
        exact[0][1] = exact[1][0] = .4;
        exact[1][1] = 3.;
        const double root_error = result.point.distance(Point<2>(.3, .4));
        CHECK(root_error < 1e-10);
        CHECK((*result.hessian - exact).norm() < 1e-10);
        for (const auto order : result.local_fit->orders)
          CHECK(order == 3);
        CHECK(result.hessian_fit_difference.has_value());
        // Check the fitted vector and its full Jacobian away from the fitted root as well.
        for (const auto &point : {Point<2>(.17, .62), Point<2>(.55, .23)}) {
          const double x = point[0] - .3, y = point[1] - .4;
          const auto values = result.local_fit->value(point);
          const auto derivative = result.local_fit->state_jacobian(point);
          CHECK(values[0] == Catch::Approx(2 * x + .4 * y + .3 * x * x + .2 * x * y + .1 * y * y * y).margin(1e-10));
          CHECK(values[1] == Catch::Approx(.4 * x + 3 * y + .1 * x * x * x + .3 * x * y + .2 * y * y).margin(1e-10));
          CHECK(derivative[0][0] == Catch::Approx(2 + .6 * x + .2 * y).margin(1e-10));
          CHECK(derivative[0][1] == Catch::Approx(.4 + .2 * x + .3 * y * y).margin(1e-10));
          CHECK(derivative[1][0] == Catch::Approx(.4 + .3 * x * x + .3 * y).margin(1e-10));
          CHECK(derivative[1][1] == Catch::Approx(3 + .3 * x + .4 * y).margin(1e-10));
        }
      },
      finite_volume ? EoMDataKind::finite_volume : EoMDataKind::finite_element);
}

TEST_CASE("FV polynomial refinement keeps explicit coordinate sources out of the mass Hessian", "[EoM][patch][source]")
{
  Config::EoMConfig config;
  config.patch_order = 3;
  run<2>(
      12, [](const auto &p) { return std::array<double, 2>{{1., 3 * (p[1] - .417)}}; },
      [](const auto &p, const auto &u) {
        return std::array<double, 2>{{u[0] - .01 / (std::sqrt(2 * p[0]) + 1e-14), u[1]}};
      },
      config,
      [](const auto &result) {
        REQUIRE(result.local_fit != nullptr);
        CHECK(result.point.distance(Point<2>(.00005, .417)) < 1e-10);
        REQUIRE(result.hessian.has_value());
        CHECK(std::abs((*result.hessian)[0][0]) < 1e-10);
        CHECK((*result.hessian)[1][1] == Catch::Approx(3.).margin(1e-10));
      });
}

TEST_CASE("Underdetermined FV patches reduce degree explicitly", "[EoM][patch][rank]")
{
  Config::EoMConfig config;
  config.patch_order = 3;
  config.patch_direction = {0., 0.};
  config.patch_skip_layers = 0;
  run<2>(
      2, [](const auto &p) { return std::array<double, 2>{{2 * (p[0] - .3), 3 * (p[1] - .4)}}; }, raw_eom, config,
      [](const auto &result) {
        REQUIRE(result.local_fit != nullptr);
        for (const auto order : result.local_fit->orders)
          CHECK(order == 1);
        CHECK(result.point.distance(Point<2>(.3, .4)) < 1e-10);
        CHECK((*result.hessian)[0][0] == Catch::Approx(2.).margin(1e-10));
      });
}

TEST_CASE("An empty one-sided FV patch leaves the coarse root usable", "[EoM][patch][fallback]")
{
  Config::EoMConfig config;
  config.patch_order = 3;
  config.patch_direction = {1., 0.};
  run<2>(
      3, [](const auto &p) { return std::array<double, 2>{{p[0] - .99, p[1] - .4}}; }, raw_eom, config,
      [](const auto &result) {
        CHECK_FALSE(result.local_fit);
        CHECK(result.point.distance(Point<2>(.99, .4)) < 1e-10);
        CHECK(result.hessian.has_value());
      });
}

TEST_CASE("One-sided FV extrapolation selects the requested Hessian limit at a kink", "[EoM][patch][one-sided]")
{
  DiFfRG::Init();
  constexpr double root = .473;
  const double direction = GENERATE(-1., 1.);
  Triangulation<1> mesh;
  GridGenerator::subdivided_hyper_cube(mesh, 12, 0., 1.);
  FESystem<1> fe(FE_DGQ<1>(0), 1);
  DoFHandler<1> dofs(mesh);
  dofs.distribute_dofs(fe);
  MappingQ1<1> mapping;
  Vector<double> state(dofs.n_dofs());
  // Continuous u, with one-sided derivatives 0.5 and 2 at its zero; exact split-cell averages.
  const auto primitive = [](double x) {
    const double s = x - root;
    return s < 0. ? .25 * s * s : s * s + .5 * s * s * s;
  };
  for (const auto &cell : dofs.active_cell_iterators()) {
    std::vector<types::global_dof_index> indices(1);
    cell->get_dof_indices(indices);
    const double a = cell->vertex(0)[0], b = cell->vertex(1)[0];
    state[indices[0]] = (primitive(b) - primitive(a)) / (b - a);
  }
  Config::EoMConfig config;
  config.patch_order = 3;
  config.patch_direction = {direction};
  config.patch_skip_layers = 1;
  config.patch_radius = 6.;
  auto cell = dofs.begin_active();
  const auto result = get_EoM_result<1>(
      cell, state, dofs, mapping, [](const auto &, const auto &u) { return std::array<double, 1>{{u[0]}}; },
      [](const auto &p, const auto &) { return p; }, config, std::optional<Point<1>>{}, nullptr, nullptr,
      EoMDataKind::finite_volume);
  REQUIRE(result.local_fit != nullptr);
  CHECK(result.point[0] == Catch::Approx(root).margin(1e-10));
  REQUIRE(result.hessian.has_value());
  CHECK((*result.hessian)[0][0] == Catch::Approx(direction > 0. ? 2. : .5).margin(1e-10));
}

TEST_CASE("Oblique one-sided FV recovery resolves a two-field Hessian jump", "[EoM][patch][one-sided][2D]")
{
  const double side = GENERATE(-1., 1.);
  Config::EoMConfig config;
  config.patch_order = 3;
  config.patch_radius = 6.;
  config.patch_direction = {side * .6, side * .8};
  config.patch_skip_layers = 1;
  run<2>(
      12,
      [](const auto &p) {
        const double x = p[0] - .43, y = p[1] - .37;
        const double normal_coordinate = .6 * x + .8 * y;
        const double normal_gradient = normal_coordinate < 0.
                                           ? .2 * normal_coordinate
                                           : 1.2 * normal_coordinate + .3 * normal_coordinate * normal_coordinate;
        return std::array<double, 2>{{2 * x + .6 * normal_gradient, 3 * y + .8 * normal_gradient}};
      },
      raw_eom, config,
      [=](const auto &result) {
        REQUIRE(result.local_fit != nullptr);
        CHECK(result.point.distance(Point<2>(.43, .37)) < 1e-9);
        REQUIRE(result.hessian.has_value());
        Tensor<2, 2> exact;
        const double slope = side > 0. ? 1.2 : .2;
        exact[0][0] = 2. + slope * .36;
        exact[1][1] = 3. + slope * .64;
        exact[0][1] = exact[1][0] = slope * .48;
        CHECK((*result.hessian - exact).norm() < 1e-9);
      });
}

TEST_CASE("Cubic FV recovery improves a nonpolynomial coupled field", "[EoM][patch][accuracy]")
{
  const unsigned int subdivisions = GENERATE(6u, 8u, 12u);
  const auto initial = [](const auto &p) {
    const double x = p[0] - .3, y = p[1] - .4;
    return std::array<double, 2>{{std::exp(2 * x) - 1 + .3 * y + .2 * x * y, .3 * x + 2 * std::sinh(y) + .2 * x * x}};
  };
  Tensor<2, 2> exact;
  exact[0][0] = exact[1][1] = 2.;
  exact[0][1] = exact[1][0] = .3;
  double old_root = 0., old_hessian = 0., new_root = 0., new_hessian = 0.;
  Config::EoMConfig config;
  config.patch_direction = {0., 0.};
  config.patch_skip_layers = 0;
  config.patch_radius = 4.;
  run<2>(subdivisions, initial, raw_eom, config, [&](const auto &result) {
    old_root = result.point.distance(Point<2>(.3, .4));
    old_hessian = (*result.hessian - exact).norm();
  });
  config.patch_order = 3;
  run<2>(subdivisions, initial, raw_eom, config, [&](const auto &result) {
    REQUIRE(result.local_fit != nullptr);
    for (const auto order : result.local_fit->orders)
      CHECK(order == 3);
    new_root = result.point.distance(Point<2>(.3, .4));
    new_hessian = (*result.hessian - exact).norm();
  });
  std::cout << std::setprecision(17) << "nonpolynomial recovery | cells_per_axis=" << subdivisions
            << " | old_root=" << old_root << " | new_root=" << new_root << " | old_hessian=" << old_hessian
            << " | new_hessian=" << new_hessian << '\n';
  CAPTURE(subdivisions, old_root, new_root, old_hessian, new_hessian);
  CHECK(new_root < old_root);
  CHECK(new_hessian < old_hessian);
}

TEST_CASE("Polynomial EoM options validate order radius direction and parity", "[EoM][patch][config]")
{
  CHECK_THROWS_AS(Config::EoMConfig(ConfigTree::from_json_string(R"({"discretization":{"EoM_patch_order":4}})")),
                  std::invalid_argument);
  CHECK_THROWS_AS(Config::EoMConfig(ConfigTree::from_json_string(R"({"discretization":{"EoM_patch_parity":[[2,0]]}})")),
                  std::invalid_argument);
  auto config = Config::EoMConfig(ConfigTree::from_json_string(R"({"discretization":{
    "EoM_patch_order":3,"EoM_patch_radius":5,"EoM_patch_direction":[1,0],
    "EoM_patch_parity":[[-1,1],[1,-1]]}})"));
  CHECK(config.patch_order == 3);
  CHECK(config.patch_radius == 5.);
  CHECK(config.patch_skip_layers == 1);
  const Config::EoMConfig defaults(ConfigTree::from_json_string(Config::EoMConfig::get_defaults()));
  CHECK(defaults.patch_direction.empty());
  CHECK(defaults.patch_skip_layers == 1);
  CHECK(defaults.patch_radius == 7.);
  CHECK(config.patch_parity[1][1] == -1);
  config.patch_order = 4;
  CHECK_THROWS_AS(config.validate(), std::invalid_argument);
  config.patch_order = 3;
  config.patch_radius = 0.;
  CHECK_THROWS_AS(config.validate(), std::invalid_argument);
  config.patch_radius = 4.;
  config.patch_direction = {0., 0.};
  CHECK_THROWS_AS(config.validate(), std::invalid_argument);
  config.patch_direction = {1., 0.};
  config.patch_parity[0][0] = 2;
  CHECK_THROWS_AS(config.validate(), std::invalid_argument);
}

TEST_CASE("Default FV patches exclude a full upper layer in every coordinate", "[EoM][patch][defaults]")
{
  DiFfRG::Init();
  const auto check = []<int dim>() {
    Triangulation<dim> mesh;
    GridGenerator::subdivided_hyper_cube(mesh, 16, 0., 1.);
    FESystem<dim> fe(FE_DGQ<dim>(0), dim);
    DoFHandler<dim> dofs(mesh);
    dofs.distribute_dofs(fe);
    MappingQ1<dim> mapping;
    Vector<double> state(dofs.n_dofs());
    Point<dim> anchor;
    for (unsigned int d = 0; d < dim; ++d)
      anchor[d] = .4;
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    for (const auto &cell : dofs.active_cell_iterators()) {
      cell->get_dof_indices(indices);
      double contamination = 0.;
      for (unsigned int d = 0; d < dim; ++d) {
        const double h = cell->extent_in_direction(d), kink = anchor[d] + .8 * h;
        const double a = std::min(0., cell->vertex(0)[d] - kink);
        const double b = std::min(0., cell->vertex(GeometryInfo<dim>::vertices_per_cell - 1)[d] - kink);
        contamination += 100. * (b * b - a * a) / (2. * h);
      }
      for (unsigned int i = 0; i < indices.size(); ++i) {
        const auto c = fe.system_to_component_index(i).first;
        const double h = cell->extent_in_direction(c), x = cell->center()[c] - anchor[c];
        state[indices[i]] = 2. * x + .3 * (x * x * x + x * h * h / 4.) + contamination;
      }
    }
    Config::EoMConfig config;
    config.patch_order = 3;
    const auto fit = DiFfRG::internal::fit_eom_polynomial(state, dofs, mapping, anchor, config);
    REQUIRE(fit.has_value());
    const auto value = fit->value(anchor);
    const auto jacobian = fit->state_jacobian(anchor);
    for (unsigned int i = 0; i < dim; ++i) {
      CHECK(fit->orders[i] == 3);
      CHECK(std::abs(value[i]) < 1e-9);
      for (unsigned int j = 0; j < dim; ++j)
        CHECK(jacobian[i][j] == Catch::Approx(i == j ? 2. : 0.).margin(1e-9));
    }
  };
  check.template operator()<1>();
  check.template operator()<2>();
  check.template operator()<3>();
}
