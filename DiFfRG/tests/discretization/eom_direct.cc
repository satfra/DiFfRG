#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <DiFfRG/common/init.hh>
#include <DiFfRG/discretization/common/eom.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <boilerplate/eom_accuracy.hh>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;

  template <int dim, typename Initial, typename EoM, typename Check>
  void check_field(const unsigned int degree, const Initial &initial, const EoM &eom, const Check &check,
                   const Config::EoMConfig &config = {}, const double mesh_lower = 0.,
                   const DiFfRG::internal::EoMSearchBox<dim> *physical_bounds = nullptr)
  {
    DiFfRG::Init();
    Triangulation<dim> mesh;
    GridGenerator::subdivided_hyper_cube(mesh, 4, mesh_lower, 1.);
    std::unique_ptr<FiniteElement<dim>> scalar;
    if (degree == 0)
      scalar = std::make_unique<FE_DGQ<dim>>(0);
    else
      scalar = std::make_unique<FE_Q<dim>>(degree);
    FESystem<dim> fe(*scalar, dim);
    DoFHandler<dim> dofs(mesh);
    dofs.distribute_dofs(fe);
    MappingQ1<dim> mapping;
    Vector<double> state(dofs.n_dofs());
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    for (const auto &cell : dofs.active_cell_iterators()) {
      cell->get_dof_indices(indices);
      for (unsigned int i = 0; i < indices.size(); ++i) {
        const auto point = mapping.transform_unit_to_real_cell(cell, fe.get_unit_support_points()[i]);
        state[indices[i]] = initial(point)[fe.system_to_component_index(i).first];
      }
    }
    auto cell = dofs.begin_active();
    const auto result = get_EoM_result<dim>(
        cell, state, dofs, mapping, eom, [](const auto &p, const auto &) { return p; }, config,
        std::optional<Point<dim>>{}, nullptr, physical_bounds);
    CHECK(cell->point_inside(result.point));
    if (physical_bounds) {
      const auto point = get_EoM_point<dim>(
          cell, state, dofs, mapping, eom, [](const auto &p, const auto &) { return p; }, config.abs_tol,
          config.max_iter, std::optional<Point<dim>>{}, physical_bounds);
      CHECK(point.distance(result.point) < 1e-10);
    }
    if constexpr (std::is_invocable_v<Check, decltype(result), Tensor<2, dim>>) {
      const auto du = Testing::direct_eom_state_derivative(state, dofs, mapping, result.point);
      check(result, du);
    } else {
      check(result);
    }
  }

  const auto state_eom = [](const auto &, const auto &u) {
    std::array<double, 2> value{{u[0], u[1]}};
    return value;
  };
} // namespace

TEST_CASE("Direct EoM rejects a symmetry-axis root with unstable normal curvature", "[EoM][direct][stability]")
{
  check_field<2>(
      2, [](const auto &p) { return std::array<double, 2>{{p[0] - .37, p[1] * (p[1] - .6)}}; }, state_eom,
      [](const auto &result, const auto &du) {
        Tensor<2, 2> exact_du;
        exact_du[0][0] = 1.;
        exact_du[1][1] = .6;
        Testing::print_direct_eom_accuracy("CG2 nonlinear axis", result.point, Point<2>(.37, .6), du, exact_du);
        CHECK(result.point.distance(Point<2>(.37, .6)) < 1e-9);
        REQUIRE(result.candidates.size() == 1);
        CHECK(result.candidates[0].residual < 1e-10);
      });
}

TEST_CASE("Direct EoM checks the one-sided critical cone at a weak corner", "[EoM][direct][stability]")
{
  // V=x^2/2+2xy+y^2/2 is minimized at (0,0) on the positive quadrant, although its full Hessian is indefinite.
  check_field<2>(
      1, [](const auto &p) { return std::array<double, 2>{{p[0] + 2 * p[1], 2 * p[0] + p[1]}}; }, state_eom,
      [](const auto &result) {
        CHECK(result.point.distance(Point<2>()) < 1e-10);
        REQUIRE(result.candidates.size() == 1);
        CHECK(result.candidates[0].active_bounds == std::array<int, 2>{{-1, -1}});
      });
}

TEST_CASE("Direct EoM checks both derivatives at an internal interface", "[EoM][direct][interface]")
{
  // F=|x-1/2| is positive on both sides: its interior zero is not a minimum of its integral.
  check_field<1>(
      1, [](const auto &p) { return std::array<double, 1>{{std::abs(p[0] - .5)}}; },
      [](const auto &, const auto &u) { return std::array<double, 1>{{u[0]}}; },
      [](const auto &result) {
        CHECK(result.point[0] == Catch::Approx(0.).margin(1e-12));
        REQUIRE(result.candidates.size() == 1);
        CHECK(result.candidates[0].point[0] == Catch::Approx(0.).margin(1e-12));
      });
}

TEST_CASE("Direct two-field EoM retains a singular source below the first FV centre", "[EoM][direct][source]")
{
  const unsigned int degree = GENERATE(0u, 1u, 3u);
  check_field<2>(
      degree, [](const auto &p) { return std::array<double, 2>{{1., 3 * (p[1] - .417)}}; },
      [](const auto &p, const auto &u) {
        return std::array<double, 2>{{u[0] - .01 / (std::sqrt(2 * p[0]) + 1e-14), u[1]}};
      },
      [](const auto &result) {
        CHECK(result.point.distance(Point<2>(.00005, .417)) < 1e-10);
        REQUIRE(result.candidates.size() == 1);
        CHECK(result.candidates[0].residual < 1e-10);
      });
}

TEST_CASE("Direct EoM respects physical planes cutting through cells", "[EoM][direct][boundary]")
{
  const DiFfRG::internal::EoMSearchBox<2> physical_bounds{Point<2>(0., 0.), Point<2>(.9, .9)};
  const double y_root = GENERATE(.37, 1.2);
  check_field<2>(
      0, [=](const auto &p) { return std::array<double, 2>{{p[0] + .05, p[1] - y_root}}; }, state_eom,
      [=](const auto &result) {
        CHECK(result.point.distance(Point<2>(0., std::min(y_root, .9))) < 1e-9);
        REQUIRE(result.candidates.size() == 1);
        CHECK(result.candidates[0].active_bounds[0] == -1);
        CHECK(result.candidates[0].active_bounds[1] == (y_root > .9 ? 1 : 0));
      },
      {}, -.125, &physical_bounds);
}

TEST_CASE("Direct EoM resolves a small boundary bias without forcing a normal zero", "[EoM][direct][boundary]")
{
  const double bias = GENERATE(-1e-5, 1e-5);
  const unsigned int degree = GENERATE(0u, 1u);
  check_field<2>(
      degree, [=](const auto &p) { return std::array<double, 2>{{p[0] - .37, 2 * p[1] + bias}}; }, state_eom,
      [=](const auto &result) {
        CHECK(result.point.distance(Point<2>(.37, std::max(0., -bias / 2))) < 1e-10);
        REQUIRE(result.candidates.size() == 1);
        CHECK(result.candidates[0].active_bounds[1] == (bias > 0. ? -1 : 0));
      });
}

TEST_CASE("Direct EoM reports failure when every stationary candidate is unstable", "[EoM][direct][failure]")
{
  CHECK_THROWS_WITH(check_field<2>(
                        1,
                        [](const auto &p) {
                          return std::array<double, 2>{
                              {-(p[0] - .5) + 10 * (p[1] - .5), -10 * (p[0] - .5) - (p[1] - .5)}};
                        },
                        state_eom, [](const auto &) {}),
                    Catch::Matchers::ContainsSubstring("no locally admissible constrained root"));
}

TEST_CASE("Direct EoM validates supplied physical mesh geometry", "[EoM][direct][geometry]")
{
  const auto initial = [](const auto &p) { return std::array<double, 2>{{p[0] - .3, p[1] - .4}}; };
  for (const auto &bounds : std::array<DiFfRG::internal::EoMSearchBox<2>, 3>{
           {{{-.1, 0.}, {.8, .9}}, {{0., 0.}, {0., .9}}, {{std::numeric_limits<double>::quiet_NaN(), 0.}, {.8, .9}}}})
    CHECK_THROWS_AS(check_field<2>(1, initial, state_eom, [](const auto &) {}, {}, 0., &bounds), std::invalid_argument);
}

TEST_CASE("Cached FV interpolation follows state updates, DoF renumbering and mesh refinement", "[EoM][direct][cache]")
{
  DiFfRG::Init init;
  Triangulation<2> mesh;
  GridGenerator::subdivided_hyper_cube(mesh, 3, 0., 1.);
  FESystem<2> fe(FE_DGQ<2>(0), 2);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  MappingQ1<2> mapping;
  DiFfRG::internal::EoMStateCache<2, double> cache;
  std::shared_ptr<DiFfRG::internal::ContinuousEoMGeometry<2, double>> original_geometry;
  for (unsigned int step = 0; step < 4; ++step) {
    if (step == 2) {
      std::vector<types::global_dof_index> reverse(dofs.n_dofs());
      for (unsigned int i = 0; i < reverse.size(); ++i)
        reverse[i] = reverse.size() - 1 - i;
      dofs.renumber_dofs(reverse);
    }
    if (step == 3) {
      dofs.begin_active()->set_refine_flag();
      mesh.execute_coarsening_and_refinement();
      dofs.distribute_dofs(fe);
    }
    Vector<double> state(dofs.n_dofs());
    std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
    for (const auto &cell : dofs.active_cell_iterators()) {
      cell->get_dof_indices(indices);
      const auto p = cell->center();
      for (unsigned int i = 0; i < indices.size(); ++i) {
        const unsigned int c = fe.system_to_component_index(i).first;
        state[indices[i]] =
            c == 0 ? p[0] - .37 + .1 * p[0] * p[1] + .01 * step : 2 * (p[1] - .417) + .2 * p[0] * p[0] - .02 * step;
      }
    }
    const auto reference = DiFfRG::internal::reconstruct_eom_state(state, dofs, mapping);
    const auto cached = DiFfRG::internal::reconstruct_eom_state(state, dofs, mapping, &cache);
    REQUIRE(cached.values.size() == reference.values.size());
    CHECK(cached.full_rank == reference.full_rank);
    for (unsigned int i = 0; i < reference.values.size(); ++i)
      CHECK(cached.values[i] == Catch::Approx(reference.values[i]).margin(2e-14));
    if (step == 0) original_geometry = cache.eom_geometry;
    CHECK((cache.eom_geometry == original_geometry) == (step < 3));
    auto fresh_cell = dofs.begin_active(), cached_cell = fresh_cell;
    const auto postprocess = [](const auto &p, const auto &) { return p; };
    const auto fresh = get_EoM_result(fresh_cell, state, dofs, mapping, state_eom, postprocess, {});
    const auto warm = get_EoM_result(cached_cell, state, dofs, mapping, state_eom, postprocess, {},
                                     std::optional<Point<2>>{}, &cache);
    CHECK(warm.point.distance(fresh.point) < 1e-11);
    REQUIRE(warm.hessian);
    REQUIRE(fresh.hessian);
    CHECK((*warm.hessian - *fresh.hessian).norm() < 1e-11);
  }
}

TEST_CASE("Fast Cartesian tensor readout agrees with native FE evaluation", "[EoM][direct][interpolation]")
{
  DiFfRG::Init init;
  const unsigned int degree = GENERATE(1u, 2u, 3u, 5u, 6u);
  Triangulation<2> mesh;
  GridGenerator::hyper_rectangle(mesh, Point<2>(-.3, .1), Point<2>(.7, 1.6));
  FESystem<2> fe(FE_Q<2>(degree), 3);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  MappingQ1<2> mapping;
  Vector<double> state(dofs.n_dofs());
  for (unsigned int i = 0; i < state.size(); ++i)
    state[i] = std::sin(1.7 * i) + .03 * i;
  const auto cell = dofs.begin_active();
  DiFfRG::internal::LocalSolution<2, Vector<double>> local(state, dofs, mapping, cell);
  Functions::FEFieldFunction<2, Vector<double>> native(dofs, state, mapping);
  for (const auto unit : {Point<2>(0., 0.), Point<2>(.23, .81), Point<2>(.5, .5), Point<2>(1., 1.)}) {
    const auto point = mapping.transform_unit_to_real_cell(cell, unit);
    Vector<double> expected(3), observed(3);
    native.vector_value(point, expected);
    local.vector_value(point, observed);
    for (unsigned int c = 0; c < 3; ++c)
      CHECK(observed[c] == Catch::Approx(expected[c]).margin(2e-13));
  }
}

TEST_CASE("Parallel direct search retains both wells in deterministic order", "[EoM][direct][parallel]")
{
  DiFfRG::Init init;
  Triangulation<2> mesh;
  GridGenerator::subdivided_hyper_cube(mesh, 8, 0., 1.);
  FESystem<2> fe(FE_Q<2>(3), 2);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  MappingQ1<2> mapping;
  Vector<double> state(dofs.n_dofs());
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  for (const auto &cell : dofs.active_cell_iterators()) {
    cell->get_dof_indices(indices);
    for (unsigned int i = 0; i < indices.size(); ++i) {
      const auto p = mapping.transform_unit_to_real_cell(cell, fe.get_unit_support_points()[i]);
      state[indices[i]] =
          fe.system_to_component_index(i).first == 0 ? (p[0] - .2) * (p[0] - .5) * (p[0] - .8) : p[1] - .43;
    }
  }
  const auto previous_threads = DiFfRG::n_threads();
  std::vector<EoMCandidate<2>> reference;
  for (const auto threads : {1u, 4u, 4u}) {
    DiFfRG::set_thread_limit(threads);
    const auto roots = DiFfRG::internal::find_direct_eom_candidates(state, dofs, mapping, state_eom,
                                                                    Config::EoMConfig{}, std::optional<Point<2>>{},
                                                                    DiFfRG::internal::eom_search_box<2>(dofs, nullptr));
    REQUIRE(roots.size() == 2);
    CHECK(roots[0].point.distance(Point<2>(.2, .43)) < 1e-10);
    CHECK(roots[1].point.distance(Point<2>(.8, .43)) < 1e-10);
    if (reference.empty()) reference = roots;
    for (unsigned int i = 0; i < roots.size(); ++i)
      CHECK(roots[i].point == reference[i].point);
  }
  DiFfRG::set_thread_limit(previous_threads);
}

TEST_CASE("Direct EoM line work switches between competing two-field minima", "[EoM][direct][ranking]")
{
  DiFfRG::Init init;
  Triangulation<2> mesh;
  GridGenerator::subdivided_hyper_cube(mesh, 4, 0., 1.);
  FESystem<2> fe(FE_Q<2>(3), 2);
  DoFHandler<2> dofs(mesh);
  dofs.distribute_dofs(fe);
  MappingQ1<2> mapping;
  Vector<double> state(dofs.n_dofs());
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  std::optional<Point<2>> previous;
  DiFfRG::internal::EoMStateCache<2, double> cache;
  for (const double tilt : {.002, -.002, .002}) {
    for (const auto &cell : dofs.active_cell_iterators()) {
      cell->get_dof_indices(indices);
      for (unsigned int i = 0; i < indices.size(); ++i) {
        const auto p = mapping.transform_unit_to_real_cell(cell, fe.get_unit_support_points()[i]);
        state[indices[i]] =
            fe.system_to_component_index(i).first == 0 ? (p[0] - .2) * (p[0] - .5) * (p[0] - .8) + tilt : p[1] - .43;
      }
    }
    double exact = tilt > 0. ? .2 : .8;
    for (unsigned int i = 0; i < 12; ++i)
      exact -= ((exact - .2) * (exact - .5) * (exact - .8) + tilt) / (3 * (exact - .5) * (exact - .5) - .09);
    auto cell = dofs.begin_active();
    const auto result = get_EoM_result(
        cell, state, dofs, mapping, state_eom, [](const auto &p, const auto &) { return p; }, Config::EoMConfig{},
        previous, &cache);
    REQUIRE(result.candidates.size() == 2);
    CHECK(result.point.distance(Point<2>(exact, .43)) < 1e-10);
    previous = result.search_point;
  }
}
