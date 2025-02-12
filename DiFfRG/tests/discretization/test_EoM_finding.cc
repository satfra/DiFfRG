
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>

#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>

#include <DiFfRG/discretization/common/EoM.hh>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test 1D EoM finding on CG Constant model", "[discretization][EoM][1d]")
{
  using namespace dealii;
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim, dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = CG::Assembler<Discretization, Model>;

  JSONValue json = json::value(
      {{"physical", {}},
       {"integration",
        {{"x_quadrature_order", 32},
         {"angle_quadrature_order", 8},
         {"x0_quadrature_order", 16},
         {"x0_summands", 8},
         {"q0_quadrature_order", 16},
         {"q0_summands", 8},
         {"x_extent_tolerance", 1e-3},
         {"x0_extent_tolerance", 1e-3},
         {"q0_extent_tolerance", 1e-3},
         {"jacobian_quadrature_factor", 0.5}}},
       {"discretization",
        {{"fe_order", GENERATE(1, 2, 3, 4, 5)},
         {"threads", 8},
         {"batch_size", 64},
         {"overintegration", 0},
         {"output_subdivisions", 2},

         {"EoM_abs_tol", 1e-14},
         {"EoM_max_iter", 200},

         {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
         {"adaptivity",
          {{"start_adapt_at", 0.},
           {"adapt_dt", 1e-1},
           {"level", 0},
           {"refine_percent", 1e-1},
           {"coarsen_percent", 5e-2}}}}},
       {"timestepping",
        {{"final_time", 1.},
         {"output_dt", 1e-1},
         {"explicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}},
         {"implicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}}}},
       {"output", {{"live_plot", false}, {"verbosity", 0}}}});

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = -1.;
  p_prm.initial_x1[0] = GENERATE(take(5, random(1.1, 10.)));

  const double expected_EoM = -p_prm.initial_x0[0] / p_prm.initial_x1[0];

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
    log->info("DiFfRG Application started");
  } catch (const spdlog::spdlog_ex &e) {
    // nothing, the logger is already set up
  }

  // Define the objects needed to run the simulation
  Model model(p_prm);
  RectangularMesh<dim> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  // Set up the initial condition
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  const VectorType &src = initial_condition.spatial_data();

  const auto &dof_handler = discretization.get_dof_handler();
  const auto &mapping = discretization.get_mapping();

  auto EoM_cell = dof_handler.begin_active();

  const double EoM_abs_tol = json.get_double("/discretization/EoM_abs_tol");
  const uint EoM_max_iter = json.get_uint("/discretization/EoM_max_iter");

  const auto EoM = get_EoM_point(
      EoM_cell, src, dof_handler, mapping, [&](const auto &p, const auto &values) { return model.EoM(p, values); },
      [&](const auto &p, const auto &) { return p; }, EoM_abs_tol, EoM_max_iter);

  if (!(std::abs(EoM[0] - expected_EoM) < 1e-6)) {
    std::cout << "ERROR: " << abs(EoM[0] - expected_EoM) << std::endl;
    std::cout << "EoM: " << EoM[0] << " expected: " << expected_EoM << std::endl;
  }
  REQUIRE(std::abs(EoM[0] - expected_EoM) < 1e-6);
}

TEST_CASE("Test 2D EoM finding on CG Constant model", "[discretization][EoM][2d]")
{
  using namespace dealii;
  using namespace DiFfRG;

  constexpr uint dim = 2;
  using Model = Testing::ModelConstant<dim, dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = CG::Assembler<Discretization, Model>;

  JSONValue json = json::value(
      {{"physical", {}},
       {"integration",
        {{"x_quadrature_order", 32},
         {"angle_quadrature_order", 8},
         {"x0_quadrature_order", 16},
         {"x0_summands", 8},
         {"q0_quadrature_order", 16},
         {"q0_summands", 8},
         {"x_extent_tolerance", 1e-3},
         {"x0_extent_tolerance", 1e-3},
         {"q0_extent_tolerance", 1e-3},
         {"jacobian_quadrature_factor", 0.5}}},
       {"discretization",
        {{"fe_order", GENERATE(1, 2, 3, 4, 5)},
         {"threads", 8},
         {"batch_size", 64},
         {"overintegration", 0},
         {"output_subdivisions", 2},

         {"EoM_abs_tol", 1e-14},
         {"EoM_max_iter", 200},

         {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
         {"adaptivity",
          {{"start_adapt_at", 0.},
           {"adapt_dt", 1e-1},
           {"level", 0},
           {"refine_percent", 1e-1},
           {"coarsen_percent", 5e-2}}}}},
       {"timestepping",
        {{"final_time", 1.},
         {"output_dt", 1e-1},
         {"explicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}},
         {"implicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}}}},
       {"output", {{"live_plot", false}, {"verbosity", 0}}}});

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = -1.;
  p_prm.initial_x1[0] = GENERATE(take(3, random(1.1, 10.)));
  p_prm.initial_x0[1] = -1.;
  p_prm.initial_x1[1] = GENERATE(take(3, random(1.1, 10.)));

  std::array<double, 2> expected_EoM{{}};
  expected_EoM[0] = -p_prm.initial_x0[0] / p_prm.initial_x1[0];
  expected_EoM[1] = -p_prm.initial_x0[1] / p_prm.initial_x1[1];

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
    log->info("DiFfRG Application started");
  } catch (const spdlog::spdlog_ex &e) {
    // nothing, the logger is already set up
  }

  // Define the objects needed to run the simulation
  Model model(p_prm);
  RectangularMesh<dim> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  // Set up the initial condition
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  const VectorType &src = initial_condition.spatial_data();

  const auto &dof_handler = discretization.get_dof_handler();
  const auto &mapping = discretization.get_mapping();

  auto EoM_cell = dof_handler.begin_active();

  const double EoM_abs_tol = json.get_double("/discretization/EoM_abs_tol");
  const uint EoM_max_iter = json.get_uint("/discretization/EoM_max_iter");

  const auto EoM = get_EoM_point(
      EoM_cell, src, dof_handler, mapping, [&](const auto &p, const auto &values) { return model.EoM(p, values); },
      [&](const auto &p, const auto &) { return p; }, EoM_abs_tol, EoM_max_iter);

  if (!(std::abs(EoM[0] - expected_EoM[0]) < 1e-6) || !(std::abs(EoM[1] - expected_EoM[1]) < 1e-6)) {
    std::cout << "ERROR: " << sqrt(powr<2>(EoM[0] - expected_EoM[0]) + powr<2>(EoM[1] - expected_EoM[1])) << std::endl;
    std::cout << "EoM: " << EoM << " expected: (" << expected_EoM[0] << ", " << expected_EoM[1] << ")" << std::endl;
  }
  REQUIRE(std::abs(EoM[0] - expected_EoM[0]) < 1e-6);
  REQUIRE(std::abs(EoM[1] - expected_EoM[1]) < 1e-6);
}

TEST_CASE("Test 3D EoM finding on CG Constant model", "[discretization][EoM][3d]")
{
  using namespace dealii;
  using namespace DiFfRG;

  constexpr uint dim = 3;
  using Model = Testing::ModelConstant<dim, dim>;
  using NumberType = double;
  using Discretization = CG::Discretization<typename Model::Components, NumberType, RectangularMesh<dim>>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = CG::Assembler<Discretization, Model>;

  JSONValue json = json::value(
      {{"physical", {}},
       {"integration",
        {{"x_quadrature_order", 32},
         {"angle_quadrature_order", 8},
         {"x0_quadrature_order", 16},
         {"x0_summands", 8},
         {"q0_quadrature_order", 16},
         {"q0_summands", 8},
         {"x_extent_tolerance", 1e-3},
         {"x0_extent_tolerance", 1e-3},
         {"q0_extent_tolerance", 1e-3},
         {"jacobian_quadrature_factor", 0.5}}},
       {"discretization",
        {{"fe_order", GENERATE(1, 2, 3)},
         {"threads", 8},
         {"batch_size", 64},
         {"overintegration", 0},
         {"output_subdivisions", 2},

         {"EoM_abs_tol", 1e-14},
         {"EoM_max_iter", 200},

         {"grid", {{"x_grid", "0:0.1:1"}, {"y_grid", "0:0.1:1"}, {"z_grid", "0:0.1:1"}, {"refine", 0}}},
         {"adaptivity",
          {{"start_adapt_at", 0.},
           {"adapt_dt", 1e-1},
           {"level", 0},
           {"refine_percent", 1e-1},
           {"coarsen_percent", 5e-2}}}}},
       {"timestepping",
        {{"final_time", 1.},
         {"output_dt", 1e-1},
         {"explicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}},
         {"implicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-14}, {"rel_tol", 1e-8}}}}},
       {"output", {{"live_plot", false}, {"verbosity", 0}}}});

  Testing::PhysicalParameters p_prm;
  for (uint d = 0; d < dim; ++d)
    p_prm.initial_x0[d] = -1.;

  p_prm.initial_x1[0] = GENERATE(take(2, random(1.1, 10.)));
  p_prm.initial_x1[1] = GENERATE(take(2, random(1.1, 10.)));
  p_prm.initial_x1[2] = GENERATE(take(2, random(1.1, 10.)));

  dealii::Point<dim> expected_EoM;
  for (uint d = 0; d < dim; ++d)
    expected_EoM[d] = -p_prm.initial_x0[d] / p_prm.initial_x1[d];

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
    log->info("DiFfRG Application started");
  } catch (const spdlog::spdlog_ex &e) {
    // nothing, the logger is already set up
  }

  // Define the objects needed to run the simulation
  Model model(p_prm);
  RectangularMesh<dim> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  // Set up the initial condition
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  const VectorType &src = initial_condition.spatial_data();

  const auto &dof_handler = discretization.get_dof_handler();
  const auto &mapping = discretization.get_mapping();

  auto EoM_cell = dof_handler.begin_active();

  const double EoM_abs_tol = json.get_double("/discretization/EoM_abs_tol");
  const uint EoM_max_iter = json.get_uint("/discretization/EoM_max_iter");

  const auto EoM = get_EoM_point(
      EoM_cell, src, dof_handler, mapping, [&](const auto &p, const auto &values) { return model.EoM(p, values); },
      [&](const auto &p, const auto &) { return p; }, EoM_abs_tol, EoM_max_iter);

  std::array<double, dim> error{{}};
  double max_error = 0.;
  for (uint d = 0; d < dim; ++d) {
    error[d] = std::abs(EoM[d] - expected_EoM[d]);
    max_error = std::max(max_error, error[d]);
  }

  if (max_error >= 1e-5) {
    std::cout << "ERROR: " << max_error << std::endl;
    std::cout << "EoM: " << EoM << " expected: " << expected_EoM << std::endl;
  }

  REQUIRE(max_error < 1e-5);
}