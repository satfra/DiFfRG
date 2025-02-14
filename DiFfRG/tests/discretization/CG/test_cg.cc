#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/models.hh>

#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>

//--------------------------------------------
// Test logic
//--------------------------------------------

TEST_CASE("Test CG on Constant model", "[discretization][dg]")
{
  using namespace dealii;
  using namespace DiFfRG;

  constexpr uint dim = 1;
  using Model = Testing::ModelConstant<dim>;
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
        {{"fe_order", GENERATE(1, 3, 5)},
         {"threads", 8},
         {"batch_size", 64},
         {"overintegration", 0},
         {"output_subdivisions", 2},

         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},

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

  Testing::PhysicalParameters p_prm = {/*x0_initial = */ 0., /*x1_initial = */ GENERATE(take(5, random(1., 10.)))};

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

  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using InverseSparseMatrixType = typename get_type::InverseSparseMatrixType<SparseMatrixType>;
  const SparseMatrixType &mass_matrix = assembler.get_mass_matrix();
  InverseSparseMatrixType inverse_mass_matrix;
  inverse_mass_matrix.initialize(mass_matrix);
  double random_value = GENERATE(take(5, random(1., 10.)));
  VectorType dst(src);
  SparseMatrixType dst_mat(assembler.get_sparsity_pattern_jacobian());
  // lose a digit due to multiplication
  const double expected_precision = std::numeric_limits<double>::epsilon() * 100.;

  auto vector_is_close = [expected_precision](const auto &v1, const auto &v2, const double factor = 1.) {
    bool valid = true;
    for (uint i = 0; i < v1.size(); ++i)
      valid &= is_close(v1[i], v2[i] * factor, expected_precision);
    return valid;
  };

  auto matrix_is_close = [expected_precision](const auto &m1, const auto &m2, const double factor = 1.) {
    bool valid = true;
    for (uint m = 0; m < m1.m(); ++m)
      for (uint n = 0; n < m1.n(); ++n)
        valid &= is_close(m1.el(m, n), m2.el(m, n) * factor, expected_precision);
    return valid;
  };

  SECTION("Test mass", "[mass]")
  {
    dst = 0;
    assembler.mass(dst, src, src, random_value);
    inverse_mass_matrix.solve(dst);
    REQUIRE(vector_is_close(dst, src, random_value));
  }
  SECTION("Test jacobian mass", "[mass][jacobian][!shouldfail]")
  {
    dst_mat = 0;
    assembler.jacobian_mass(dst_mat, src, src, 1., 0.);
    REQUIRE(matrix_is_close(dst_mat, mass_matrix));

    dst_mat = 0;
    assembler.jacobian_mass(dst_mat, src, src, 0., 1.);
    REQUIRE(matrix_is_close(dst_mat, mass_matrix, 0.));
  }

  SECTION("Test residual", "[residual]")
  {
    dst = 0;
    assembler.residual(dst, src, random_value, src, 0.);
    inverse_mass_matrix.solve(dst);
    REQUIRE(vector_is_close(dst, src, 0.));

    dst = 0;
    assembler.residual(dst, src, 0., src, random_value);
    inverse_mass_matrix.solve(dst);
    REQUIRE(vector_is_close(dst, src, random_value));

    dst = 0;
    assembler.residual(dst, src, random_value, src, random_value);
    inverse_mass_matrix.solve(dst);
    REQUIRE(vector_is_close(dst, src, random_value));
  }

  SECTION("Test jacobian", "[jacobian]")
  {
    dst_mat = 0;
    assembler.jacobian(dst_mat, src, random_value, src, 0., 0.);
    REQUIRE(matrix_is_close(dst_mat, mass_matrix, 0.));

    dst_mat = 0;
    assembler.jacobian(dst_mat, src, 0., src, random_value, 0.);
    REQUIRE(matrix_is_close(dst_mat, mass_matrix, random_value));

    dst_mat = 0;
    assembler.jacobian(dst_mat, src, 0., src, 0., random_value);
    REQUIRE(matrix_is_close(dst_mat, mass_matrix, 0.));
  }
}