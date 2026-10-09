#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <boilerplate/benchmark_sweep.hh>
#include <boilerplate/models.hh>

#include <DiFfRG/common/types.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/FEM/ldg.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>

using namespace dealii;
using namespace DiFfRG;

TEST_CASE("Benchmark LDG Constant", "[benchmark][ldg]")
{
  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
  } catch (const spdlog::spdlog_ex &) {
  }

  constexpr uint dim = 1;
  using Model = Testing::LDGModelConstant<dim>;
  using NumberType = double;
  using Discretization = LDG::Discretization<Model, RectangularMeshSerial<dim>, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using Assembler = LDG::Assembler<Discretization>;

  const int fe_order = GENERATE_COPY(from_range(Testing::benchmark_fe_orders({0, 1, 3, 5})));

  ConfigTree json = json::value(
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
        {{"fe_order", fe_order},
         {"overintegration", 0},
         {"output_subdivisions", 2},
         {"EoM_abs_tol", 1e-10},
         {"EoM_max_iter", 0},
         {"grid", {{"x_grid", "0:0.01:1"}, {"y_grid", "0:0.01:1"}, {"z_grid", "0:0.01:1"}, {"refine", 0}}},
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

  Testing::PhysicalParameters p_prm = {/*x0_initial = */ {0., 0., 0.}, /*x1_initial = */ {3.14, 0., 0.}};
  std::string label = "LDG p=" + std::to_string(fe_order);

  Model model(p_prm);
  Testing::apply_benchmark_sweep(json);

  RectangularMeshSerial<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);

  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);
  const VectorType &src = initial_condition.spatial_data();

  using SparseMatrixType = typename Discretization::SparseMatrixType;
  VectorType dst(src);
  SparseMatrixType dst_mat(assembler.get_sparsity_pattern_jacobian());

  BENCHMARK_ADVANCED(label + " mass")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      dst = 0;
      assembler.mass(dst, src, src, 1.0);
    });
  };

  BENCHMARK_ADVANCED(label + " residual")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      dst = 0;
      assembler.residual(dst, src, 1.0, src, 1.0);
    });
  };

  BENCHMARK_ADVANCED(label + " jacobian_mass")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      dst_mat = 0;
      assembler.jacobian_mass(dst_mat, src, src, 1.0, 1.0);
    });
  };

  // LDG jacobian requires weight != 0
  BENCHMARK_ADVANCED(label + " jacobian")(Catch::Benchmark::Chronometer meter)
  {
    meter.measure([&] {
      dst_mat = 0;
      assembler.jacobian(dst_mat, src, 1.0, src, 1.0, 1.0);
    });
  };
}
