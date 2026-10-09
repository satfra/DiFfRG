#include "kt_regression_helpers.hh"

#include "DiFfRG/discretization/FV/assembler/KurganovTadmor.hh"
#include "DiFfRG/discretization/FV/discretization.hh"
#include "DiFfRG/discretization/mesh/no_adaptivity.hh"
#include "DiFfRG/timestepping/timestepping.hh"

#include <DiFfRG/common/config_tree.hh>
#include <DiFfRG/discretization/data/output_session.hh>
#include <DiFfRG/discretization/mesh/rectangular_mesh.hh>
#include <DiFfRG/model/model.hh>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  using namespace DiFfRG;
  using namespace dealii;
  using std::get;

  constexpr uint dim = 2;
  constexpr std::size_t n_components = 1;
  constexpr unsigned int n_coarse_cells = 50;
  constexpr unsigned int n_medium_cells = 100;
  constexpr unsigned int n_fine_cells = 200;
  constexpr double domain_min = -1.5;
  constexpr double domain_max = 1.5;
  constexpr double domain_length = domain_max - domain_min;
  constexpr double final_time = 0.5;
  constexpr double disk_radius_squared = 0.5;
  constexpr double diffusion_epsilon = 0.01;
  constexpr double implicit_dt = 1.0e-2;
  // IDA's FIRST step must be far below its largest. The initial condition is a discontinuous
  // disk, and opening with a 1e-2 step drives the nonlinear solve into repeated rejections.
  constexpr double implicit_initial_dt = 1.0e-5;
  constexpr double medium_l1_refinement_factor = 0.75;
  constexpr double medium_l2_refinement_factor = 0.85;
  constexpr double min_tolerance = -5.0e-2;
  constexpr double max_tolerance = 1.05;
  constexpr double mass_tolerance = 5.0e-3;

  using NumberType = double;
  using FEFunctionDesc = FEFunctionDescriptor<Scalar<"u">>;
  using Components = ComponentDescriptor<FEFunctionDesc>;
  using Mesh = RectangularMeshSerial<dim>;
  using Discretization = FV::Discretization<Components, Mesh, NumberType>;
  using VectorType = typename Discretization::VectorType;
  using SparseMatrixType = typename Discretization::SparseMatrixType;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, class BuckleyLeverett2DExample12Model, Reconstructor>;
  using TimeStepper = TimeStepperSUNDIALS_IDA<Assembler>;

  double initial_value(const Point<dim> &pos) { return pos.square() < disk_radius_squared ? 1.0 : 0.0; }

  class BuckleyLeverett2DExample12Model : public def::AbstractModel<BuckleyLeverett2DExample12Model, Components>,
                                          public def::Time,
                                          public def::LLFFlux<BuckleyLeverett2DExample12Model>,
                                          public def::FlowBoundaries<BuckleyLeverett2DExample12Model>,
                                          public def::FVDefaultBoundaries<BuckleyLeverett2DExample12Model>,
                                          public def::AD<BuckleyLeverett2DExample12Model>
  {
  public:
    template <typename Vector> void initial_condition(const Point<dim> &pos, Vector &values) const
    {
      values[0] = initial_value(pos);
    }

    template <typename NT, typename Solution>
    void flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
              const Solution &sol) const
    {
      const auto &u = get<0>(sol);
      const auto u_value = u[0];
      const auto one_minus_u = NT(1.0) - u_value;
      const auto u_squared = u_value * u_value;
      const auto one_minus_u_squared = one_minus_u * one_minus_u;
      const auto f = u_squared / (u_squared + one_minus_u_squared);

      F_i[0][0] = f;
      F_i[0][1] = f * (NT(1.0) - NT(5.0) * one_minus_u_squared);
    }

    template <typename NT, typename Solution>
    void diffusion_flux(std::array<Tensor<1, dim, NT>, n_components> &F_i, [[maybe_unused]] const Point<dim> &pos,
                        const Solution &sol) const
    {
      const auto &fe_derivatives = get<1>(sol);
      F_i[0] = -diffusion_epsilon * fe_derivatives[0];
    }

    template <typename NT, typename Solution>
    void source(std::array<NT, n_components> &s_i, [[maybe_unused]] const Point<dim> &pos,
                [[maybe_unused]] const Solution &sol) const
    {
      s_i[0] = NT(0.0);
    }
  };

  std::string grid_axis_string(const unsigned int n_cells)
  {
    std::ostringstream stream;
    stream << std::setprecision(17) << domain_min << ':' << domain_length / static_cast<double>(n_cells) << ':'
           << domain_max;
    return stream.str();
  }

  ConfigTree make_run_json(const unsigned int n_cells)
  {
    const auto axis = grid_axis_string(n_cells);
    return json::value(
        {{"physical", {{"Lambda", 1.0}}},
         {"discretization",
          {{"fe_order", 0},
           {"overintegration", 0},
           {"output_subdivisions", 1},
           {"EoM_abs_tol", 1.0e-10},
           {"EoM_max_iter", 0},
           {"grid", {{"x_grid", axis}, {"y_grid", axis}, {"z_grid", "0:1:1"}, {"refine", 0}}}}},
         {"timestepping",
          {{"final_time", final_time},
           {"output_dt", final_time},
           {"explicit",
            {{"dt", 1.0e-3}, {"minimal_dt", 1.0e-8}, {"maximal_dt", 1.0e-3}, {"abs_tol", 1.0e-9}, {"rel_tol", 1.0e-9}}},
           {"implicit",
            {{"dt", implicit_initial_dt},
             {"minimal_dt", 1.0e-8},
             {"maximal_dt", implicit_dt},
             {"abs_tol", 1.0e-6},
             {"rel_tol", 1.0e-5}}}}},
         {"output", {{"verbosity", 1}, {"vtk", false}, {"hdf5", false}}}});
  }

  Config::ConfigurationMesh<dim> make_mesh_config(const unsigned int n_cells)
  {
    const double dx = domain_length / static_cast<double>(n_cells);
    const std::vector<Config::GridAxis> axis{Config::GridAxis(domain_min, dx, domain_max)};
    return Config::ConfigurationMesh<dim>(0u, axis, axis);
  }

  std::size_t row_major_cell_index(const Point<dim> &pos, const unsigned int n_cells)
  {
    const double dx = domain_length / static_cast<double>(n_cells);
    const auto index = [dx, n_cells](const double coordinate) {
      const auto raw = static_cast<int>(std::floor((coordinate - domain_min) / dx));
      return static_cast<unsigned int>(std::clamp(raw, 0, static_cast<int>(n_cells) - 1));
    };
    const unsigned int ix = index(pos[0]);
    const unsigned int iy = index(pos[1]);
    return static_cast<std::size_t>(iy) * n_cells + ix;
  }

  std::vector<double> row_major_cell_values(const Discretization &discretization, const VectorType &values,
                                            const unsigned int n_cells)
  {
    std::vector<double> result(static_cast<std::size_t>(n_cells) * n_cells, std::numeric_limits<double>::quiet_NaN());
    std::vector<unsigned int> counts(result.size(), 0);

    const auto &support_points = discretization.get_support_points();
    if (support_points.size() != values.size())
      throw std::runtime_error("Unexpected support-point/value size mismatch.");

    for (unsigned int dof = 0; dof < values.size(); ++dof) {
      const std::size_t row_major = row_major_cell_index(support_points[dof], n_cells);
      if (counts[row_major] != 0) throw std::runtime_error("Duplicate row-major cell index while sampling FV state.");
      result[row_major] = values[dof];
      ++counts[row_major];
    }

    for (const auto count : counts)
      if (count != 1) throw std::runtime_error("Missing row-major cell index while sampling FV state.");

    return result;
  }

  double cell_measure(const unsigned int n_cells)
  {
    const double dx = domain_length / static_cast<double>(n_cells);
    return dx * dx;
  }

  double integrated_mass(const std::vector<double> &values, const unsigned int n_cells)
  {
    const double measure = cell_measure(n_cells);
    double mass = 0.0;
    for (const double value : values)
      mass += measure * value;
    return mass;
  }

  struct RunResult {
    std::vector<double> initial_values;
    std::vector<double> final_values;
    double initial_mass = 0.0;
    double final_mass = 0.0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
  };

  RunResult run_buckley_leverett(const unsigned int n_cells, const std::string &run_name)
  {
    kt_regression::ensure_logger();

    const ConfigTree json = make_run_json(n_cells);
    BuckleyLeverett2DExample12Model model;
    Mesh mesh(make_mesh_config(n_cells));

    Discretization discretization(mesh, json);
    Assembler assembler(discretization, model, json);

    auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, run_name, "output");
    OutputSession<Discretization> data_out(data_out_path, json);
    auto adaptor = std::make_unique<NoAdaptivity<VectorType>>();
    TimeStepper time_stepper(json, assembler, data_out, *adaptor);

    FV::FlowingVariables<Discretization> state(discretization);
    state.interpolate(model);
    discretization.get_constraints().distribute(state.spatial_data());

    RunResult result;
    result.initial_values = row_major_cell_values(discretization, state.spatial_data(), n_cells);
    result.initial_mass = integrated_mass(result.initial_values, n_cells);

    time_stepper.run(state, 0.0, final_time);

    result.final_values = row_major_cell_values(discretization, state.spatial_data(), n_cells);
    result.final_mass = integrated_mass(result.final_values, n_cells);
    for (const double value : result.final_values) {
      result.min = std::min(result.min, value);
      result.max = std::max(result.max, value);
    }
    return result;
  }

  std::vector<double> average_to_grid(const std::vector<double> &fine_values, const unsigned int fine_cells,
                                      const unsigned int target_cells)
  {
    REQUIRE(fine_values.size() == static_cast<std::size_t>(fine_cells) * fine_cells);
    REQUIRE(fine_cells % target_cells == 0);

    const unsigned int ratio = fine_cells / target_cells;

    std::vector<double> target(static_cast<std::size_t>(target_cells) * target_cells, 0.0);
    std::vector<unsigned int> counts(target.size(), 0);

    for (unsigned int fine_y = 0; fine_y < fine_cells; ++fine_y) {
      const unsigned int target_y = fine_y / ratio;
      for (unsigned int fine_x = 0; fine_x < fine_cells; ++fine_x) {
        const unsigned int target_x = fine_x / ratio;
        const std::size_t fine_index = static_cast<std::size_t>(fine_y) * fine_cells + fine_x;
        const std::size_t target_index = static_cast<std::size_t>(target_y) * target_cells + target_x;
        target[target_index] += fine_values[fine_index];
        ++counts[target_index];
      }
    }

    for (std::size_t i = 0; i < target.size(); ++i) {
      REQUIRE(counts[i] == ratio * ratio);
      target[i] /= static_cast<double>(counts[i]);
    }
    return target;
  }

  std::filesystem::path reference_fixture_path()
  {
    return std::filesystem::path(__FILE__).parent_path() / "data" / "buckley_leverett_2D_reference_fine.json";
  }

  // Downsampled fine-grid reference, produced by the hidden [generate] test case below.
  // Regenerate with: ./buckley_leverett_2D "GENERATE 2D Buckley-Leverett fine-grid reference fixture"
  json::object load_reference_fixture()
  {
    const auto path = reference_fixture_path();
    if (!std::filesystem::exists(path))
      throw std::runtime_error("Missing 2D Buckley-Leverett reference fixture: " + path.string() +
                               " (regenerate with the [generate] test case)");
    const ConfigTree fixture(path.string());
    const json::value value = static_cast<json::value>(fixture);
    json::object root = value.as_object();
    REQUIRE(root.at("n_fine_cells").as_int64() == static_cast<std::int64_t>(n_fine_cells));
    REQUIRE(root.at("final_time").as_double() == final_time);
    return root;
  }

  std::vector<double> downsampled_reference(const json::object &root, const unsigned int target_cells)
  {
    const auto key = std::to_string(target_cells);
    if (!root.contains(key))
      throw std::runtime_error("Buckley-Leverett reference fixture has no entry for " + key + " cells");
    auto values = kt_regression::json_array_to_doubles(root.at(key).as_array());
    REQUIRE(values.size() == static_cast<std::size_t>(target_cells) * target_cells);
    return values;
  }

  struct ComparisonMetrics {
    double l1 = 0.0;
    double l2 = 0.0;
    double linf = 0.0;
  };

  ComparisonMetrics compute_comparison_metrics(const unsigned int n_cells, const std::vector<double> &numerical,
                                               const std::vector<double> &downsampled_fine)
  {
    REQUIRE(numerical.size() == static_cast<std::size_t>(n_cells) * n_cells);
    REQUIRE(downsampled_fine.size() == numerical.size());

    const double measure = cell_measure(n_cells);
    ComparisonMetrics metrics;
    for (std::size_t i = 0; i < numerical.size(); ++i) {
      const double error = numerical[i] - downsampled_fine[i];
      const double abs_error = std::abs(error);
      metrics.l1 += measure * abs_error;
      metrics.l2 += measure * error * error;
      metrics.linf = std::max(metrics.linf, abs_error);
    }
    metrics.l2 = std::sqrt(metrics.l2);
    return metrics;
  }

  void require_finite_metrics(const ComparisonMetrics &metrics)
  {
    REQUIRE(std::isfinite(metrics.l1));
    REQUIRE(std::isfinite(metrics.l2));
    REQUIRE(std::isfinite(metrics.linf));
  }

  void check_run_invariants(const RunResult &run)
  {
    REQUIRE(std::isfinite(run.initial_mass));
    REQUIRE(std::isfinite(run.final_mass));
    REQUIRE(std::isfinite(run.min));
    REQUIRE(std::isfinite(run.max));
    CHECK(run.min >= min_tolerance);
    CHECK(run.max <= max_tolerance);
    CHECK(std::abs(run.final_mass - run.initial_mass) < mass_tolerance);
  }
} // namespace

// Regenerates data/buckley_leverett_2D_reference_fine.json. Hidden ("[.]") so it never runs in a
// normal pass: it performs the expensive fine-grid solve that the benchmark itself avoids, and it
// is where the fine run's own invariants are asserted.
TEST_CASE("GENERATE 2D Buckley-Leverett fine-grid reference fixture", "[.][generate][2d][FV][KT][buckley-leverett]")
{
  const auto fine = run_buckley_leverett(n_fine_cells, "buckley_leverett_2D_fine_grid");
  check_run_invariants(fine);

  json::object root;
  root["n_fine_cells"] = static_cast<std::int64_t>(n_fine_cells);
  root["final_time"] = final_time;
  root["comment"] = "Fine-grid 2D Buckley-Leverett run, cell-averaged onto the coarser ladder grids. "
                    "Generated by the [generate] test case in buckley_leverett_2D.cc.";
  for (const unsigned int target : {n_coarse_cells, n_medium_cells}) {
    const auto downsampled = average_to_grid(fine.final_values, n_fine_cells, target);
    json::array entries;
    entries.reserve(downsampled.size());
    for (const double value : downsampled)
      entries.push_back(json::value(value));
    root[std::to_string(target)] = std::move(entries);
  }

  const auto path = reference_fixture_path();
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write reference fixture: " + path.string());
  out << json::serialize(json::value(std::move(root))) << '\n';
  out.close();
  std::cout << "Wrote 2D Buckley-Leverett reference fixture to " << path << '\n';
  REQUIRE(std::filesystem::exists(path));
}

TEST_CASE("2D Buckley-Leverett Example 12 refines toward the fine-grid run", "[2d][FV][KT][buckley-leverett][slow]")
{
  // The fine grid is a committed fixture; its invariants are asserted by the [generate] case.
  const json::object fixture = load_reference_fixture();

  const auto downsampled_fine_coarse = downsampled_reference(fixture, n_coarse_cells);
  const auto coarse = run_buckley_leverett(n_coarse_cells, "buckley_leverett_2D_coarse_grid");
  const ComparisonMetrics coarse_metrics =
      compute_comparison_metrics(n_coarse_cells, coarse.final_values, downsampled_fine_coarse);

  const auto downsampled_fine_medium = downsampled_reference(fixture, n_medium_cells);
  const auto medium = run_buckley_leverett(n_medium_cells, "buckley_leverett_2D_medium_grid");
  const ComparisonMetrics medium_metrics =
      compute_comparison_metrics(n_medium_cells, medium.final_values, downsampled_fine_medium);

  std::cout << std::scientific << std::setprecision(8)
            << "2D Buckley-Leverett 50x50 vs downsampled 200x200: L1=" << coarse_metrics.l1
            << ", L2=" << coarse_metrics.l2 << ", Linf=" << coarse_metrics.linf
            << ", mass_error=" << coarse.final_mass - coarse.initial_mass << ", min=" << coarse.min
            << ", max=" << coarse.max << '\n'
            << "2D Buckley-Leverett 100x100 vs downsampled 200x200: L1=" << medium_metrics.l1
            << ", L2=" << medium_metrics.l2 << ", Linf=" << medium_metrics.linf
            << ", mass_error=" << medium.final_mass - medium.initial_mass << ", min=" << medium.min
            << ", max=" << medium.max << '\n'
            << "2D Buckley-Leverett fine reference: committed fixture (" << n_fine_cells << "x" << n_fine_cells
            << ")\n";

  require_finite_metrics(coarse_metrics);
  require_finite_metrics(medium_metrics);

  CHECK(medium_metrics.l1 < medium_l1_refinement_factor * coarse_metrics.l1);
  CHECK(medium_metrics.l2 < medium_l2_refinement_factor * coarse_metrics.l2);

  check_run_invariants(coarse);
  check_run_invariants(medium);
}
