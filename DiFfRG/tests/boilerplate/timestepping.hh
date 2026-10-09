#pragma once

// external libraries
#include "DiFfRG/discretization/mesh/h_adaptivity.hh"
#include "DiFfRG/discretization/mesh/no_adaptivity.hh"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <type_traits>
#include <unordered_set>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>
#include <DiFfRG/timestepping/timestepping.hh>

//--------------------------------------------
// Helper functions
//--------------------------------------------

template <typename Discretization, typename = void> struct UsesFVFlowingVariables : std::false_type {
};

template <typename Discretization>
struct UsesFVFlowingVariables<Discretization, std::void_t<decltype(Discretization::is_fv_discretization)>>
    : std::bool_constant<Discretization::is_fv_discretization> {
};

template <typename Discretization>
using FlowingVariablesFor =
    std::conditional_t<UsesFVFlowingVariables<Discretization>::value, DiFfRG::FV::FlowingVariables<Discretization>,
                       DiFfRG::FE::FlowingVariables<Discretization>>;

template <typename Model, typename Discretization, typename Assembler, typename TimeStepper, bool expl = false,
          bool adapt = false>
bool run(std::string test_name, double expected_precision, const std::string &required_diagnostic_table = {},
         const double expected_stepper_kind = std::numeric_limits<double>::quiet_NaN(),
         const std::vector<double> &required_stages = {}, const bool jacobian_diagnostics = true)
{
  using namespace dealii;
  using namespace DiFfRG;

  Testing::PhysicalParameters p_prm;
  p_prm.initial_x0[0] = 0.;
  p_prm.initial_x1[0] = 1.;
  p_prm.initial_x2[0] = 0.5;

  ConfigTree config = json::value(
      {{"physical", {{"Lambda", 1.}}},
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
        {{"fe_order", 3},
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
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-16}, {"rel_tol", 1e-12}}},
         {"implicit",
          {{"dt", 1e-4},
           {"minimal_dt", 1e-6},
           {"maximal_dt", 1e-1},
           {"abs_tol", 1e-16},
           {"rel_tol", 1e-12},
           {"jacobian_diagnostics", jacobian_diagnostics}}}}},
       {"output", {{"verbosity", 0}, {"vtk", false}}}});

  const double final_time = config.get_double("/timestepping/final_time");

  try {
    auto log = spdlog::stdout_color_mt("log");
    log->set_pattern("log: [%v]");
    log->info("DiFfRG Application started");
  } catch (const spdlog::spdlog_ex &e) {
    // nothing, the logger is already set up
  }

  // Choices for types
  constexpr uint dim = Discretization::dim;
  using VectorType = typename Discretization::VectorType;

  // Define the objects needed to run the simulation
  Model model(p_prm);
  // The discretization names its mesh; a serial-only test pins RectangularMeshSerial there.
  typename Discretization::Mesh mesh{Config::ConfigurationMesh<dim>(config)};
  Discretization discretization(mesh, config);
  Assembler assembler(discretization, model, config);
  auto data_out_path = OutputPath::temporary(TemporaryRetention::remove_on_destruction, test_name, test_name);
  OutputSession<Assembler> data_out(data_out_path, config);

  const int n_components = Model::Components::count_fe_functions(0);

  std::unique_ptr<AbstractAdaptor<VectorType>> adaptor;
  if constexpr (adapt)
    adaptor = std::make_unique<HAdaptivity<Assembler>>(assembler, config);
  else
    adaptor = std::make_unique<NoAdaptivity<VectorType>>();

  TimeStepper time_stepper(config, assembler, data_out, *adaptor);

  // Set up the initial condition
  FlowingVariablesFor<Discretization> initial_condition(discretization);
  initial_condition.interpolate(model);

  // Now we start the timestepping
  try {
    if constexpr (expl)
      time_stepper.run_explicit(initial_condition, 0., final_time);
    else
      time_stepper.run(initial_condition, 0., final_time);
  } catch (std::exception &e) {
    std::cout << "Simulation finished with exception " << e.what() << std::endl;
    return false;
  }

  // Validate the results
  bool valid = true;

  const auto table_path = data_out_path.root() / (data_out_path.run_name() + "_" + required_diagnostic_table);

  if (!required_diagnostic_table.empty() && !jacobian_diagnostics) {
    // Switching the diagnostics off has to suppress the table itself, not just its contents.
    valid &= !std::filesystem::exists(table_path);
  } else if (!required_diagnostic_table.empty()) {
    std::ifstream table(table_path);
    std::string header_line, data_line;
    const bool has_header = static_cast<bool>(std::getline(table, header_line));
    const bool has_data = static_cast<bool>(std::getline(table, data_line));
    valid &= has_header && has_data;

    std::unordered_set<std::string> columns;
    std::vector<std::string> ordered_columns;
    std::istringstream header(header_line);
    for (std::string column; std::getline(header, column, ',');) {
      columns.insert(column);
      ordered_columns.push_back(column);
    }
    constexpr std::array required_columns = {"t",
                                             "jacobian_build_id",
                                             "stepper_kind",
                                             "stage",
                                             "step",
                                             "retry_index",
                                             "ida_step",
                                             "alpha",
                                             "beta",
                                             "residual_weight",
                                             "last_h",
                                             "current_h",
                                             "n_rows",
                                             "nnz",
                                             "max_abs_entry",
                                             "frobenius_norm",
                                             "one_norm",
                                             "infinity_norm",
                                             "min_abs_diagonal",
                                             "max_abs_diagonal",
                                             "zero_diagonal_count",
                                             "min_diagonal_dominance",
                                             "min_row_max_abs",
                                             "max_row_max_abs",
                                             "min_column_max_abs",
                                             "max_column_max_abs",
                                             "factorization_ms",
                                             "factorization_success",
                                             "scaled_rcond_estimate"};
    for (const auto *column : required_columns)
      valid &= columns.contains(column);

    if (has_data && std::isfinite(expected_stepper_kind)) {
      const auto column_index = [&](const std::string &name) {
        return static_cast<std::size_t>(
            std::distance(ordered_columns.begin(), std::find(ordered_columns.begin(), ordered_columns.end(), name)));
      };
      const auto kind_index = column_index("stepper_kind");
      const auto stage_index = column_index("stage");
      const auto build_id_index = column_index("jacobian_build_id");
      const auto alpha_index = column_index("alpha");
      const auto beta_index = column_index("beta");
      const auto residual_weight_index = column_index("residual_weight");
      const auto current_h_index = column_index("current_h");
      std::unordered_set<unsigned int> observed_stages;
      double previous_build_id = -1.;

      do {
        std::vector<double> values;
        std::istringstream row(data_line);
        for (std::string value; std::getline(row, value, ',');)
          values.push_back(std::stod(value));
        valid &= values.size() == ordered_columns.size();
        if (values.size() != ordered_columns.size()) continue;

        valid &= values[kind_index] == expected_stepper_kind;
        valid &= values[build_id_index] > previous_build_id;
        previous_build_id = values[build_id_index];
        observed_stages.insert(static_cast<unsigned int>(values[stage_index]));

        if (expected_stepper_kind == static_cast<double>(ImplicitTimestepperKind::implicit_euler)) {
          valid &= values[alpha_index] == 1.;
          valid &= values[beta_index] == 1.;
          valid &= is_close(values[residual_weight_index], values[current_h_index], 1.e-12);
        } else if (expected_stepper_kind == static_cast<double>(ImplicitTimestepperKind::trbdf2)) {
          const double gamma = 2. - std::sqrt(2.);
          const auto stage = static_cast<ImplicitTimestepperStage>(static_cast<unsigned int>(values[stage_index]));
          if (stage == ImplicitTimestepperStage::trapezoidal) {
            valid &= values[alpha_index] == 1.;
            valid &= values[beta_index] == 1.;
            valid &= is_close(values[residual_weight_index] / values[current_h_index], gamma / 2., 1.e-5);
          } else if (stage == ImplicitTimestepperStage::bdf2) {
            valid &= is_close(values[alpha_index], 2. - gamma, 1.e-5);
            valid &= is_close(values[beta_index], 2. - gamma, 1.e-5);
            valid &= is_close(values[residual_weight_index] / values[current_h_index], 1. - gamma, 1.e-5);
          }
        }
      } while (std::getline(table, data_line));

      for (const double required_stage : required_stages)
        valid &= observed_stages.contains(static_cast<unsigned int>(required_stage));
    }

    if (!valid) std::cerr << "Invalid diagnostic table " << table_path << std::endl;
  }

  const auto &support_points = discretization.get_support_points();
  model.set_time(final_time);
  const uint n_points = initial_condition.spatial_data().size() / n_components;
  for (uint i = 0; i < n_points; ++i) {
    std::array<double, n_components> analytical_solution = model.solution(support_points[i * n_components]);
    for (uint component = 0; component < n_components; component++) {
      double numeric_solution = initial_condition.spatial_data()[n_components * i + component];
      const auto &error_condition = is_close(numeric_solution, analytical_solution[component], expected_precision);
      if (!error_condition && 1) {
        const auto abs_error = std::abs(numeric_solution - analytical_solution[component]);
        std::cout << std::setprecision(17) << "at x = " << support_points[i * n_components]
                  << " component: " << component << " numerical: " << numeric_solution
                  << " analytical: " << analytical_solution[component] << " abs_error: " << abs_error
                  << " tolerance: " << expected_precision << std::endl;
      }
      valid &= error_condition;
    }
  }

  if (!valid) std::cerr << "Failed " << test_name << std::endl;
  return valid;
}
