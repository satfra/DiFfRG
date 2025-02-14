#pragma once

// external libraries
#include "DiFfRG/discretization/mesh/h_adaptivity.hh"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

// DiFfRG
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/model/model.hh>
#include <DiFfRG/physics/physics.hh>
#include <DiFfRG/timestepping/timestepping.hh>

//--------------------------------------------
// Helper functions
//--------------------------------------------

template <typename Model, typename Discretization, typename Assembler, typename TimeStepper, bool expl = false>
bool run(std::string test_name, double expected_precision)
{
  using namespace dealii;
  using namespace DiFfRG;

  Testing::PhysicalParameters p_prm = {/*x0_initial = */ 0., /*x1_initial = */ 1.};

  JSONValue json = json::value(
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
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-16}, {"rel_tol", 1e-12}}},
         {"implicit",
          {{"dt", 1e-4}, {"minimal_dt", 1e-6}, {"maximal_dt", 1e-1}, {"abs_tol", 1e-16}, {"rel_tol", 1e-12}}}}},
       {"output", {{"verbosity", 0}}}});

  const double final_time = json.get_double("/timestepping/final_time");

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
  RectangularMesh<dim> mesh(json);
  Discretization discretization(mesh, json);
  Assembler assembler(discretization, model, json);
  DataOutput<dim, VectorType> data_out("./", test_name, test_name + '/', json);
  HAdaptivity mesh_adaptor(assembler, json);
  TimeStepper time_stepper(json, &assembler, &data_out, &mesh_adaptor);

  // Set up the initial condition
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  // Now we start the timestepping
  try {
    if constexpr (expl)
      time_stepper.run_explicit(&initial_condition, 0., final_time);
    else
      time_stepper.run(&initial_condition, 0., final_time);
  } catch (std::exception &e) {
    std::cout << "Simulation finished with exception " << e.what() << std::endl;
    return false;
  }

  // Validate the results
  bool valid = true;

  const auto &support_points = discretization.get_support_points();
  model.set_time(final_time);
  for (uint i = 0; i < support_points.size(); ++i) {
    if (!is_close(model.solution(support_points[i]), initial_condition.data()[i], expected_precision))
      std::cout << "is: " << model.solution(support_points[i]) << " should be: " << initial_condition.data()[i]
                << std::endl;
    valid &= is_close(model.solution(support_points[i]), initial_condition.data()[i], expected_precision);
  }

  if (!valid) std::cerr << "Failed " << test_name << std::endl;
  return valid;
}