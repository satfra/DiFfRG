#include <filesystem>

#include <DiFfRG/common/csv_reader.hh>
#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/root_finding.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include "model.hh"

using namespace dealii;
using namespace DiFfRG;

// Choices for types
using Model = FourFermi;
using Assembler = Variables::Assembler<Model>;
using TimeStepper = TimeStepperBoostABM<Assembler>;

int main(int argc, char *argv[])
{
  Timer timer;

  // Initialize DiFfRG and thus the MPI and Kokkos environments
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  // get all needed parameters and parse from the CLI
  auto json = config_helper.get_json();

  // Define the objects needed to run the simulation
  OutputSession<Assembler> data_out(json);
  const auto log = data_out.report_port();
  Model model(json, log);
  Assembler assembler(model, json);
  TimeStepper time_stepper(json, assembler, data_out);

  // Set up the initial condition
  FlowingVariables initial_condition;
  initial_condition.interpolate(model);

  // Start the timestepping
  try {
    time_stepper.run(initial_condition, 0., json.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Timestepping finished with exception {}", e.what());
    log.flush();
  }

  // We print a bit of exit information.
  const auto time = timer.wall_time();
  log.info("Program finished after " + time_format(time));
}
