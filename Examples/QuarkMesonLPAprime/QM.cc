#include <DiFfRG/common/init.hh>
#include <DiFfRG/common/utils.hh>
#include <DiFfRG/discretization/discretization.hh>
#include <DiFfRG/timestepping/timestepping.hh>

#include "model.hh"

using namespace dealii;
using namespace DiFfRG;

// Choices for types
using Model = QuarkMesonLPAprime;
constexpr uint dim = Model::dim;
using Discretization = CG::Discretization<Model, RectangularMesh<dim>>;
using Assembler = CG::Assembler<Discretization>;
using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<Assembler>;

int main(int argc, char *argv[])
{
  // Initialize DiFfRG and thus the MPI and Kokkos environments
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  // get all needed parameters and parse from the CLI
  const auto json = config_helper.get_json();

  // Define the objects needed to run the simulation
  Model model(json);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  OutputSession<Assembler> data_out(json);
  const auto log = data_out.report_port();
  Discretization discretization(mesh, json, log);
  Assembler assembler(discretization, model, json);
  TimeStepper time_stepper(json, assembler, data_out);

  // Set up the initial condition
  FE::FlowingVariables initial_condition(discretization);
  initial_condition.interpolate(model);

  // Now we start the timestepping
  Timer timer;
  try {
    time_stepper.run(initial_condition, 0., json.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Simulation finished with exception {}", e.what());
    return -1;
  }

  // We print a bit of exit information.
  const auto time = timer.wall_time();
  log.info("Simulation finished after " + time_format(time));
  return 0;
}
