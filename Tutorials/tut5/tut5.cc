#include <DiFfRG/DiFfRG.hh>

#include "model.hh"

using namespace DiFfRG;

int main(int argc, char *argv[])
{
  // Initialize DiFfRG and thus the MPI and Kokkos environments
  const auto config_helper = DiFfRG::Init(argc, argv).get_configuration_helper();
  // get all needed parameters and parse from the CLI
  const auto json = config_helper.get_json();

  // Choices for types
  using Model = Tut5;
  constexpr uint dim = Model::dim;
  using Discretization = FV::Discretization<Model, RectangularMesh<dim>>;
  using Reconstructor = def::TVDReconstructor<dim, def::MinModLimiter, double>;
  using Assembler = FV::KurganovTadmor::Assembler<Discretization, Model, Reconstructor>;
  // IDA drives the stiff field-space sector; the Adams-Bashforth-Moulton stepper carries the
  // variable A explicitly, so its jacobian is never needed.
  using TimeStepper = TimeStepperSUNDIALS_IDA_BoostABM<Assembler>;

  // Define the objects needed to run the simulation
  Model model(json);
  RectangularMesh<dim> mesh{Config::ConfigurationMesh<dim>(json)};
  OutputSession<Assembler> data_out(json);
  const auto log = data_out.report_port();
  Discretization discretization(mesh, json, log);
  Assembler assembler(discretization, model, json);
  // Kurganov-Tadmor requires a fixed rectangular mesh; no h-adaptivity.
  NoAdaptivity mesh_adaptor(assembler);
  TimeStepper time_stepper(json, assembler, data_out, mesh_adaptor);

  // Set up the initial condition
  FV::FlowingVariables<Discretization> initial_condition(discretization);
  initial_condition.interpolate(model);

  // Now we start the timestepping
  Timer timer;
  try {
    time_stepper.run(initial_condition, 0., json.get_double("/timestepping/final_time"));
  } catch (std::exception &e) {
    log.error("Simulation finished with exception {}", e.what());
    return -1;
  }

  log.info("Simulation finished after " + time_format(timer.wall_time()));
  return 0;
}
